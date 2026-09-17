"""Proximal Policy Optimization (PPO) Trainer.

Implements the clipped surrogate objective, value function loss, entropy bonus,
and diagnostic tracking for training neural and connectome policies in MuJoCo.
Reference: Schulman et al., "Proximal Policy Optimization Algorithms" (2017).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from connectome_rl.src.rl.buffer import RolloutBuffer


@dataclass
class PPOConfig:
    """Hyperparameter configuration for PPO optimization."""

    learning_rate: float = 3e-4
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    adam_eps: float = 1e-5


class PPOTrainer:
    """Trainer implementing Clipped Proximal Policy Optimization (PPO).

    Supports:
      - Standard MLP Actor-Critic models.
      - Connectome-constrained MaskedLinear policies (preserves zero-weight biological constraints).
      - Connectome-RNN, Synaptic-GNN, and Hierarchical Brain-VNC controllers.
    """

    def __init__(
        self,
        policy: nn.Module,
        config: PPOConfig | None = None,
        optimizer: optim.Optimizer | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        """Initialize PPO Trainer.

        Args:
            policy: Actor-Critic policy module implementing `get_action_and_value`.
            config: PPOConfig dataclass with hyperparameters.
            optimizer: Custom PyTorch optimizer. If None, instantiates Adam with config.learning_rate.
            device: Target execution device (CPU or CUDA).
        """
        self.policy = policy
        self.config = config or PPOConfig()
        self.device = torch.device(device)
        self.policy.to(self.device)

        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = optim.Adam(
                self.policy.parameters(),
                lr=self.config.learning_rate,
                eps=self.config.adam_eps,
            )

    def train_step(
        self,
        buffer: RolloutBuffer,
        batch_size: int = 64,
        update_epochs: int = 10,
    ) -> dict[str, float]:
        """Execute PPO optimization over the collected rollout buffer.

        Args:
            buffer: RolloutBuffer populated with experience and computed GAE advantages.
            batch_size: Mini-batch size for SGD updates.
            update_epochs: Number of complete passes over the rollout buffer.

        Returns:
            Dictionary containing averaged training diagnostics:
              - 'policy_loss': Clipped surrogate policy objective.
              - 'value_loss': Mean squared error critic loss.
              - 'entropy': Policy action entropy (exploration measure).
              - 'approx_kl': Approximate Kullback-Leibler divergence.
              - 'clip_fraction': Proportion of samples clipped by ε.
              - 'explained_variance': Fraction of return variance explained by V(s).
        """
        self.policy.train()

        pg_losses: list[float] = []
        v_losses: list[float] = []
        entropy_losses: list[float] = []
        approx_kls: list[float] = []
        clip_fractions: list[float] = []
        explained_vars: list[float] = []

        early_stopped = False

        for epoch in range(update_epochs):
            if early_stopped:
                break

            for batch in buffer.get_generator(batch_size=batch_size, normalize_advantages=True):
                obs = batch["obs"].to(self.device)
                actions = batch["actions"].to(self.device)
                old_log_probs = batch["log_probs"].to(self.device)
                advantages = batch["advantages"].to(self.device)
                returns = batch["returns"].to(self.device)
                old_values = batch["values"].to(self.device)

                # 1. Forward pass: evaluate current policy on the collected actions
                out = self.policy.get_action_and_value(obs, action=actions)
                _, new_log_prob, entropy, new_value = out[:4]
                new_value = new_value.view(-1)

                # 2. Probability ratio: r_t(θ) = π_θ(a_t | s_t) / π_{θ_old}(a_t | s_t)
                log_ratio = new_log_prob - old_log_probs
                ratio = torch.exp(log_ratio)

                # 3. Approximate KL divergence: KL(π_{old} || π) ≈ (r - 1) - log(r)
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
                    approx_kls.append(approx_kl)

                # Early stopping if policy changes too drastically
                if self.config.target_kl is not None and approx_kl > 1.5 * self.config.target_kl:
                    early_stopped = True
                    break

                # 4. Clipped Surrogate Policy Loss
                # L^{CLIP}(θ) = - min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)
                surr1 = -advantages * ratio
                surr2 = -advantages * torch.clamp(
                    ratio, 1.0 - self.config.clip_coef, 1.0 + self.config.clip_coef
                )
                policy_loss = torch.max(surr1, surr2).mean()
                pg_losses.append(policy_loss.item())

                # Track fraction of samples that triggered clipping
                with torch.no_grad():
                    clip_fraction = (torch.abs(ratio - 1.0) > self.config.clip_coef).float().mean().item()
                    clip_fractions.append(clip_fraction)

                # 5. Value Function Loss (with optional value clipping)
                if self.config.clip_vloss:
                    v_loss_unclipped = (new_value - returns) ** 2
                    v_clipped = old_values + torch.clamp(
                        new_value - old_values,
                        -self.config.clip_coef,
                        self.config.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - returns) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    value_loss = 0.5 * v_loss_max.mean()
                else:
                    value_loss = 0.5 * ((new_value - returns) ** 2).mean()

                v_losses.append(value_loss.item())

                # 6. Action Entropy Bonus (encourages exploratory coverage)
                ent_loss = entropy.mean()
                entropy_losses.append(ent_loss.item())

                # 7. Total Composite PPO Objective
                total_loss = (
                    policy_loss
                    - self.config.ent_coef * ent_loss
                    + self.config.vf_coef * value_loss
                )

                # 8. Gradient step with gradient clipping
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                # 9. Explained variance of the value function
                with torch.no_grad():
                    y_true = returns.cpu().numpy()
                    y_pred = new_value.cpu().numpy()
                    var_y = np.var(y_true)
                    exp_var = float(np.nan if var_y == 0 else 1.0 - np.var(y_true - y_pred) / var_y)
                    explained_vars.append(exp_var)

        return {
            "policy_loss": float(np.mean(pg_losses)) if pg_losses else 0.0,
            "value_loss": float(np.mean(v_losses)) if v_losses else 0.0,
            "entropy": float(np.mean(entropy_losses)) if entropy_losses else 0.0,
            "approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
            "clip_fraction": float(np.mean(clip_fractions)) if clip_fractions else 0.0,
            "explained_variance": float(np.mean(explained_vars)) if explained_vars else 0.0,
        }

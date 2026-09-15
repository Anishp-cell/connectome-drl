"""Unconstrained MLP Actor-Critic Policy (Baseline Model).

Standard deep reinforcement learning architecture with fully-connected (dense)
layers. Serves as the control baseline to benchmark connectome-constrained
architectures against.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    """Initialize linear layer weights orthogonally for stable RL training."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class MLPPolicy(nn.Module):
    """Standard unconstrained Multi-Layer Perceptron (MLP) Actor-Critic policy.
    
    Architecture:
      - Shared or separate 2-layer dense trunk (default: 256 hidden units, Tanh/ReLU).
      - Actor head: outputs action mean μ for each continuous actuator (dim = act_dim).
      - Learnable log standard deviation parameter for Gaussian exploration.
      - Critic head: outputs scalar state-value estimate V(s).
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 256,
        activation: str = "tanh",
        init_log_std: float = -0.5,
    ) -> None:
        """Initialize the MLP policy.
        
        Args:
            obs_dim: Dimension of observation vector (joint angles, velocities, etc.).
            act_dim: Dimension of action vector (42 leg joint torque/position targets).
            hidden_dim: Number of neurons per hidden layer (default: 256).
            activation: Non-linear activation function ('tanh' or 'relu').
            init_log_std: Initial value for log standard deviation (exploration noise).
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        
        act_fn = nn.Tanh if activation.lower() == "tanh" else nn.ReLU

        # Actor network (Policy / Motor Commands)
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            act_fn(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            act_fn(),
            layer_init(nn.Linear(hidden_dim, act_dim), std=0.01),  # Small initial actions
        )

        # Learnable log standard deviation for diagonal Gaussian policy
        self.actor_logstd = nn.Parameter(torch.ones(act_dim) * init_log_std)

        # Critic network (Value Function V(s))
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            act_fn(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            act_fn(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Estimate the state value V(s).
        
        Args:
            obs: Observation tensor of shape (batch_size, obs_dim).
            
        Returns:
            Scalar value estimates of shape (batch_size, 1).
        """
        return self.critic(obs)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute action distribution, sample action (or evaluate given action), and state value.
        
        Args:
            obs: Observation tensor of shape (batch_size, obs_dim).
            action: Optional existing action tensor for computing log probabilities
                   during PPO optimization epochs.
                   
        Returns:
            action: Sampled continuous actions of shape (batch_size, act_dim).
            log_prob: Log probability of the actions under current policy.
            entropy: Distribution entropy for exploration incentive.
            value: Estimated state value V(s).
        """
        action_mean = self.actor(obs)
        # Clamp log_std to prevent numerical overflow/underflow in Gaussian math
        action_logstd = torch.clamp(self.actor_logstd, min=-20.0, max=2.0)
        action_std = torch.exp(action_logstd)
        
        dist = Normal(action_mean, action_std)

        if action is None:
            # During rollout collection: sample new action
            action = dist.sample()

        # Sum log probabilities across all independent action dimensions (diagonal covariance)
        log_prob = dist.log_prob(action).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)
        value = self.critic(obs)

        return action, log_prob, entropy, value

    def get_deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Get the deterministic mean action for evaluation and rendering."""
        with torch.no_grad():
            return self.actor(obs)


if __name__ == "__main__":
    # Smoke test: create dummy observation and test forward pass
    torch.manual_seed(42)
    batch_size = 8
    dummy_obs_dim = 100   # Sample observation dimension
    dummy_act_dim = 42    # 42 FlyGym leg joints
    
    policy = MLPPolicy(obs_dim=dummy_obs_dim, act_dim=dummy_act_dim)
    dummy_obs = torch.randn(batch_size, dummy_obs_dim)
    
    action, log_prob, entropy, value = policy.get_action_and_value(dummy_obs)
    
    print("=" * 60)
    print("MLP POLICY (BASELINE MODEL) SMOKE TEST")
    print("=" * 60)
    print(f"Observation shape: {dummy_obs.shape}")
    print(f"Sampled action:    {action.shape} (Range: [{action.min():.2f}, {action.max():.2f}])")
    print(f"Log probability:   {log_prob.shape}")
    print(f"Entropy:           {entropy.shape} (Mean: {entropy.mean():.2f})")
    print(f"State value V(s):  {value.shape}")
    
    # Test gradient flow through actor and critic
    loss = value.mean() - log_prob.mean()
    loss.backward()
    
    has_grad = all(p.grad is not None for p in policy.parameters())
    print(f"Gradient flow verified: {has_grad}")
    print(f"Total Parameters: {sum(p.numel() for p in policy.parameters()):,}")
    print("=" * 60)

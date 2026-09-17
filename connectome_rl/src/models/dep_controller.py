"""Differential Extrinsic Plasticity (DEP) Exploration Controller.

Based on Der & Martius (2015) and Schmidt, Tourbier et al. (Nature Machine Intelligence 2023).

DEP is an online, self-organizing sensorimotor learning rule that discovers natural
coordinated multi-joint rhythms through physical interaction with the environment,
without requiring external reward shaping.

Core Mathematics:
  1. Sensor Smoothing:
     x_smooth_t = x_smooth_{t-1} + (s_t - x_smooth_{t-1}) / s4avg
  2. Velocity Correlation:
     chi = x_t - x_{t-1}            (current sensory difference)
     v   = x_{t-d} - x_{t-d-1}      (delayed sensory difference)
     mu  = M @ chi                  (internal model expectation)
     C  += mu ⊗ v                   (plastic correlation matrix)
  3. Action Output:
     q = C_norm @ x_smooth
     y = tanh(q * kappa + bias)
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import torch


class DEPController:
    """Differential Extrinsic Plasticity (DEP) Exploration Controller.
    
    Acts as a bio-inspired sensorimotor exploration mechanism for biomechanical
    bodies like FlyGym (42 leg joints). Instead of applying independent white noise
    to each joint (which leads to erratic flailing and tripping), DEP self-organizes
    coupled multi-joint synergies directly from proprioceptive physical feedback.
    """

    def __init__(
        self,
        act_dim: int = 42,
        sens_dim: int | None = None,
        kappa: float = 1.0,
        tau: int = 40,
        time_dist: int = 1,
        regularization: float = 0.01,
        bias_rate: float = 0.01,
        s4avg: int = 2,
        buffer_size: int = 100,
        device: str | torch.device = "cpu",
    ) -> None:
        """Initialize the DEP controller.
        
        Args:
            act_dim: Number of actuated joint degrees of freedom (42 for FlyGym).
            sens_dim: Number of proprioceptive sensors (defaults to act_dim).
            kappa: Feedback strength gain factor.
            tau: Time horizon window for correlation accumulation.
            time_dist: Delay steps between cause and effect in sensory feedback.
            regularization: Regularization constant preventing numerical singularity.
            bias_rate: Adaptive drift rate for controller bias to prevent motor locking.
            s4avg: Exponential rolling average smoothing window for sensor inputs.
            buffer_size: Maximum transitions stored in history buffer.
            device: Compute device ('cpu' recommended for low latency).
        """
        self.act_dim = act_dim
        self.sens_dim = sens_dim if sens_dim is not None else act_dim
        self.kappa = float(kappa)
        self.tau = int(tau)
        self.time_dist = int(time_dist)
        self.regularization = float(regularization)
        self.bias_rate = float(bias_rate)
        self.s4avg = int(s4avg)
        self.buffer_size = int(buffer_size)
        self.device = torch.device(device)

        # Initialize internal states
        self.reset()

    def reset(self) -> None:
        """Reset the internal controller matrices, buffers, and step counter."""
        # Internal inverse model matrix M (negative feedback by default)
        # Shape: (act_dim, sens_dim)
        self.M = -torch.eye(self.act_dim, self.sens_dim, device=self.device)

        # Plastic correlation matrix C and normalized C_norm
        self.C = torch.zeros(self.act_dim, self.sens_dim, device=self.device)
        self.C_norm = torch.zeros(self.act_dim, self.sens_dim, device=self.device)

        # Adaptive bias vector to prevent joint freeze
        self.bias = torch.zeros(self.act_dim, device=self.device)

        # Smoothed sensory state
        self.obs_smoothed = torch.zeros(self.sens_dim, device=self.device)

        # Transition buffer storing pairs: (smoothed_sens, action)
        self.buffer: deque[tuple[torch.Tensor, torch.Tensor | None]] = deque(maxlen=self.buffer_size)

        # Step counter
        self.t = 0

    def step(self, obs: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Execute one step of DEP sensorimotor exploration.
        
        Args:
            obs: Proprioceptive sensory state (e.g. joint angles/velocities).
                 Can be 1D (sens_dim,) or 2D (1, sens_dim).
                 Accepts either numpy.ndarray or torch.Tensor.
                 
        Returns:
            action: Coordinated motor joint commands in [-1, +1], returned in the
                    same type (numpy or torch) as input obs.
        """
        is_numpy = isinstance(obs, np.ndarray)
        if is_numpy:
            obs_t = torch.from_numpy(obs).float().to(self.device)
        else:
            obs_t = obs.float().to(self.device)

        # Handle batch dimension if 2D (1, sens_dim)
        squeeze_output = False
        if obs_t.ndim == 2:
            if obs_t.shape[0] == 1:
                obs_t = obs_t.squeeze(0)
                squeeze_output = True
            else:
                # Multi-batch: process row-by-row
                actions = [self.step(row) for row in obs_t]
                if is_numpy:
                    return np.stack(actions, axis=0)
                return torch.stack(actions, dim=0)

        # Truncate or slice to sens_dim if observation contains extra global features
        if obs_t.shape[0] > self.sens_dim:
            obs_sens = obs_t[: self.sens_dim]
        elif obs_t.shape[0] < self.sens_dim:
            raise ValueError(
                f"Observation dimension {obs_t.shape[0]} is smaller than sens_dim {self.sens_dim}"
            )
        else:
            obs_sens = obs_t

        # 1. Rolling average smoothing
        if self.s4avg > 1 and self.t > 0:
            self.obs_smoothed = self.obs_smoothed + (obs_sens - self.obs_smoothed) / self.s4avg
        else:
            self.obs_smoothed = obs_sens.clone()

        self.buffer.append((self.obs_smoothed.clone(), None))

        # 2. Plasticity learning step (update C from history buffer)
        min_required_steps = 2 + self.time_dist
        if len(self.buffer) > min_required_steps:
            self._update_plasticity()

        # 3. Compute motor command
        action = self._compute_action()

        # Update last buffer entry with executed action
        self.buffer[-1] = (self.buffer[-1][0], action.clone())
        self.t += 1

        # Format output
        if squeeze_output:
            action = action.unsqueeze(0)

        if is_numpy:
            return action.cpu().numpy()
        return action

    def _update_plasticity(self) -> None:
        """Update correlation matrix C and normalize it."""
        # Recompute C from transitions in buffer
        C_new = torch.zeros_like(self.C)
        max_horizon = min(len(self.buffer) - 1, self.tau)

        for s in range(2, max_horizon):
            if s + self.time_dist + 1 > len(self.buffer):
                break

            x = self.buffer[-s][0]
            xx = self.buffer[-s - 1][0]
            xx_t = self.buffer[-s - self.time_dist][0]
            xxx_t = self.buffer[-s - 1 - self.time_dist][0]

            chi = x - xx            # Current difference
            v = xx_t - xxx_t        # Delayed difference
            mu = self.M @ chi       # Model projected response

            # Outer product accumulation: mu ⊗ v
            C_new += torch.outer(mu, v)

        self.C = C_new

        # Normalize controller matrix C
        # R = C @ M: linear response in motor space
        R = self.C @ self.M
        reg = 10.0 ** (-self.regularization)
        r_norm = torch.linalg.norm(R)
        if r_norm > 0:
            self.C_norm = self.C * self.kappa / (r_norm + reg)
        else:
            self.C_norm = self.C.clone()

        # Bias update (anti-freeze mechanism)
        if self.bias_rate > 0 and len(self.buffer) >= 2:
            prev_action = self.buffer[-2][1]
            if prev_action is not None:
                self.bias = self.bias - (
                    torch.clamp(prev_action * self.bias_rate, -0.05, 0.05) + self.bias * 0.001
                )

    def _compute_action(self) -> torch.Tensor:
        """Compute motor joint command from normalized correlation matrix."""
        # If early in exploration, initialize with small exploratory kick
        if self.t < (2 + self.time_dist) or torch.all(self.C_norm == 0):
            return torch.sin(torch.linspace(0, 2 * np.pi, self.act_dim, device=self.device) + self.t * 0.2) * 0.2

        # Projected motor stimulus: q = C_norm @ obs_smoothed
        q = self.C_norm @ self.obs_smoothed

        # Normalization
        reg = 10.0 ** (-self.regularization)
        q_norm = torch.linalg.norm(q) + reg
        q_normalized = q / q_norm

        # Saturated action command
        action = torch.tanh(q_normalized * self.kappa + self.bias)
        return torch.clamp(action, -1.0, 1.0)

    def get_correlation_matrix(self) -> torch.Tensor:
        """Return the current normalized correlation matrix C_norm."""
        return self.C_norm.detach().clone()

    def get_synergy_patterns(self, num_synergies: int = 4) -> np.ndarray:
        """Extract dominant motor synergies (joint groupings) via SVD.
        
        In biomechanics, motor synergies represent low-dimensional co-activation
        patterns where groups of leg joints flex and extend together in phase.
        
        Args:
            num_synergies: Number of principal synergy vectors to extract.
            
        Returns:
            synergies: Matrix of shape (num_synergies, act_dim) showing how
                       joints are coupled during walking.
        """
        C_mat = self.C_norm.cpu().numpy()
        U, S, Vt = np.linalg.svd(C_mat, full_matrices=False)
        k = min(num_synergies, U.shape[1])
        return U[:, :k].T

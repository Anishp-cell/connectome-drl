"""Reward formulation and stability checks for FlyGym locomotion.

Encourages forward stepping while penalizing lateral drift, excessive joint energy,
and postural instability (flipping onto back).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class LocomotionRewardWeights:
    """Hyperparameters weighting individual reward and penalty terms."""

    forward: float = 2.0           # Reward forward displacement (mm/step)
    lateral_penalty: float = 0.5   # Penalty for sideways drift (|vy|)
    upright: float = 1.0           # Bonus for keeping dorsal side pointing upward
    energy_penalty: float = 0.05   # Penalty for large motor activations (mean(a^2))
    alive_bonus: float = 0.5       # Constant bonus for remaining upright
    fall_penalty: float = -10.0    # Terminal penalty if fly flips onto back


class LocomotionRewardCalculator:
    """Calculates step rewards and termination conditions for fruit fly walking."""

    def __init__(self, weights: LocomotionRewardWeights | None = None) -> None:
        self.weights = weights if weights is not None else LocomotionRewardWeights()

    def compute_reward(
        self,
        prev_pos: np.ndarray,
        curr_pos: np.ndarray,
        curr_cardinal_vectors: np.ndarray,
        action: np.ndarray,
        is_terminated: bool = False,
    ) -> tuple[float, dict[str, float]]:
        """Calculate total step reward and breakdown dictionary.
        
        Args:
            prev_pos: Thorax world position at previous decision step (3,) [x, y, z] in mm.
            curr_pos: Thorax world position at current decision step (3,) [x, y, z] in mm.
            curr_cardinal_vectors: Fly body orientation cardinal vectors (3, 3).
                                   Row 0 = forward vector, Row 1 = lateral vector, Row 2 = vertical/dorsal vector.
            action: Continuous joint action commands in [-1, +1] of shape (42,).
            is_terminated: Whether the episode terminated early due to flipping/falling.
            
        Returns:
            total_reward: Scalar reward value for RL optimizer.
            reward_breakdown: Dictionary of individual reward components for logging.
        """
        # 1. Forward displacement along fly heading
        displacement = curr_pos - prev_pos  # (3,) in mm
        forward_axis = curr_cardinal_vectors[0]  # (3,) unit vector pointing forward
        forward_progress = float(np.dot(displacement, forward_axis))
        r_forward = forward_progress * self.weights.forward

        # 2. Lateral drift penalty (sliding sideways)
        lateral_axis = curr_cardinal_vectors[1]
        lateral_drift = float(abs(np.dot(displacement, lateral_axis)))
        r_lateral = -lateral_drift * self.weights.lateral_penalty

        # 3. Upright posture bonus: dorsal vector z-component should align with world +Z (0, 0, 1)
        # curr_cardinal_vectors[2] is the fly's upward dorsal vector
        upright_alignment = float(curr_cardinal_vectors[2, 2])
        r_upright = max(0.0, upright_alignment) * self.weights.upright

        # 4. Energy penalty (sum of squared joint commands)
        energy_expenditure = float(np.mean(np.square(action)))
        r_energy = -energy_expenditure * self.weights.energy_penalty

        # 5. Alive bonus and terminal fall penalty
        if is_terminated:
            r_alive = self.weights.fall_penalty
        else:
            r_alive = self.weights.alive_bonus

        total_reward = r_forward + r_lateral + r_upright + r_energy + r_alive

        breakdown = {
            "reward_forward": r_forward,
            "reward_lateral": r_lateral,
            "reward_upright": r_upright,
            "reward_energy": r_energy,
            "reward_alive": r_alive,
            "total_reward": total_reward,
            "forward_progress_mm": forward_progress,
            "upright_alignment": upright_alignment,
        }
        return total_reward, breakdown

    def check_termination(
        self,
        curr_pos: np.ndarray,
        curr_cardinal_vectors: np.ndarray,
        min_height_mm: float = 0.15,
        min_upright_dot: float = 0.2,
    ) -> bool:
        """Check if fly has flipped over onto its back or collapsed onto the floor.
        
        Args:
            curr_pos: Thorax position (3,) in mm.
            curr_cardinal_vectors: Fly orientation cardinal vectors (3, 3).
            min_height_mm: Minimum allowable thorax height before considered collapsed.
            min_upright_dot: Minimum dot product between fly dorsal axis and world +Z.
            
        Returns:
            is_fallen: True if fly has flipped or collapsed.
        """
        # Fly dorsal vector z-component: < 0.2 means fly is tilted > 78 degrees or completely inverted
        upright_dot = curr_cardinal_vectors[2, 2]
        if upright_dot < min_upright_dot:
            return True

        # Thorax height check
        height = curr_pos[2]
        if height < min_height_mm:
            return True

        return False

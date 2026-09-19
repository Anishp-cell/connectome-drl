"""Unit tests for Biological CPG Locomotion Environment (Phase 7: Step 7.1).

Verifies:
  - CPGLocomotionEnv action/observation spaces and reset dynamics.
  - Closed-loop physical forward walking (continuous tripod stepping).
  - Descending steering asymmetry (left vs right turning).
  - 4D biological connectome descending command mapping.
"""

from __future__ import annotations

import numpy as np
import pytest

from connectome_rl.src.envs.cpg_wrapper import CPGLocomotionEnv


class TestCPGLocomotionEnv:
    """Test suite for Central Pattern Generator locomotion engine."""

    def test_cpg_env_spaces_and_reset(self) -> None:
        env = CPGLocomotionEnv()
        assert env.action_space.shape == (2,)
        assert env.observation_space.shape == (12,)

        obs, info = env.reset(seed=42)
        assert obs.shape == (12,)
        assert not np.isnan(obs).any()
        assert "position_mm" in info
        assert "forward_dist_mm" in info
        assert info["forward_dist_mm"] == 0.0
        env.close()

    def test_cpg_forward_locomotion(self) -> None:
        """Verify that 2D symmetric drive produces positive forward walking in MuJoCo."""
        env = CPGLocomotionEnv()
        obs, info = env.reset(seed=0)

        # Step 150 simulation steps with straight forward drive
        for _ in range(150):
            obs, reward, terminated, truncated, info = env.step(np.array([1.0, 1.0]))
            assert not np.isnan(obs).any()

        # The fly must have physically walked forward along X
        forward_dist = info["forward_dist_mm"]
        assert forward_dist > 0.05, f"Fly failed to walk forward (dist={forward_dist:.4f} mm)"
        env.close()

    def test_cpg_turning_steering_asymmetry(self) -> None:
        """Verify that asymmetric left vs right drive produces physical heading deflection."""
        env = CPGLocomotionEnv()

        # 1. Left turn test: right legs drive harder (drive=[0.3, 1.3])
        env.reset(seed=0)
        for _ in range(100):
            obs, _, _, _, info_left = env.step(np.array([0.3, 1.3]))
        yaw_left = info_left["yaw_deg"]

        # 2. Right turn test: left legs drive harder (drive=[1.3, 0.3])
        env.reset(seed=0)
        for _ in range(100):
            obs, _, _, _, info_right = env.step(np.array([1.3, 0.3]))
        yaw_right = info_right["yaw_deg"]

        # Opposite steering commands must produce distinct yaw heading responses
        assert yaw_left != yaw_right
        env.close()

    def test_cpg_4d_biological_descending_mapping(self) -> None:
        """Verify that 4D biological connectome signals [DNa01_L, DNa01_R, DNa02_L, DNa02_R] work."""
        env = CPGLocomotionEnv()
        env.reset(seed=0)

        # Forward drive: DNa02 active (velocity), DNa01 neutral (no steering)
        action_4d = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
        for _ in range(100):
            obs, reward, term, trunc, info = env.step(action_4d)

        assert info["forward_dist_mm"] > 0.02
        env.close()

"""Unit tests for FlyLocomotionEnv and LocomotionRewardCalculator.

Tests:
1. Gymnasium space specifications (42 actions, 100 observations).
2. Environment reset and step dynamics in MuJoCo.
3. Reward calculation components (forward progress, lateral drift, energy penalty).
4. Postural stability and flip termination detection.
5. Offscreen camera rendering (RGB frame shape).
6. End-to-end closed-loop stepping with a ConnectomePolicy.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import torch

from connectome_rl.src.envs.fly_wrapper import FlyLocomotionEnv
from connectome_rl.src.envs.rewards import LocomotionRewardCalculator, LocomotionRewardWeights
from connectome_rl.src.models.connectome_policy import ConnectomePolicy


class TestFlyLocomotionEnv:
    """Test suite for the FlyGym Gymnasium environment wrapper."""

    @pytest.fixture
    def env_setup(self):
        env = FlyLocomotionEnv(
            physics_steps_per_action=5,  # Short physics step for fast unit testing
            max_episode_steps=50,
            enable_render=False,
        )
        yield env
        env.close()

    def test_env_spaces_and_reset(self, env_setup):
        """Plain English: Does the environment have 42 action dimensions and 100 observation dimensions, and does reset produce clean states?"""
        env = env_setup

        assert env.action_space.shape == (42,)
        assert env.observation_space.shape == (100,)
        assert env.action_space.low.min() == -1.0
        assert env.action_space.high.max() == 1.0

        obs, info = env.reset()
        assert obs.shape == (100,)
        assert obs.dtype == np.float32
        assert not np.isnan(obs).any(), "Reset observation must not contain NaNs"
        assert not np.isinf(obs).any(), "Reset observation must not contain Infs"
        assert info["step"] == 0

    def test_env_step_execution(self, env_setup):
        """Plain English: When we send a 42-dim joint command to MuJoCo, does the physics engine advance and return valid rewards and states?"""
        env = env_setup
        obs, _ = env.reset()

        # Step with neutral zero actions (maintain standing posture)
        action = np.zeros(42, dtype=np.float32)
        next_obs, reward, terminated, truncated, info = env.step(action)

        assert next_obs.shape == (100,)
        assert not np.isnan(next_obs).any()
        assert isinstance(reward, float)
        assert not np.isnan(reward)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert info["step"] == 1
        assert "reward_forward" in info
        assert "reward_energy" in info

    def test_env_reward_calculator_components(self):
        """Plain English: Does forward progress award positive points, while lateral slipping and large joint commands get penalized?"""
        calc = LocomotionRewardCalculator()

        # Cardinal vectors: forward=[1,0,0], lateral=[0,1,0], dorsal=[0,0,1]
        cardinals = np.eye(3)

        # Scenario A: Clean forward progress (moving +1 mm along X)
        prev_pos = np.array([0.0, 0.0, 0.5])
        curr_pos = np.array([1.0, 0.0, 0.5])
        zero_action = np.zeros(42)

        r_fwd, info_fwd = calc.compute_reward(prev_pos, curr_pos, cardinals, zero_action)
        assert info_fwd["reward_forward"] > 0.0, "Forward progress must yield positive reward"
        assert info_fwd["reward_lateral"] == 0.0, "Zero sideways drift should not be penalized"

        # Scenario B: Sideways drift (moving +1 mm along Y)
        curr_pos_lat = np.array([0.0, 1.0, 0.5])
        r_lat, info_lat = calc.compute_reward(prev_pos, curr_pos_lat, cardinals, zero_action)
        assert info_lat["reward_lateral"] < 0.0, "Sideways drift must be penalized"

        # Scenario C: High joint energy expenditure (action = 1.0 everywhere)
        high_action = np.ones(42)
        r_energy, info_energy = calc.compute_reward(prev_pos, curr_pos, cardinals, high_action)
        assert info_energy["reward_energy"] < info_fwd["reward_energy"], "High motor effort must be penalized"

    def test_env_termination_on_flip(self):
        """Plain English: If the fly tilts or rolls upside down, does the safety check catch it and flag terminated?"""
        calc = LocomotionRewardCalculator()

        pos = np.array([0.0, 0.0, 0.5])
        # Inverted cardinal vectors: dorsal axis points down towards -Z (0, 0, -1)
        flipped_cardinals = np.array([
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],  # Inverted dorsal vector
        ])

        is_flipped = calc.check_termination(pos, flipped_cardinals)
        assert is_flipped is True, "Flipped fly must be marked as terminated"

        # Normal upright orientation
        upright_cardinals = np.eye(3)
        assert calc.check_termination(pos, upright_cardinals) is False

    def test_env_camera_rendering(self):
        """Plain English: When camera rendering is enabled, does env.render() return a valid 480x640x3 color image of the fly?"""
        env = FlyLocomotionEnv(
            physics_steps_per_action=2,
            enable_render=True,
            render_camera_name="camera_top",
        )
        try:
            env.reset()
            frame = env.render()
            assert frame is not None, "Camera render must return an image array"
            assert isinstance(frame, np.ndarray)
            assert frame.shape == (480, 640, 3), f"Expected shape (480, 640, 3), got {frame.shape}"
            assert frame.dtype == np.uint8
        finally:
            env.close()

    def test_env_closed_loop_with_connectome_policy(self):
        """Plain English: Can our biological ConnectomePolicy successfully steer the FlyGym MuJoCo simulation in a closed loop?"""
        data_path = Path(__file__).resolve().parents[1] / "data" / "dna_circuit_tensors.pt"
        if not data_path.exists():
            pytest.skip("dna_circuit_tensors.pt not found on disk")

        env = FlyLocomotionEnv(physics_steps_per_action=5, max_episode_steps=10)
        policy = ConnectomePolicy(obs_dim=100, act_dim=42, circuit_data=data_path)

        try:
            obs, _ = env.reset()
            for step in range(5):
                obs_tensor = torch.from_numpy(obs).unsqueeze(0).float()
                with torch.no_grad():
                    action, _, _, _ = policy.get_action_and_value(obs_tensor)
                act_np = action.squeeze(0).numpy()

                next_obs, reward, terminated, truncated, info = env.step(act_np)
                assert next_obs.shape == (100,)
                assert not np.isnan(next_obs).any()
                obs = next_obs
                if terminated or truncated:
                    break
        finally:
            env.close()

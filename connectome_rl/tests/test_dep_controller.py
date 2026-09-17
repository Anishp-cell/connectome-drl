"""Unit tests for the Differential Extrinsic Plasticity (DEP) Exploration Controller.

Tests:
1. Initialization, buffer structures, and reset behavior.
2. Action generation and type handling (NumPy arrays vs PyTorch tensors).
3. Self-organizing plasticity updates to the correlation matrix C.
4. Long-run numerical stability (zero NaNs, bounded in [-1, 1]).
5. Biomechanical motor synergy pattern extraction via SVD.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from connectome_rl.src.models.dep_controller import DEPController


class TestDEPController:
    """Test suite for DEPController sensorimotor exploration."""

    @pytest.fixture
    def dep_setup(self):
        act_dim = 42
        sens_dim = 42
        controller = DEPController(
            act_dim=act_dim,
            sens_dim=sens_dim,
            kappa=1.0,
            tau=30,
            time_dist=1,
            regularization=0.01,
            bias_rate=0.01,
            s4avg=2,
        )
        return controller, act_dim, sens_dim

    def test_dep_initialization_and_reset(self, dep_setup):
        """Plain English: Does the controller initialize its internal 42x42 correlation matrix to zeros and cleanly reset its buffers?"""
        controller, act_dim, sens_dim = dep_setup

        assert controller.act_dim == act_dim
        assert controller.sens_dim == sens_dim
        assert controller.t == 0
        assert len(controller.buffer) == 0

        # Correlation matrix C and C_norm must start at shape (42, 42)
        assert controller.C.shape == (act_dim, sens_dim)
        assert controller.C_norm.shape == (act_dim, sens_dim)
        assert torch.all(controller.C == 0.0)

        # Feed some dummy data
        dummy_obs = np.random.randn(sens_dim).astype(np.float32)
        controller.step(dummy_obs)
        assert controller.t == 1
        assert len(controller.buffer) == 1

        # Reset must restore pristine initial state
        controller.reset()
        assert controller.t == 0
        assert len(controller.buffer) == 0
        assert torch.all(controller.C == 0.0)

    def test_dep_action_generation_and_types(self, dep_setup):
        """Plain English: Can the controller accept both NumPy arrays and PyTorch tensors, returning properly bounded 42-dim joint commands?"""
        controller, act_dim, sens_dim = dep_setup

        # 1. NumPy input -> NumPy output
        np_obs = np.sin(np.linspace(0, np.pi, sens_dim, dtype=np.float32))
        np_act = controller.step(np_obs)
        assert isinstance(np_act, np.ndarray), "NumPy input must return NumPy action"
        assert np_act.shape == (act_dim,), f"Action shape must be ({act_dim},)"
        assert (-1.0 <= np_act).all() and (np_act <= 1.0).all(), "Actions must be in [-1, +1]"

        # 2. PyTorch tensor input -> PyTorch tensor output
        torch_obs = torch.cos(torch.linspace(0, np.pi, sens_dim))
        torch_act = controller.step(torch_obs)
        assert isinstance(torch_act, torch.Tensor), "Torch input must return Torch action"
        assert torch_act.shape == (act_dim,)
        assert (-1.0 <= torch_act).all() and (torch_act <= 1.0).all()

        # 3. 2D batched input (1, sens_dim)
        batched_obs = np.random.randn(1, sens_dim).astype(np.float32)
        batched_act = controller.step(batched_obs)
        assert batched_act.shape == (1, act_dim)

    def test_dep_self_organizing_plasticity(self, dep_setup):
        """Plain English: As rhythmic sensory oscillations pass through the controller, does the correlation matrix C self-organize from all zeros into a non-zero structure?"""
        controller, act_dim, sens_dim = dep_setup
        controller.reset()

        # Initial correlation matrix is all zeros
        assert torch.all(controller.get_correlation_matrix() == 0.0)

        # Feed 15 steps of oscillating joint movements (like swinging legs)
        for step in range(15):
            t_val = step * 0.2
            osc_obs = np.sin(np.linspace(0, 2 * np.pi, sens_dim) + t_val).astype(np.float32)
            controller.step(osc_obs)

        # Correlation matrix must have learned non-zero relationships
        C_norm = controller.get_correlation_matrix()
        assert C_norm.shape == (act_dim, sens_dim)
        assert not torch.all(C_norm == 0.0), "Plasticity updates must produce non-zero correlation matrix C"
        assert not torch.isnan(C_norm).any(), "Correlation matrix must not contain NaNs"

    def test_dep_numerical_stability_long_run(self, dep_setup):
        """Plain English: Over 100 continuous stepping cycles with varied sensory inputs, does the controller run stably without any NaNs or numerical explosions?"""
        controller, act_dim, sens_dim = dep_setup
        controller.reset()

        for step in range(100):
            # Simulated proprioceptive joint angles with noise
            obs = np.sin(step * 0.15 + np.arange(sens_dim) * 0.1).astype(np.float32)
            obs += np.random.randn(sens_dim).astype(np.float32) * 0.05
            action = controller.step(obs)

            assert not np.isnan(action).any(), f"NaN detected at step {step}"
            assert not np.isinf(action).any(), f"Inf detected at step {step}"
            assert np.all(action >= -1.0) and np.all(action <= 1.0), f"Action out of bounds at step {step}"

    def test_dep_synergy_pattern_extraction(self, dep_setup):
        """Plain English: Can we mathematically extract coordinated multi-joint motor synergies from the learned correlation matrix via SVD?"""
        controller, act_dim, sens_dim = dep_setup
        controller.reset()

        # Run for 25 steps to populate correlation matrix
        for step in range(25):
            obs = np.cos(step * 0.2 + np.arange(sens_dim) * 0.3).astype(np.float32)
            controller.step(obs)

        num_synergies = 4
        synergies = controller.get_synergy_patterns(num_synergies=num_synergies)
        assert synergies.shape == (num_synergies, act_dim), f"Expected shape ({num_synergies}, {act_dim})"
        assert not np.isnan(synergies).any(), "Synergy matrix must not contain NaNs"

        # Check orthonormal properties from SVD (dot product of distinct synergy vectors should be ~0)
        dot_01 = np.dot(synergies[0], synergies[1])
        assert abs(dot_01) < 1e-5, "Synergy vectors extracted from SVD must be orthogonal"

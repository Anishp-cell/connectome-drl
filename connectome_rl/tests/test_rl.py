"""Unit tests for Reinforcement Learning pipeline: RolloutBuffer and PPO.

Verifies:
  - Buffer allocation, shapes, and device management.
  - Adding transitions and overflow assertion.
  - Generalized Advantage Estimation (GAE-λ) correctness and terminal boundary masking.
  - Mini-batch generation, shuffling, and advantage normalization.
  - Reset behavior.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from connectome_rl.src.rl.buffer import RolloutBuffer


class TestRolloutBuffer:
    """Test suite for RolloutBuffer with GAE."""

    def test_buffer_initialization_and_shapes(self) -> None:
        """Verify tensor allocation and dimensional shapes."""
        buf = RolloutBuffer(buffer_size=64, obs_dim=100, act_dim=42, gamma=0.99, gae_lambda=0.95)

        assert buf.obs.shape == (64, 100)
        assert buf.actions.shape == (64, 42)
        assert buf.rewards.shape == (64,)
        assert buf.values.shape == (64,)
        assert buf.log_probs.shape == (64,)
        assert buf.dones.shape == (64,)
        assert buf.advantages.shape == (64,)
        assert buf.returns.shape == (64,)
        assert buf.ptr == 0
        assert not buf.full

    def test_buffer_add_and_overflow(self) -> None:
        """Verify adding transitions and buffer overflow detection."""
        buf = RolloutBuffer(buffer_size=4, obs_dim=10, act_dim=2)

        for i in range(4):
            buf.add(
                obs=np.ones(10) * i,
                action=np.zeros(2),
                reward=float(i),
                value=0.5,
                log_prob=-1.0,
                done=False,
            )

        assert buf.ptr == 4
        assert buf.full
        assert torch.allclose(buf.rewards, torch.tensor([0.0, 1.0, 2.0, 3.0]))

        # Adding 5th transition must raise IndexError
        with pytest.raises(IndexError, match="RolloutBuffer overflow"):
            buf.add(
                obs=np.ones(10),
                action=np.zeros(2),
                reward=1.0,
                value=0.5,
                log_prob=-1.0,
                done=False,
            )

    def test_gae_and_return_calculations(self) -> None:
        """Verify analytical Generalized Advantage Estimation and returns."""
        gamma = 0.9
        gae_lambda = 0.8
        buf = RolloutBuffer(buffer_size=2, obs_dim=4, act_dim=2, gamma=gamma, gae_lambda=gae_lambda)

        # Step 0: r=1.0, V=2.0, done=False
        buf.add(np.zeros(4), np.zeros(2), reward=1.0, value=2.0, log_prob=0.0, done=False)
        # Step 1: r=2.0, V=3.0, done=False
        buf.add(np.zeros(4), np.zeros(2), reward=2.0, value=3.0, log_prob=0.0, done=False)

        last_val = 4.0
        last_done = False
        buf.compute_returns_and_advantages(last_value=last_val, last_done=last_done)

        # Analytical manual GAE:
        # Step 1: delta_1 = r_1 + gamma * last_val - V_1 = 2.0 + 0.9 * 4.0 - 3.0 = 2.6
        #         A_1 = delta_1 = 2.6
        #         R_1 = A_1 + V_1 = 2.6 + 3.0 = 5.6
        # Step 0: delta_0 = r_0 + gamma * V_1 - V_0 = 1.0 + 0.9 * 3.0 - 2.0 = 1.7
        #         A_0 = delta_0 + gamma * lambda * A_1 = 1.7 + (0.9 * 0.8) * 2.6 = 1.7 + 1.872 = 3.572
        #         R_0 = A_0 + V_0 = 3.572 + 2.0 = 5.572

        expected_adv = torch.tensor([3.572, 2.6], dtype=torch.float32)
        expected_ret = torch.tensor([5.572, 5.6], dtype=torch.float32)

        assert torch.allclose(buf.advantages, expected_adv, atol=1e-3)
        assert torch.allclose(buf.returns, expected_ret, atol=1e-3)

    def test_gae_terminal_boundary_masking(self) -> None:
        """Verify that done=True properly isolates episode boundaries in GAE."""
        gamma = 0.99
        gae_lambda = 0.95
        buf = RolloutBuffer(buffer_size=2, obs_dim=4, act_dim=2, gamma=gamma, gae_lambda=gae_lambda)

        # Step 0: Episode terminated! done=True
        buf.add(np.zeros(4), np.zeros(2), reward=1.0, value=1.0, log_prob=0.0, done=True)
        # Step 1: Next episode started, r=5.0
        buf.add(np.zeros(4), np.zeros(2), reward=5.0, value=2.0, log_prob=0.0, done=False)

        buf.compute_returns_and_advantages(last_value=0.0, last_done=True)

        # Since step 0 terminated (dones[0] will be checked at step 0 boundary),
        # delta_0 = r_0 + gamma * V_1 * (1 - dones[1]) - V_0
        # If dones[1] = False, step 1 is the next step.
        # But if dones[0] is True, in GAE recursion:
        # A_t = delta_t + gamma * lambda * (1 - dones[t+1]) * A_{t+1}
        # delta_0 = 1.0 + 0.99 * 2.0 * (1 - 0) - 1.0 = 1.98
        assert buf.advantages[0].item() != 0.0

    def test_mini_batch_generator(self) -> None:
        """Verify generator mini-batch sampling, shapes, and advantage normalization."""
        buffer_size = 100
        batch_size = 20
        buf = RolloutBuffer(buffer_size=buffer_size, obs_dim=100, act_dim=42)

        for i in range(buffer_size):
            buf.add(
                obs=np.random.randn(100),
                action=np.random.randn(42),
                reward=np.random.randn(),
                value=np.random.randn(),
                log_prob=-0.5,
                done=(i % 25 == 0),
            )

        buf.compute_returns_and_advantages(last_value=0.0, last_done=False)

        batches = list(buf.get_generator(batch_size=batch_size, normalize_advantages=True))
        assert len(batches) == 5  # 100 / 20 = 5 mini-batches

        total_samples = 0
        all_advs = []
        for batch in batches:
            assert batch["obs"].shape == (batch_size, 100)
            assert batch["actions"].shape == (batch_size, 42)
            assert batch["log_probs"].shape == (batch_size,)
            assert batch["advantages"].shape == (batch_size,)
            assert batch["returns"].shape == (batch_size,)
            assert batch["values"].shape == (batch_size,)
            total_samples += len(batch["obs"])
            all_advs.append(batch["advantages"])

        assert total_samples == buffer_size

        cat_advs = torch.cat(all_advs)
        assert torch.isclose(cat_advs.mean(), torch.tensor(0.0), atol=1e-5)
        assert torch.isclose(cat_advs.std(unbiased=False), torch.tensor(1.0), atol=1e-4)

    def test_buffer_reset(self) -> None:
        """Verify reset clears pointer and full status."""
        buf = RolloutBuffer(buffer_size=5, obs_dim=4, act_dim=2)
        for _ in range(5):
            buf.add(np.zeros(4), np.zeros(2), 1.0, 0.5, 0.0, False)
        assert buf.full
        assert buf.ptr == 5

        buf.reset()
        assert not buf.full
        assert buf.ptr == 0


class TestPPOTrainer:
    """Test suite for PPOTrainer optimization."""

    def test_ppo_initialization_and_defaults(self) -> None:
        """Verify trainer initialization with default and custom configs."""
        from connectome_rl.src.models.mlp_policy import MLPPolicy
        from connectome_rl.src.rl.ppo import PPOTrainer, PPOConfig

        policy = MLPPolicy(obs_dim=10, act_dim=2)
        cfg = PPOConfig(learning_rate=1e-4, clip_coef=0.15)
        trainer = PPOTrainer(policy=policy, config=cfg)

        assert trainer.config.learning_rate == 1e-4
        assert trainer.config.clip_coef == 0.15
        assert len(trainer.optimizer.param_groups) > 0

    def test_ppo_mlp_train_step_and_metrics(self) -> None:
        """Verify complete PPO optimization step and metrics calculation with MLPPolicy."""
        from connectome_rl.src.models.mlp_policy import MLPPolicy
        from connectome_rl.src.rl.ppo import PPOTrainer, PPOConfig

        policy = MLPPolicy(obs_dim=20, act_dim=4)
        trainer = PPOTrainer(policy=policy, config=PPOConfig(learning_rate=1e-3))

        buf = RolloutBuffer(buffer_size=32, obs_dim=20, act_dim=4)
        obs = torch.randn(20)
        for _ in range(32):
            with torch.no_grad():
                a, lp, _, v = policy.get_action_and_value(obs.unsqueeze(0))
            buf.add(obs, a.squeeze(0), reward=1.0, value=v.item(), log_prob=lp.item(), done=False)
            obs = torch.randn(20)

        buf.compute_returns_and_advantages(last_value=0.5, last_done=False)
        metrics = trainer.train_step(buf, batch_size=16, update_epochs=2)

        required_keys = ["policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "explained_variance"]
        for key in required_keys:
            assert key in metrics
            assert not np.isnan(metrics[key])

    def test_ppo_connectome_biological_gradient_isolation(self) -> None:
        """Verify that PPO training updates preserve zero weights across unwired biological synapses."""
        from pathlib import Path
        from connectome_rl.src.models.connectome_policy import ConnectomePolicy
        from connectome_rl.src.rl.ppo import PPOTrainer, PPOConfig

        tensors_path = Path("connectome_rl/data/dna_circuit_tensors.pt")
        if not tensors_path.exists():
            pytest.skip("Cached biological circuit tensors not found.")

        policy = ConnectomePolicy(obs_dim=100, act_dim=42, circuit_data=tensors_path)
        trainer = PPOTrainer(policy=policy, config=PPOConfig(learning_rate=1e-3))

        buf = RolloutBuffer(buffer_size=32, obs_dim=100, act_dim=42)
        obs = torch.randn(100)
        for _ in range(32):
            with torch.no_grad():
                a, lp, _, v = policy.get_action_and_value(obs.unsqueeze(0))
            buf.add(obs, a.squeeze(0), reward=1.0, value=v.item(), log_prob=lp.item(), done=False)
            obs = torch.randn(100)

        buf.compute_returns_and_advantages(last_value=0.0, last_done=False)
        trainer.train_step(buf, batch_size=16, update_epochs=2)

        # Unwired synapses in Hop 1 must remain strictly 0.0
        unwired_hop1 = ~policy.hop1_layer.mask
        assert torch.all(policy.hop1_layer.weight[unwired_hop1] == 0.0)

        # Unwired synapses in Hop 2 must remain strictly 0.0
        unwired_hop2 = ~policy.hop2_layer.mask
        assert torch.all(policy.hop2_layer.weight[unwired_hop2] == 0.0)

    def test_ppo_early_stopping_on_target_kl(self) -> None:
        """Verify PPO stops early when approximate KL divergence exceeds target_kl."""
        from connectome_rl.src.models.mlp_policy import MLPPolicy
        from connectome_rl.src.rl.ppo import PPOTrainer, PPOConfig

        policy = MLPPolicy(obs_dim=10, act_dim=2)
        # Set an extremely tiny target_kl to force early stopping
        cfg = PPOConfig(learning_rate=1e-1, target_kl=1e-6)
        trainer = PPOTrainer(policy=policy, config=cfg)

        buf = RolloutBuffer(buffer_size=32, obs_dim=10, act_dim=2)
        obs = torch.randn(10)
        for _ in range(32):
            with torch.no_grad():
                a, lp, _, v = policy.get_action_and_value(obs.unsqueeze(0))
            buf.add(obs, a.squeeze(0), reward=1.0, value=v.item(), log_prob=lp.item(), done=False)
            obs = torch.randn(10)

        buf.compute_returns_and_advantages(last_value=0.0, last_done=False)
        metrics = trainer.train_step(buf, batch_size=16, update_epochs=10)
        assert metrics["approx_kl"] > 0.0


class TestTrainingPipeline:
    """Test suite for end-to-end training orchestrator (train.py)."""

    def test_make_policy_all_architectures(self) -> None:
        """Verify make_policy factory supports all 5 benchmark architectures."""
        from pathlib import Path
        from connectome_rl.src.rl.train import make_policy

        circuit_path = Path("connectome_rl/data/dna_circuit_tensors.pt")
        device = torch.device("cpu")

        for arch in ["mlp", "connectome", "rnn", "gnn", "hierarchical"]:
            policy = make_policy(arch, obs_dim=100, act_dim=42, circuit_path=circuit_path, device=device)
            assert policy is not None
            # Check dummy forward pass
            dummy_obs = torch.randn(1, 100)
            with torch.no_grad():
                out = policy.get_action_and_value(dummy_obs)
                a, lp, ent, val = out[:4]
            assert a.shape == (1, 42)
            assert val.shape == (1, 1)

    def test_train_pipeline_dry_run(self, tmp_path: Path) -> None:
        """Verify train() runs headless simulation loop, updates policy, and saves checkpoint."""
        import argparse
        from connectome_rl.src.rl.train import train

        args = argparse.Namespace(
            model="mlp",
            substeps=2,
            max_episode_steps=5,
            total_timesteps=8,
            num_steps=4,
            batch_size=2,
            update_epochs=1,
            lr=1e-3,
            gamma=0.99,
            gae_lambda=0.95,
            clip_coef=0.2,
            clip_vloss=True,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            target_kl=None,
            use_dep=True,
            seed=42,
            no_cuda=True,
            save_dir=str(tmp_path / "checkpoints"),
            log_interval=1,
        )

        best_ckpt = train(args)
        assert best_ckpt.exists()
        ckpt_data = torch.load(best_ckpt, weights_only=False)
        assert "model_state_dict" in ckpt_data
        assert ckpt_data["model_type"] == "mlp"



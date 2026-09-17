"""Master Training Orchestrator for Connectome-Constrained Drosophila Locomotion.

Coordinates:
  - Vectorized or single FlyGym + MuJoCo headless physics environments.
  - Policy initialization across all 5 benchmark models:
      1. MLP (unconstrained baseline)
      2. Connectome (hop-1 and hop-2 biological synaptic masks)
      3. Connectome-RNN (recurrent central pattern generator)
      4. Synaptic-GNN (synaptic edge message passing)
      5. Hierarchical (descending brain command -> VNC connectome)
  - Optional DEP-RL (Differential Extrinsic Plasticity) motor synergy exploration.
  - On-policy rollout collection with RolloutBuffer & GAE-λ.
  - PPOTrainer optimization with gradient clipping & biological mask preservation.
  - TensorBoard telemetry logging and model checkpoint serialization.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn


class MetricLogger:
    """Resilient telemetry logger supporting TensorBoard and structured JSONL logs."""

    def __init__(self, log_dir: Path | str) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.log_dir / "metrics.jsonl"
        self.writer = None

        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(str(self.log_dir))
        except Exception:
            self.writer = None

    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        if self.writer is not None:
            try:
                self.writer.add_scalar(tag, scalar_value, global_step)
            except Exception:
                pass

        try:
            record = json.dumps({"tag": tag, "value": float(scalar_value), "step": int(global_step)})
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(record + "\n")
        except Exception:
            pass

    def close(self) -> None:
        if self.writer is not None:
            try:
                self.writer.close()
            except Exception:
                pass


from connectome_rl.src.connectome.graph_utils import load_circuit_data
from connectome_rl.src.envs.fly_wrapper import FlyLocomotionEnv
from connectome_rl.src.models.connectome_policy import ConnectomePolicy
from connectome_rl.src.models.connectome_rnn import ConnectomeRNNPolicy
from connectome_rl.src.models.dep_controller import DEPController
from connectome_rl.src.models.hierarchical_policy import HierarchicalPolicy
from connectome_rl.src.models.mlp_policy import MLPPolicy
from connectome_rl.src.models.synaptic_gnn import SynapticGNNPolicy
from connectome_rl.src.rl.buffer import RolloutBuffer
from connectome_rl.src.rl.ppo import PPOConfig, PPOTrainer


def make_policy(
    model_type: str,
    obs_dim: int,
    act_dim: int,
    circuit_path: Path,
    device: torch.device,
) -> nn.Module:
    """Instantiate the requested policy architecture."""
    model_type = model_type.lower()

    if model_type == "mlp":
        policy = MLPPolicy(obs_dim=obs_dim, act_dim=act_dim)
    elif model_type == "connectome":
        policy = ConnectomePolicy(obs_dim=obs_dim, act_dim=act_dim, circuit_data=circuit_path)
    elif model_type == "rnn":
        policy = ConnectomeRNNPolicy(obs_dim=obs_dim, act_dim=act_dim, circuit_data=circuit_path)
    elif model_type == "gnn":
        policy = SynapticGNNPolicy(obs_dim=obs_dim, act_dim=act_dim, circuit_data=circuit_path)
    elif model_type == "hierarchical":
        policy = HierarchicalPolicy(obs_dim=obs_dim, act_dim=act_dim, circuit_data=circuit_path)
    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Choose from ['mlp', 'connectome', 'rnn', 'gnn', 'hierarchical']."
        )

    return policy.to(device)


def get_safe_device(no_cuda: bool = False) -> torch.device:
    """Safely determine compute device, checking for CUDA kernel compatibility."""
    if not no_cuda and torch.cuda.is_available():
        try:
            # Check if CUDA kernels are compiled for the host GPU architecture
            test_tensor = torch.zeros(1, device="cuda")
            _ = test_tensor + 1.0
            return torch.device("cuda")
        except Exception:
            pass
    return torch.device("cpu")


def train(args: argparse.Namespace) -> Path:
    """Execute end-to-end PPO training on the fruit fly locomotion task."""
    device = get_safe_device(args.no_cuda)

    # 1. Reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    print("=" * 80)
    print(f"  CONNECTOME-RL TRAINING PIPELINE: Drosophila Locomotion")
    print(f"  Model Architecture:  {args.model.upper()}")
    print(f"  DEP Exploration:     {'ENABLED' if args.use_dep else 'DISABLED'}")
    print(f"  Execution Device:    {device}")
    print(f"  Total Timesteps:     {args.total_timesteps:,}")
    print(f"  Rollout Buffer Size: {args.num_steps}")
    print(f"  Mini-Batch Size:     {args.batch_size}")
    print(f"  PPO Update Epochs:   {args.update_epochs}")
    print("=" * 80)


    # 2. Paths and Directories
    project_root = Path(__file__).resolve().parents[3]
    circuit_path = project_root / "connectome_rl" / "data" / "dna_circuit_tensors.pt"
    if args.model != "mlp" and not circuit_path.exists():
        raise FileNotFoundError(f"Biological circuit data not found at: {circuit_path}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"{args.model}_seed{args.seed}_{int(time.time())}"
    log_dir = Path("runs") / run_name
    writer = MetricLogger(log_dir)

    # 3. Biomechanical Environment (Headless MuJoCo for maximal SPS)
    print("\n[1/4] Initializing Headless FlyGym Biomechanical Environment...")
    env = FlyLocomotionEnv(
        physics_steps_per_action=args.substeps,
        init_pose="tripod",
        max_episode_steps=args.max_episode_steps,
        enable_render=False,
    )
    obs_dim = env.observation_space.shape[0]  # 100
    act_dim = env.action_space.shape[0]        # 42

    # 4. Policy and Trainer
    print(f"[2/4] Constructing {args.model.upper()} Policy Network...")
    policy = make_policy(args.model, obs_dim, act_dim, circuit_path, device)
    ppo_cfg = PPOConfig(
        learning_rate=args.lr,
        clip_coef=args.clip_coef,
        clip_vloss=args.clip_vloss,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
    )
    trainer = PPOTrainer(policy=policy, config=ppo_cfg, device=device)

    # 5. Optional DEP Controller
    dep_controller: DEPController | None = None
    if args.use_dep:
        print("[2b/4] Initializing Self-Organizing DEP Exploration Controller...")
        dep_controller = DEPController(act_dim=act_dim)

    # 6. Pre-allocated Rollout Buffer
    buffer = RolloutBuffer(
        buffer_size=args.num_steps,
        obs_dim=obs_dim,
        act_dim=act_dim,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        device=device,
    )

    # 7. Training Loop
    print("\n[3/4] Starting PPO Rollout Collection and Optimization Loop...")
    print("-" * 80)

    global_step = 0
    start_time = time.time()
    obs, info = env.reset(seed=args.seed)

    episode_return = 0.0
    episode_length = 0
    episode_count = 0
    best_return = -float("inf")
    best_model_path = save_dir / f"{args.model}_best.pt"

    num_updates = args.total_timesteps // args.num_steps

    for update in range(1, num_updates + 1):
        buffer.reset()

        # Phase A: Rollout Collection
        for step in range(args.num_steps):
            global_step += 1
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).float().to(device)

            with torch.no_grad():
                out = policy.get_action_and_value(obs_tensor)
                action, log_prob, _, value = out[:4]

            act_np = action.squeeze(0).cpu().numpy()

            # Optional DEP self-organizing motor synergy blending
            if dep_controller is not None:
                # Anneal DEP exploration weight from 0.4 down to 0.05
                dep_weight = max(0.05, 0.4 * (1.0 - global_step / args.total_timesteps))
                dep_act = dep_controller.step(act_np)
                combined_act = (1.0 - dep_weight) * act_np + dep_weight * dep_act
                step_act = np.clip(combined_act, -1.0, 1.0)
            else:
                step_act = act_np

            # Step headless MuJoCo physics
            next_obs, reward, terminated, truncated, step_info = env.step(step_act)
            done = terminated or truncated

            buffer.add(
                obs=obs,
                action=act_np,
                reward=reward,
                value=value.squeeze(0).cpu(),
                log_prob=log_prob.squeeze(0).cpu(),
                done=done,
            )

            obs = next_obs
            episode_return += reward
            episode_length += 1

            if done:
                episode_count += 1
                forward_vel = step_info.get("forward_velocity", 0.0)
                writer.add_scalar("charts/episodic_return", episode_return, global_step)
                writer.add_scalar("charts/episodic_length", episode_length, global_step)
                writer.add_scalar("charts/forward_velocity", forward_vel, global_step)

                if episode_return > best_return:
                    best_return = episode_return
                    torch.save(
                        {
                            "model_state_dict": policy.state_dict(),
                            "optimizer_state_dict": trainer.optimizer.state_dict(),
                            "model_type": args.model,
                            "global_step": global_step,
                            "best_return": best_return,
                            "args": vars(args),
                        },
                        best_model_path,
                    )

                obs, info = env.reset()
                episode_return = 0.0
                episode_length = 0
                if dep_controller is not None:
                    dep_controller.reset()

        # Phase B: Advantage & Return Computation (GAE-λ)
        with torch.no_grad():
            last_obs_tensor = torch.from_numpy(obs).unsqueeze(0).float().to(device)
            last_value = policy.get_value(last_obs_tensor).squeeze(0).cpu()

        buffer.compute_returns_and_advantages(last_value=last_value, last_done=done)

        # Phase C: PPO Optimization Update
        metrics = trainer.train_step(
            buffer=buffer,
            batch_size=args.batch_size,
            update_epochs=args.update_epochs,
        )

        # Telemetry Logging
        sps = int(global_step / (time.time() - start_time))
        writer.add_scalar("losses/policy_loss", metrics["policy_loss"], global_step)
        writer.add_scalar("losses/value_loss", metrics["value_loss"], global_step)
        writer.add_scalar("losses/entropy", metrics["entropy"], global_step)
        writer.add_scalar("losses/approx_kl", metrics["approx_kl"], global_step)
        writer.add_scalar("losses/clip_fraction", metrics["clip_fraction"], global_step)
        writer.add_scalar("losses/explained_variance", metrics["explained_variance"], global_step)
        writer.add_scalar("charts/SPS", sps, global_step)

        # Console Progress Report
        if update % args.log_interval == 0 or update == num_updates:
            print(
                f"Update {update:4d}/{num_updates} | Step: {global_step:7d} | "
                f"SPS: {sps:4d} | Return: {episode_return:6.2f} (Best: {best_return:6.2f}) | "
                f"Loss: {metrics['policy_loss']:+.4f} | Value: {metrics['value_loss']:6.2f} | "
                f"KL: {metrics['approx_kl']:.4f}"
            )

    env.close()
    writer.close()

    # Final checkpoint
    final_model_path = save_dir / f"{args.model}_final.pt"
    torch.save(
        {
            "model_state_dict": policy.state_dict(),
            "model_type": args.model,
            "global_step": global_step,
            "best_return": best_return,
            "args": vars(args),
        },
        final_model_path,
    )

    print("\n[4/4] Training Complete!")
    print(f"  Best Model Saved to:  {best_model_path}")
    print(f"  Final Model Saved to: {final_model_path}")
    print(f"  TensorBoard Logs in:  {log_dir}")
    print("=" * 80)
    return best_model_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser(description="PPO Training Pipeline for Biomechanical Drosophila")

    # Architecture & Environment
    parser.add_argument(
        "--model",
        type=str,
        default="connectome",
        choices=["mlp", "connectome", "rnn", "gnn", "hierarchical"],
        help="Policy architecture to train",
    )
    parser.add_argument("--substeps", type=int, default=20, help="MuJoCo physics substeps per action step")
    parser.add_argument("--max-episode-steps", type=int, default=1000, help="Maximum steps per episode")

    # Training Timesteps & Buffer
    parser.add_argument("--total-timesteps", type=int, default=50000, help="Total environment steps")
    parser.add_argument("--num-steps", type=int, default=1024, help="Rollout buffer size per update")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size for SGD")
    parser.add_argument("--update-epochs", type=int, default=10, help="Number of SGD passes per rollout")

    # Hyperparameters
    parser.add_argument("--lr", type=float, default=3e-4, help="Adam learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda parameter")
    parser.add_argument("--clip-coef", type=float, default=0.2, help="PPO clipping epsilon")
    parser.add_argument("--clip-vloss", action="store_true", default=True, help="Clip value function loss")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy bonus coefficient")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value function loss coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Maximum gradient norm")
    parser.add_argument("--target-kl", type=float, default=None, help="Target KL divergence for early stopping")

    # DEP Exploration
    parser.add_argument("--use-dep", action="store_true", default=False, help="Enable DEP-RL motor synergy exploration")

    # Experiment Management
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-cuda", action="store_true", default=False, help="Force CPU execution")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--log-interval", type=int, default=1, help="Console progress log interval")

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())

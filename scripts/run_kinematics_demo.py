"""Drosophila Locomotion Kinematics and Gait Analysis Demonstration.

Executes a simulation rollout, analyzes 6-leg stepping patterns using KinematicAnalyzer,
and exports:
  1. Comprehensive Biomechanical Gait Scorecard (terminal output).
  2. Publication-grade Footfall Diagram (outputs/footfall_diagram.png).
  3. Optional live interactive 3D viewer with gait telemetry.

Usage:
  # Headless analysis and footfall diagram export:
  wsl -e /home/anish/miniconda3/envs/connectome-rl/bin/python scripts/run_kinematics_demo.py

  # With a trained checkpoint:
  wsl -e /home/anish/miniconda3/envs/connectome-rl/bin/python scripts/run_kinematics_demo.py --checkpoint checkpoints/connectome_final.pt

  # With an in-silico lesion (e.g., knock out DNa01 Left):
  wsl -e /home/anish/miniconda3/envs/connectome-rl/bin/python scripts/run_kinematics_demo.py --lesion DNa01 --side left
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch

from connectome_rl.src.analysis.kinematics import KinematicAnalyzer
from connectome_rl.src.analysis.lesion import LesionConfig, LesionController
from connectome_rl.src.envs.fly_wrapper import FlyLocomotionEnv
from connectome_rl.src.models.connectome_policy import ConnectomePolicy


def run_kinematics_demo(
    checkpoint: str | None = None,
    lesion_target: str | None = None,
    lesion_side: str = "left",
    num_steps: int = 100,
    physics_substeps: int = 20,
    contact_threshold: float = 0.5,
    out_plot: str = "outputs/footfall_diagram.png",
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    circuit_path = project_root / "connectome_rl" / "data" / "dna_circuit_tensors.pt"
    if not circuit_path.exists():
        raise FileNotFoundError(f"Circuit tensors not found at: {circuit_path}")

    print("=" * 80)
    print("  DROSOPHILA BIOMECHANICAL LOCOMOTION KINEMATICS DEMO")
    print(f"  Condition:       {'INTACT HEALTHY' if not lesion_target else f'LESION {lesion_target} ({lesion_side.upper()})'}")
    print(f"  Checkpoint:      {checkpoint if checkpoint else 'Biological Baseline'}")
    print(f"  Evaluation:      {num_steps} policy decision steps ({num_steps * physics_substeps * 0.0001:.2f}s simulated physics)")
    print("=" * 80)

    # 1. Load Policy
    print("\n[1/3] Instantiating Connectome Policy...")
    policy = ConnectomePolicy(obs_dim=100, act_dim=42, circuit_data=circuit_path)
    if checkpoint is not None:
        ckpt_path = project_root / checkpoint
        if ckpt_path.exists():
            data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            policy.load_state_dict(data["model_state_dict"])
            print(f"      Loaded trained weights from: {ckpt_path}")
        else:
            print(f"      [!] Checkpoint not found at {ckpt_path}, using baseline.")
    policy.eval()

    # Apply lesion if requested
    lesion_ctrl = None
    if lesion_target is not None:
        lesion_ctrl = LesionController(policy=policy, circuit_data=circuit_path)
        ablated = lesion_ctrl.apply_lesion(LesionConfig(target_type=lesion_target, side=lesion_side))
        print(f"      Ablated {lesion_target} ({lesion_side}) target columns: {ablated}")

    # 2. Run Simulation Rollout
    print("\n[2/3] Simulating Locomotion in MuJoCo Physics...")
    env = FlyLocomotionEnv(
        physics_steps_per_action=physics_substeps,
        init_pose="tripod",
        max_episode_steps=num_steps + 10,
        enable_render=False,
    )

    obs, _ = env.reset(seed=42)
    dt = physics_substeps * 0.0001  # e.g. 20 * 0.0001 = 0.002s

    contact_history: list[np.ndarray] = []
    position_history: list[np.ndarray] = []
    yaw_history: list[float] = []

    init_heading = obs[84:87].copy()

    for step in range(num_steps):
        # Observation indices 93..98 are leg contact forces: [LF, LM, LH, RF, RM, RH]
        contact_forces = obs[93:99].copy()
        contact_history.append(contact_forces)

        # Fly center of mass position (mm)
        fly_pos = env.sim.physics.data.qpos[:3].copy() * 1000.0
        position_history.append(fly_pos)

        # Yaw heading
        cur_h = obs[84:87].copy()
        a0 = np.arctan2(init_heading[1], init_heading[0])
        a1 = np.arctan2(cur_h[1], cur_h[0])
        yaw_history.append(float(np.degrees(np.arctan2(np.sin(a1 - a0), np.cos(a1 - a0)))))

        # Forward pass through connectome policy
        with torch.no_grad():
            action, _, _, _ = policy.get_action_and_value(torch.from_numpy(obs).unsqueeze(0).float())
        act_np = action.squeeze(0).numpy()

        obs, _, terminated, truncated, _ = env.step(act_np)
        if terminated or truncated:
            break

    env.close()
    if lesion_ctrl is not None:
        lesion_ctrl.restore()

    contacts_arr = np.array(contact_history, dtype=np.float32)
    positions_arr = np.array(position_history, dtype=np.float32)
    yaws_arr = np.array(yaw_history, dtype=np.float32)

    # 3. Kinematic Analysis
    print("\n[3/3] Performing Gait & Kinematic Analysis...")
    analyzer = KinematicAnalyzer(contact_force_threshold=contact_threshold)
    metrics = analyzer.analyze_rollout(
        contact_forces=contacts_arr,
        positions=positions_arr,
        yaws=yaws_arr,
        dt=dt,
    )

    # Print Formatted Telemetry Scorecard
    print("\n" + metrics.summary())

    # Export Footfall Diagram
    out_file = project_root / out_plot
    out_file.parent.mkdir(parents=True, exist_ok=True)
    cond_title = f"{'Intact Control' if not lesion_target else f'Lesion {lesion_target} ({lesion_side.upper()})'}"
    title_str = (
        f"6-Leg Footfall Diagram ({cond_title})\n"
        f"Tripod Index (TCI): {metrics.tripod_index:.3f} | Step Freq: {metrics.mean_step_frequency_hz:.1f} Hz"
    )
    analyzer.plot_footfall_diagram(
        contact_forces=contacts_arr,
        dt=dt,
        title=title_str,
        out_path=out_file,
    )
    print(f"\n[+] Generated biological footfall gait diagram saved to:\n    {out_file}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Kinematics & Gait Analysis Demo")
    parser.add_argument("--checkpoint", type=str, default=None, help="Trained checkpoint path")
    parser.add_argument("--lesion", type=str, default=None, choices=["DNa01", "DNa02", "random"], help="Neuron to ablate")
    parser.add_argument("--side", type=str, default="left", choices=["left", "right", "bilateral"], help="Ablation side")
    parser.add_argument("--steps", type=int, default=100, help="Simulation steps")
    parser.add_argument("--substeps", type=int, default=20, help="Physics substeps")
    parser.add_argument("--threshold", type=float, default=0.5, help="Ground contact force threshold (mN)")
    parser.add_argument("--out", type=str, default="outputs/footfall_diagram.png", help="Output plot path")

    args = parser.parse_args()
    run_kinematics_demo(
        checkpoint=args.checkpoint,
        lesion_target=args.lesion,
        lesion_side=args.side,
        num_steps=args.steps,
        physics_substeps=args.substeps,
        contact_threshold=args.threshold,
        out_plot=args.out,
    )

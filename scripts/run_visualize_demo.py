"""Drosophila Scientific Visualization and Publication Figures Demonstration.

Integrates the full visualization suite:
  1. Synchronized side-by-side animated video with live telemetry HUD overlay
     (outputs/side_by_side_comparison.gif).
  2. Multi-panel publication-grade scientific dashboard:
     - 2D trajectory arena tracking with heading orientation.
     - 6-leg stance/swing footfall raster diagram (tripod alternation).
     - Quantitative bar charts (forward speed vs steering asymmetry).
     - Bilateral leg duty factor symmetry profiles.
     (outputs/publication_dashboard.png).
  3. In-silico connectome lesion impact summary scorecard
     (outputs/lesion_impact_summary.png).

Usage:
  # Run full visualization pipeline:
  wsl -e /home/anish/miniconda3/envs/connectome-rl/bin/python scripts/run_visualize_demo.py

  # With a trained checkpoint:
  wsl -e /home/anish/miniconda3/envs/connectome-rl/bin/python scripts/run_visualize_demo.py --checkpoint checkpoints/connectome_final.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
import numpy as np
import torch

from connectome_rl.src.analysis.kinematics import KinematicAnalyzer
from connectome_rl.src.analysis.lesion import LesionConfig, LesionController
from connectome_rl.src.analysis.visualize import (
    compose_side_by_side_video,
    plot_connectome_lesion_summary,
    plot_publication_dashboard,
)
from connectome_rl.src.envs.fly_wrapper import FlyLocomotionEnv
from connectome_rl.src.models.connectome_policy import ConnectomePolicy


def run_condition_rollout(
    policy: ConnectomePolicy,
    lesion_ctrl: LesionController | None,
    lesion_cfg: LesionConfig | None,
    num_steps: int = 50,
    physics_substeps: int = 20,
    enable_render: bool = True,
) -> dict[str, Any]:
    """Execute a single simulation rollout with telemetry and rendered frames."""
    env = FlyLocomotionEnv(
        physics_steps_per_action=physics_substeps,
        init_pose="tripod",
        max_episode_steps=num_steps + 10,
        enable_render=enable_render,
        render_camera_name="camera_top",
    )

    if lesion_ctrl is not None and lesion_cfg is not None:
        lesion_ctrl.apply_lesion(lesion_cfg)
    elif lesion_ctrl is not None:
        lesion_ctrl.restore_all()

    obs, _ = env.reset(seed=42)
    dt = physics_substeps * 0.0001

    frames: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    contacts: list[np.ndarray] = []
    yaws: list[float] = []

    init_heading = obs[84:87].copy()

    for _ in range(num_steps):
        # Record camera frame
        if enable_render:
            frame = env.render()
            if frame is not None:
                frames.append(frame)

        # Record center of mass position (mm)
        fly_pos = env.sim.physics.data.qpos[:3].copy() * 1000.0
        positions.append(fly_pos)

        # Record foot contact forces [LF, LM, LH, RF, RM, RH]
        contact_forces = obs[93:99].copy()
        contacts.append(contact_forces)

        # Record heading yaw
        curr_heading = obs[84:87]
        yaw_rad = np.arctan2(
            init_heading[0] * curr_heading[1] - init_heading[1] * curr_heading[0],
            init_heading[0] * curr_heading[0] + init_heading[1] * curr_heading[1],
        )
        yaws.append(float(np.degrees(yaw_rad)))

        # Step policy
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
            if hasattr(policy, "forward_actor"):
                action_mean, _ = policy.forward_actor(obs_tensor)
                action = action_mean.squeeze(0).cpu().numpy()
            else:
                out = policy.get_action_and_value(obs_tensor)
                action = out[0].squeeze(0).cpu().numpy()

        obs, _, done, truncated, _ = env.step(action)
        if done or truncated:
            break

    env.close()

    # Restore policy after rollout
    if lesion_ctrl is not None:
        lesion_ctrl.restore_all()

    analyzer = KinematicAnalyzer()
    traj_arr = np.array(positions)
    contact_arr = np.array(contacts)
    metrics = analyzer.analyze_rollout(
        contact_forces=contact_arr,
        positions=traj_arr,
        yaws=np.array(yaws),
        dt=dt,
    )

    return {
        "frames": frames,
        "positions": traj_arr,
        "contacts": contact_arr,
        "yaws": yaws,
        "metrics": metrics,
        "dt": dt,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Drosophila Locomotion Visualization Suite Demo")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained policy checkpoint")
    parser.add_argument("--steps", type=int, default=50, help="Simulation steps per condition")
    parser.add_argument("--fps", type=int, default=8, help="Output video framerate")
    parser.add_argument("--out-dir", type=str, default="outputs", help="Output directory")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    circuit_path = project_root / "connectome_rl" / "data" / "dna_circuit_tensors.pt"
    out_dir = project_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  DROSOPHILA CONNECTOME VISUALIZATION & PUBLICATION DASHBOARD")
    print(f"  Checkpoint:        {args.checkpoint if args.checkpoint else 'Biological Baseline'}")
    print(f"  Steps per trial:   {args.steps}")
    print(f"  Destination:       {out_dir}")
    print("=" * 80)

    # 1. Load Policy
    print("\n[1/4] Loading Connectome Policy...")
    policy = ConnectomePolicy(obs_dim=100, act_dim=42, circuit_data=circuit_path)
    if args.checkpoint is not None:
        ckpt_file = project_root / args.checkpoint
        if ckpt_file.exists():
            data = torch.load(ckpt_file, map_location="cpu", weights_only=False)
            policy.load_state_dict(data["model_state_dict"])
            print(f"      Loaded weights: {ckpt_file}")
    policy.eval()
    lesion_ctrl = LesionController(policy=policy, circuit_data=circuit_path)

    # 2. Rollout 1: Intact Fly
    print("\n[2/4] Simulating Intact Control Fly...")
    intact_data = run_condition_rollout(
        policy=policy,
        lesion_ctrl=lesion_ctrl,
        lesion_cfg=None,
        num_steps=args.steps,
        enable_render=True,
    )
    print(f"      Intact speed: {intact_data['metrics'].mean_forward_velocity_mm_s:.2f} mm/s | Net yaw: {intact_data['metrics'].net_yaw_angle_deg:+.2f}°")

    # 3. Rollout 2: In-silico Lesion (DNa01 Left unilateral knockout)
    print("\n[3/4] Simulating In-Silico Lesioned Fly (DNa01 Left Knockout)...")
    lesion_cfg = LesionConfig(target_type="DNa01", side="left")
    lesioned_data = run_condition_rollout(
        policy=policy,
        lesion_ctrl=lesion_ctrl,
        lesion_cfg=lesion_cfg,
        num_steps=args.steps,
        enable_render=True,
    )
    print(f"      Lesioned speed: {lesioned_data['metrics'].mean_forward_velocity_mm_s:.2f} mm/s | Net yaw: {lesioned_data['metrics'].net_yaw_angle_deg:+.2f}°")

    # 4. Generate Visual Deliverables
    print("\n[4/4] Generating Publication Figures & Side-by-Side Video...")

    # A. Side-by-side synchronized video with HUD banner
    video_out = out_dir / "side_by_side_comparison.gif"
    compose_side_by_side_video(
        frames_left=intact_data["frames"],
        frames_right=lesioned_data["frames"],
        label_left="INTACT CONTROL",
        label_right="DNa01-L ABLATED",
        yaws_left=intact_data["yaws"],
        yaws_right=lesioned_data["yaws"],
        fps=args.fps,
        out_path=video_out,
    )
    print(f"      [✓] Animated video with HUD: {video_out}")

    # B. Multi-panel publication dashboard
    dashboard_out = out_dir / "publication_dashboard.png"
    trajectories = {
        "Intact": intact_data["positions"][:, :2],
        "DNa01 Left": lesioned_data["positions"][:, :2],
    }
    metrics_by_cond = {
        "Intact": intact_data["metrics"],
        "DNa01 Left": lesioned_data["metrics"],
    }
    plot_publication_dashboard(
        trajectories=trajectories,
        footfall_contacts=intact_data["contacts"],
        dt=intact_data["dt"],
        metrics_by_condition=metrics_by_cond,
        title="Drosophila Biomechanical Locomotion & Connectome Analysis",
        out_path=dashboard_out,
    )
    print(f"      [✓] Publication Dashboard:  {dashboard_out}")

    # C. Lesion impact summary scorecard
    summary_out = out_dir / "lesion_impact_summary.png"
    env_eval = FlyLocomotionEnv(
        physics_steps_per_action=20,
        init_pose="tripod",
        max_episode_steps=args.steps + 5,
        enable_render=False,
    )
    eval_battery = [
        LesionConfig(target_type="intact", side="both"),
        LesionConfig(target_type="DNa01", side="left"),
        LesionConfig(target_type="DNa01", side="both"),
        LesionConfig(target_type="DNa02", side="both"),
        LesionConfig(target_type="random", side="both", intensity=0.5),
    ]
    results = [lesion_ctrl.evaluate_lesion(env=env_eval, config=cfg, num_steps=args.steps) for cfg in eval_battery]
    env_eval.close()
    plot_connectome_lesion_summary(results=results, out_path=summary_out)
    print(f"      [✓] Lesion Scorecard:       {summary_out}")

    print("\n" + "=" * 80)
    print("  VISUALIZATION PIPELINE COMPLETE")
    print(f"  Outputs generated:")
    print(f"    - {video_out}")
    print(f"    - {dashboard_out}")
    print(f"    - {summary_out}")
    print("=" * 80)


if __name__ == "__main__":
    main()

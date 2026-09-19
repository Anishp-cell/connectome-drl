"""Systematic In-Silico Lesion Battery Demonstration (Multi-Seed).

Executes a full computational neuroscience lesion battery on the trained
connectome policy across multiple random seeds (representing biological replicates):
  1. Intact Control (healthy biological baseline)
  2. DNa01 Left Knockout (steering asymmetry: contralateral turn bias)
  3. DNa01 Right Knockout (steering asymmetry: ipsilateral turn bias)
  4. DNa01 Bilateral Knockout (complete steering channel loss)
  5. DNa02 Bilateral Knockout (forward velocity channel ablation)
  6. Random Knockdown Control (50% random synaptic knockdown negative control)

Outputs:
  - outputs/systematic_lesion_battery.png (4-panel publication figure with error bars)
  - outputs/systematic_lesion_battery.json (Statistical mean ± std summaries and trial data)
  - outputs/systematic_lesion_battery.gif (Synchronized side-by-side comparison video)

Usage:
  wsl -e /home/anish/miniconda3/envs/connectome-rl/bin/python scripts/run_systematic_lesion_battery.py --steps 35 --seeds 42 43 44
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from connectome_rl.src.analysis.battery import (
    ConditionSummaryStats,
    LesionBatteryTrial,
    SystematicLesionBattery,
    plot_lesion_battery_results,
    save_battery_results,
)
from connectome_rl.src.analysis.benchmark import extract_deterministic_action
from connectome_rl.src.analysis.visualize import compose_side_by_side_video
from connectome_rl.src.envs.fly_wrapper import FlyLocomotionEnv
from connectome_rl.src.models.connectome_policy import ConnectomePolicy


def render_comparison_rollout(
    policy: torch.nn.Module,
    battery: SystematicLesionBattery,
    spec: dict[str, Any],
    num_steps: int = 35,
    seed: int = 42,
) -> tuple[list[np.ndarray], list[float]]:
    """Execute a single rollout with camera frames recorded for video export."""
    env = FlyLocomotionEnv(
        physics_steps_per_action=20,
        init_pose="tripod",
        max_episode_steps=num_steps + 10,
        enable_render=True,
        render_camera_name="camera_top",
    )
    obs, _ = env.reset(seed=seed)
    device = next(policy.parameters()).device

    frames: list[np.ndarray] = []
    yaws: list[float] = []
    init_heading = obs[84:87].copy()

    from connectome_rl.src.analysis.lesion import LesionConfig
    cfg = LesionConfig(
        target_type=spec["target"],
        side=spec.get("side", "both"),
        intensity=spec.get("intensity", 1.0),
    )
    battery.controller.apply_lesion(cfg)

    try:
        for _ in range(num_steps):
            frame = env.render()
            if frame is not None:
                frames.append(frame)

            curr_heading = obs[84:87]
            yaw_rad = np.arctan2(
                init_heading[0] * curr_heading[1] - init_heading[1] * curr_heading[0],
                init_heading[0] * curr_heading[0] + init_heading[1] * curr_heading[1],
            )
            yaws.append(float(np.degrees(yaw_rad)))

            obs_t = torch.from_numpy(obs).unsqueeze(0).float().to(device)
            act = extract_deterministic_action(policy, obs_t)
            act = np.clip(act, -1.0, 1.0)

            next_obs, _, done, truncated, _ = env.step(act)
            obs = next_obs
            if done or truncated:
                break
    finally:
        battery.controller.restore()
        env.close()

    return frames, yaws


def main() -> None:
    parser = argparse.ArgumentParser(description="Systematic In-Silico Lesion Battery Suite")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/connectome_final.pt", help="Policy checkpoint")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44], help="Random seeds for replication")
    parser.add_argument("--steps", type=int, default=35, help="Simulation steps per trial")
    parser.add_argument("--out-dir", type=str, default="outputs", help="Output directory")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    circuit_path = project_root / "connectome_rl" / "data" / "dna_circuit_tensors.pt"
    out_dir = project_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  PHASE 6: SYSTEMATIC IN-SILICO LESION BATTERY (MULTI-SEED REPLICATION)")
    print(f"  Checkpoint:        {args.checkpoint}")
    print(f"  Evaluation Seeds:  {args.seeds} (N = {len(args.seeds)} biological replicates)")
    print(f"  Steps per trial:   {args.steps}")
    print(f"  Destination:       {out_dir}")
    print("=" * 80)

    # 1. Load Policy
    print("\n[1/4] Loading Trained Biological Connectome Policy...")
    policy = ConnectomePolicy(obs_dim=100, act_dim=42, circuit_data=circuit_path)
    ckpt_path = project_root / args.checkpoint
    if ckpt_path.exists():
        data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        policy.load_state_dict(data["model_state_dict"])
        print(f"      Loaded weights from: {ckpt_path}")
    else:
        print("      [!] Checkpoint not found, using biological baseline weights.")
    policy.eval()

    battery = SystematicLesionBattery(policy=policy, circuit_data=circuit_path)

    # 2. Run Headless Multi-Seed Battery
    print(f"\n[2/4] Executing Systematic Lesion Trials ({len(battery.battery_specs)} conditions × {len(args.seeds)} seeds = {len(battery.battery_specs) * len(args.seeds)} trials)...")
    env = FlyLocomotionEnv(physics_steps_per_action=20, init_pose="tripod", max_episode_steps=args.steps + 10, enable_render=False)
    trials, summaries = battery.run_battery(env=env, seeds=args.seeds, num_steps=args.steps)
    env.close()

    # 3. Generate Video Replay: Intact vs DNa01 Left
    print("\n[3/4] Rendering Side-by-Side Video Replay (Intact vs DNa01 Left)...")
    spec_intact = battery.battery_specs[0]
    spec_dna01_l = battery.battery_specs[1]
    frames_intact, yaws_intact = render_comparison_rollout(policy, battery, spec_intact, num_steps=args.steps, seed=args.seeds[0])
    frames_dna01_l, yaws_dna01_l = render_comparison_rollout(policy, battery, spec_dna01_l, num_steps=args.steps, seed=args.seeds[0])

    video_path = out_dir / "systematic_lesion_battery.gif"
    compose_side_by_side_video(
        frames_left=frames_intact,
        frames_right=frames_dna01_l,
        label_left="INTACT CONTROL",
        label_right="DNa01 LEFT ABLATED",
        yaws_left=yaws_intact,
        yaws_right=yaws_dna01_l,
        fps=8,
        out_path=video_path,
    )
    print(f"      [✓] Synchronized Video:  {video_path}")

    # 4. Export Deliverables
    print("\n[4/4] Exporting Scientific Deliverables & Summary Scorecard...")
    json_path = out_dir / "systematic_lesion_battery.json"
    save_battery_results(trials=trials, summaries=summaries, out_path=json_path)
    print(f"      [✓] Statistical Data:    {json_path}")

    plot_path = out_dir / "systematic_lesion_battery.png"
    plot_lesion_battery_results(summaries=summaries, out_path=plot_path)
    print(f"      [✓] Publication Figure:  {plot_path}")

    # 5. Terminal Scorecard Table
    print("\n" + "=" * 90)
    print("  SYSTEMATIC IN-SILICO LESION BATTERY: STATISTICAL SCORECARD (MEAN ± STD)")
    print("=" * 90)
    print(f"  {'Experimental Condition':<28} | {'Forward (mm)':<16} | {'Yaw Turn (|deg|)':<18} | {'Speed (mm/s)':<18}")
    print("-" * 90)
    for s in summaries:
        fwd_str = f"{s.forward_dist_mean:+.1f} ± {s.forward_dist_std:.1f}"
        yaw_str = f"{s.yaw_deg_mean:.1f}° ± {s.yaw_deg_std:.1f}°"
        spd_str = f"{s.speed_mean:+.1f} ± {s.speed_std:.1f}"
        print(f"  {s.condition_name:<28} | {fwd_str:<16} | {yaw_str:<18} | {spd_str:<18}")
    print("=" * 90)


if __name__ == "__main__":
    main()

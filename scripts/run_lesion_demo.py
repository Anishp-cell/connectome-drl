"""Demonstration of In-Silico Lesion Experiments on Drosophila Locomotion.

Compares an Intact Fly against a Lesioned Fly (e.g. knocking out DNa01-L)
with clear slow-motion video playback and on-screen biological annotations.

Outputs:
  - Visual annotated comparison video in outputs/lesion_comparison.gif
  - Quantitative telemetry table showing forward displacement, drift, and yaw turning.

Usage:
  wsl -e /home/anish/miniconda3/envs/connectome-rl/bin/python scripts/run_lesion_demo.py --lesion DNa01 --side left --fps 8
  wsl -e /home/anish/miniconda3/envs/connectome-rl/bin/python scripts/run_lesion_demo.py --lesion DNa01 --side left --gui
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from connectome_rl.src.analysis.lesion import LesionConfig, LesionController
from connectome_rl.src.envs.fly_wrapper import FlyLocomotionEnv
from connectome_rl.src.models.connectome_policy import ConnectomePolicy


def draw_label_on_frame(
    frame: np.ndarray,
    label_left: str,
    label_right: str,
    step_num: int,
    dyaw_left: float,
    dyaw_right: float,
) -> np.ndarray:
    """Overlay clean visual banners on top of the side-by-side comparison video."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    w, h = img.size
    mid_x = w // 2

    # Draw dark header banner
    banner_h = 42
    draw.rectangle([(0, 0), (w, banner_h)], fill=(20, 24, 32, 220))
    draw.line([(mid_x, 0), (mid_x, h)], fill=(255, 255, 255), width=3)

    # Text annotations
    text_l = f"[1] {label_left} | Yaw: {dyaw_left:+.1f}°"
    text_r = f"[2] {label_right} | Yaw: {dyaw_right:+.1f}°"
    step_text = f"Step {step_num}"

    draw.text((20, 12), text_l, fill=(100, 220, 255))
    draw.text((mid_x + 20, 12), text_r, fill=(255, 120, 100))
    draw.text((mid_x - 35, 12), step_text, fill=(200, 200, 200))

    return np.array(img)


def run_lesion_demo(
    target_type: str = "DNa01",
    side: str = "left",
    checkpoint: str | None = None,
    num_steps: int = 60,
    physics_substeps: int = 20,
    fps: int = 8,
    output_path: str = "outputs/lesion_comparison.gif",
) -> None:
    print("=" * 80)
    print("  IN-SILICO LESION COMPARISON DEMO: Drosophila Locomotion")
    print(f"  Target Neuron:     {target_type.upper()}")
    print(f"  Ablation Side:     {side.upper()}")
    print(f"  Checkpoint:        {checkpoint if checkpoint else 'Biological Default'}")
    print(f"  Simulation Steps:  {num_steps} steps per condition")
    print(f"  Video Playback:    Slow-motion {fps} FPS (easy to observe leg strides)")
    print("=" * 80)

    project_root = Path(__file__).resolve().parents[1]
    circuit_path = project_root / "connectome_rl" / "data" / "dna_circuit_tensors.pt"
    if not circuit_path.exists():
        raise FileNotFoundError(f"Circuit tensors not found at: {circuit_path}")

    # 1. Initialize Policy
    print("\n[1/4] Loading Connectome Policy...")
    policy = ConnectomePolicy(obs_dim=100, act_dim=42, circuit_data=circuit_path)

    if checkpoint is not None:
        ckpt_path = project_root / checkpoint
        if ckpt_path.exists():
            ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            policy.load_state_dict(ckpt_data["model_state_dict"])
            print(f"      Loaded trained weights from: {ckpt_path}")
        else:
            print(f"      [!] Checkpoint not found at {ckpt_path}, using biological baseline weights.")

    policy.eval()

    # 2. Initialize Controller
    lesion_ctrl = LesionController(policy=policy, circuit_data=circuit_path)

    # 3. Environment
    print("\n[2/4] Initializing FlyGym MuJoCo Simulation...")
    env = FlyLocomotionEnv(
        physics_steps_per_action=physics_substeps,
        init_pose="tripod",
        max_episode_steps=num_steps + 20,
        enable_render=True,
        render_camera_name="camera_top",
    )

    # -------------------------------------------------------------
    # Condition A: Intact Control
    # -------------------------------------------------------------
    print(f"\n[3/4] Running Condition 1: INTACT FLY (Healthy Control)...")
    obs, info = env.reset(seed=42)
    intact_frames: list[np.ndarray] = []
    intact_yaws: list[float] = []
    intact_init_pos = env.sim.physics.data.qpos[:3].copy()
    intact_init_heading = obs[84:87].copy()

    for step in range(1, num_steps + 1):
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).float()
        with torch.no_grad():
            action, _, _, _ = policy.get_action_and_value(obs_tensor)
        act_np = action.squeeze(0).numpy()

        obs, reward, terminated, truncated, _ = env.step(act_np)
        frame = env.render()
        if frame is not None:
            intact_frames.append(frame)

        cur_h = obs[84:87].copy()
        a0 = np.arctan2(intact_init_heading[1], intact_init_heading[0])
        a1 = np.arctan2(cur_h[1], cur_h[0])
        intact_yaws.append(float(np.degrees(np.arctan2(np.sin(a1 - a0), np.cos(a1 - a0)))))

    intact_final_pos = env.sim.physics.data.qpos[:3].copy()
    intact_dx = float(intact_final_pos[0] - intact_init_pos[0]) * 1000.0
    intact_dy = float(intact_final_pos[1] - intact_init_pos[1]) * 1000.0
    intact_dyaw = intact_yaws[-1] if intact_yaws else 0.0

    # -------------------------------------------------------------
    # Condition B: Lesioned Fly
    # -------------------------------------------------------------
    config = LesionConfig(target_type=target_type, side=side)
    ablated_cols = lesion_ctrl.apply_lesion(config)
    print(f"\n[4/4] Running Condition 2: LESIONED FLY ({target_type} {side.upper()} Knockout, cols={ablated_cols})...")

    obs, info = env.reset(seed=42)
    lesion_frames: list[np.ndarray] = []
    lesion_yaws: list[float] = []
    lesion_init_pos = env.sim.physics.data.qpos[:3].copy()
    lesion_init_heading = obs[84:87].copy()

    for step in range(1, num_steps + 1):
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).float()
        with torch.no_grad():
            action, _, _, _ = policy.get_action_and_value(obs_tensor)
        act_np = action.squeeze(0).numpy()

        obs, reward, terminated, truncated, _ = env.step(act_np)
        frame = env.render()
        if frame is not None:
            lesion_frames.append(frame)

        cur_h = obs[84:87].copy()
        b0 = np.arctan2(lesion_init_heading[1], lesion_init_heading[0])
        b1 = np.arctan2(cur_h[1], cur_h[0])
        lesion_yaws.append(float(np.degrees(np.arctan2(np.sin(b1 - b0), np.cos(b1 - b0)))))

    lesion_final_pos = env.sim.physics.data.qpos[:3].copy()
    lesion_dx = float(lesion_final_pos[0] - lesion_init_pos[0]) * 1000.0
    lesion_dy = float(lesion_final_pos[1] - lesion_init_pos[1]) * 1000.0
    lesion_dyaw = lesion_yaws[-1] if lesion_yaws else 0.0

    # Restore policy
    lesion_ctrl.restore()
    env.close()

    # -------------------------------------------------------------
    # Comparative Telemetry Table
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  QUANTITATIVE IN-SILICO LESION COMPARISON RESULTS")
    print("=" * 80)
    print(f"{'Condition':<25} | {'Forward (mm)':<14} | {'Lateral (mm)':<14} | {'Net Yaw Turn (deg)':<18}")
    print("-" * 80)
    print(f"{'1. Intact Control':<25} | {intact_dx:<14.3f} | {intact_dy:<14.3f} | {intact_dyaw:<+18.2f}")
    print(f"{f'2. {target_type} ({side.upper()})':<25} | {lesion_dx:<14.3f} | {lesion_dy:<14.3f} | {lesion_dyaw:<+18.2f}")
    print("-" * 80)

    yaw_diff = lesion_dyaw - intact_dyaw
    print(f"BIOLOGICAL EFFECT OBSERVED:")
    if abs(yaw_diff) > 2.0:
        turn_dir = "RIGHT (clockwise)" if yaw_diff < 0 else "LEFT (counterclockwise)"
        print(f"  [+] Asymmetric Steering Bias: Lesioning {side} {target_type} caused the fly to veer {turn_dir} by {abs(yaw_diff):.1f}°!")
    else:
        speed_ratio = (lesion_dx / (intact_dx + 1e-6)) * 100.0
        print(f"  [+] Locomotion Speed Impact: Forward translation was {speed_ratio:.1f}% of intact baseline.")

    # -------------------------------------------------------------
    # Side-by-Side Video Composition with Banners
    # -------------------------------------------------------------
    out_file = project_root / output_path
    out_file.parent.mkdir(parents=True, exist_ok=True)

    min_len = min(len(intact_frames), len(lesion_frames))
    if min_len > 0:
        print(f"\nRendering Side-by-Side Annotated Video ({min_len} frames at slow-motion {fps} FPS) -> {out_file}...")
        combined_frames = []
        label_l = "INTACT CONTROL"
        label_r = f"{target_type.upper()} ({side.upper()})"

        for i in range(min_len):
            frame_left = intact_frames[i]
            frame_right = lesion_frames[i]
            # Horizontal concatenation
            combined = np.concatenate([frame_left, frame_right], axis=1)
            annotated = draw_label_on_frame(
                combined,
                label_left=label_l,
                label_right=label_r,
                step_num=i + 1,
                dyaw_left=intact_yaws[i],
                dyaw_right=lesion_yaws[i],
            )
            combined_frames.append(annotated)

        # Save slow motion GIF
        imageio.mimsave(str(out_file), combined_frames, fps=fps, loop=0)
        duration_sec = min_len / fps
        print(f"Successfully saved side-by-side demo: {out_file} (Duration: {duration_sec:.1f}s)!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run In-Silico Lesion Comparison Demo")
    parser.add_argument("--lesion", type=str, default="DNa01", choices=["DNa01", "DNa02", "random"], help="Neuron to ablate")
    parser.add_argument("--side", type=str, default="left", choices=["left", "right", "bilateral"], help="Ablation side")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained checkpoint (e.g. checkpoints/connectome_final.pt)")
    parser.add_argument("--steps", type=int, default=50, help="Simulation steps per condition")
    parser.add_argument("--substeps", type=int, default=20, help="Physics substeps per action")
    parser.add_argument("--fps", type=int, default=8, help="Playback speed in frames per second (default 8 = slow motion)")
    parser.add_argument("--out", type=str, default="outputs/lesion_comparison.gif", help="Output GIF path")
    parser.add_argument("--gui", action="store_true", help="Launch interactive 3D window for live visual inspection")
    parser.add_argument("--cam-dist", type=float, default=8.0, help="Camera distance in mm")

    args = parser.parse_args()

    if args.gui:
        from scripts.view_fly_interactive import launch_interactive_viewer
        launch_interactive_viewer(
            cam_distance=args.cam_dist,
            lesion_target=args.lesion,
            lesion_side=args.side,
            checkpoint=args.checkpoint,
        )
    else:
        run_lesion_demo(
            target_type=args.lesion,
            side=args.side,
            checkpoint=args.checkpoint,
            num_steps=args.steps,
            physics_substeps=args.substeps,
            fps=args.fps,
            output_path=args.out,
        )

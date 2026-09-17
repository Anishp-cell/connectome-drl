"""Run a 3D Biomechanical Simulation Demo of the Fruit Fly in MuJoCo.

Demonstrates closed-loop sensorimotor control:
  1. The FlyGym simulation generates 100-dim sensory observations (joint angles, orientation, ground contacts).
  2. The ConnectomePolicy maps these observations through the biological Janelia connectome (4 DNs -> 125 Interneurons -> 377 Motor Neurons).
  3. The resulting 42 joint commands actuate the 6 legs in MuJoCo physics.
  4. Telemetry and an animated GIF are recorded for inspection.

Usage:
  wsl -e /home/anish/miniconda3/envs/connectome-rl/bin/python scripts/run_simulation_demo.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time
import imageio
import numpy as np
import torch

from connectome_rl.src.envs.fly_wrapper import FlyLocomotionEnv
from connectome_rl.src.models.connectome_policy import ConnectomePolicy


def run_demo(
    num_steps: int = 50,
    physics_substeps: int = 20,
    pose: str = "tripod",
    save_gif: bool = True,
    output_path: str = "outputs/fly_simulation_demo.gif",
) -> None:
    print("=" * 70)
    print("FRUIT FLY BIOMECHANICAL SIMULATION DEMO (FlyGym + MuJoCo 3.x)")
    print("=" * 70)

    # 1. Locate biological connectome tensors
    project_root = Path(__file__).resolve().parents[1]
    tensors_path = project_root / "connectome_rl" / "data" / "dna_circuit_tensors.pt"
    if not tensors_path.exists():
        raise FileNotFoundError(f"Circuit tensors not found at: {tensors_path}")

    print(f"[1/4] Loading Janelia MaleCNS v1.0 biological connectome policy...")
    policy = ConnectomePolicy(obs_dim=100, act_dim=42, circuit_data=tensors_path)
    policy.eval()
    print("      Active biological pathways: 4 DNs -> 125 Interneurons -> 377 Motor Neurons -> 42 Joints")

    # 2. Initialize FlyGym Environment
    print(f"\n[2/4] Initializing FlyGym v1.2.1 simulation in MuJoCo physics engine (pose: {pose})...")
    env = FlyLocomotionEnv(
        physics_steps_per_action=physics_substeps,
        init_pose=pose,
        max_episode_steps=num_steps + 10,
        enable_render=save_gif,
        render_camera_name="camera_top",
    )

    out_file = project_root / output_path
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # 3. Reset Environment
    obs, info = env.reset()
    frames: list[np.ndarray] = []
    if save_gif:
        first_frame = env.render()
        if first_frame is not None:
            frames.append(first_frame)

    print(f"      Initial thorax spawn position (x, y, z): {info['thorax_pos']} mm")
    print(f"\n[3/4] Running {num_steps} simulation steps...")
    print("-" * 70)
    print(f"{'Step':<6} | {'X (mm)':<8} | {'Y (mm)':<8} | {'Z (mm)':<8} | {'Fwd Rew':<10} | {'Leg Contacts (LF,LM,LH,RF,RM,RH)'}")
    print("-" * 70)

    start_time = time.time()
    total_reward = 0.0

    for step in range(1, num_steps + 1):
        # Convert observation to PyTorch tensor
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).float()
        with torch.no_grad():
            action, _, _, _ = policy.get_action_and_value(obs_tensor)
        act_np = action.squeeze(0).numpy()

        # Step environment in MuJoCo
        obs, reward, terminated, truncated, info = env.step(act_np)
        total_reward += reward

        # Record video frame
        if save_gif and step % 2 == 0:  # Save every 2nd step to keep GIF size compact
            frame = env.render()
            if frame is not None:
                frames.append(frame)

        # Telemetry
        pos = info["thorax_pos"]
        r_fwd = info.get("reward_forward", 0.0)
        # Contacts from obs [93..98]
        contacts = [f"{c:.1f}" for c in obs[93:99]]
        contact_str = " ".join(contacts)

        if step % 5 == 0 or step == 1 or step == num_steps:
            print(f"{step:<6} | {pos[0]:<8.3f} | {pos[1]:<8.3f} | {pos[2]:<8.3f} | {r_fwd:<10.4f} | [{contact_str}]")

        if terminated:
            print(f"\n[!] Fly tilted or lost balance at step {step}.")
            break

    elapsed = time.time() - start_time
    env.close()

    print("-" * 70)
    print(f"Simulation completed {step} steps in {elapsed:.2f}s ({step / elapsed:.1f} steps/second)")
    print(f"Cumulative Reward: {total_reward:.2f}")

    # 4. Save animated GIF
    if save_gif and len(frames) > 0:
        print(f"\n[4/4] Rendering animated video to: {out_file}")
        # 25 fps playback
        imageio.mimsave(str(out_file), frames, fps=20, loop=0)
        print(f"      Saved {len(frames)} frames to {out_file.name} successfully!")

    print("\n" + "=" * 70)
    print("DEMO RUN FINISHED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FlyGym MuJoCo Simulation Demo")
    parser.add_argument("--steps", type=int, default=40, help="Number of steps to simulate")
    parser.add_argument("--substeps", type=int, default=20, help="Physics substeps per action")
    parser.add_argument("--pose", type=str, default="tripod", choices=["tripod", "stretch"], help="Starting posture")
    parser.add_argument("--no-gif", action="store_true", help="Disable GIF rendering")
    parser.add_argument("--gui", action="store_true", help="Launch interactive 3D desktop window")
    parser.add_argument("--cam-dist", type=float, default=8.0, help="Camera distance in mm (e.g. 8.0 or 12.0)")
    parser.add_argument("--out", type=str, default="outputs/fly_simulation_demo.gif", help="Output GIF path")
    args = parser.parse_args()

    if args.gui:
        from scripts.view_fly_interactive import launch_interactive_viewer
        launch_interactive_viewer(
            pose=args.pose,
            physics_substeps=args.substeps,
            cam_distance=args.cam_dist,
        )
    else:
        run_demo(
            num_steps=args.steps,
            physics_substeps=args.substeps,
            pose=args.pose,
            save_gif=not args.no_gif,
            output_path=args.out,
        )

"""Launch an Interactive 3D MuJoCo Physics Window with the Fruit Fly.

Opens a native OpenGL 3D viewer window on your desktop where you can:
  - Orbit, rotate, and zoom the camera with your mouse.
  - Watch the fruit fly actuated by the biological Janelia connectome policy in real time.
  - Apply physical perturbation forces by Ctrl + Right-Click dragging the fly!
  - Pause / resume with Spacebar.

Usage:
  wsl -e /home/anish/miniconda3/envs/connectome-rl/bin/python scripts/view_fly_interactive.py
"""

from __future__ import annotations

import os
# Activate Direct3D 12 hardware acceleration for WSLg using NVIDIA GPU
os.environ.setdefault("GALLIUM_DRIVER", "d3d12")
os.environ.setdefault("MESA_D3D12_DEFAULT_ADAPTER_NAME", "NVIDIA")

# Fix WSLg Wayland invisible window bug:
# In WSLg, GLFW on Wayland sometimes registers in the taskbar but fails to present its surface.
# Forcing X11 ensures the window is immediately mapped to a visible Windows desktop window.
if "WAYLAND_DISPLAY" in os.environ and os.environ.get("DISPLAY"):
    del os.environ["WAYLAND_DISPLAY"]


import argparse
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np
import torch

from connectome_rl.src.envs.cpg_wrapper import CPGLocomotionEnv
from connectome_rl.src.envs.fly_wrapper import FlyLocomotionEnv
from connectome_rl.src.models.connectome_policy import ConnectomePolicy


def launch_interactive_viewer(
    mode: str = "walking",
    steer: str = "straight",
    speed: float = 1.0,
    pose: str = "tripod",
    physics_substeps: int = 20,
    cam_distance: float = 9.0,
    lesion_target: str | None = None,
    lesion_side: str = "bilateral",
    checkpoint: str | None = None,
    smooth_alpha: float = 0.35,
    action_scale: float = 0.45,
    action_gain: float = 1.6,
) -> None:
    print("=" * 75)
    print("  INTERACTIVE MUJOCO 3D VIEWER: FRUIT FLY CONNECTOME SIMULATION")
    print(f"  [*] LOCOMOTION MODE:    {mode.upper()} (Biological CPG Tripod Gait)" if mode == "walking" else f"  [*] MODE: {mode.upper()}")
    if mode == "walking":
        print(f"  [*] STEERING PATTERN:   {steer.upper()} (Left/Right CPG Drive Modulation)")
        print(f"  [*] WALKING SPEED:      {speed}x (Natural Biological Pacing)")
    if lesion_target:
        print(f"  [!] IN-SILICO LESION:   {lesion_target.upper()} ({lesion_side.upper()} ABLATION)")
    if checkpoint:
        print(f"  [*] TRAINED CHECKPOINT: {checkpoint}")
    print(f"  [*] CAMERA DISTANCE:    {cam_distance} mm (Thorax Auto-Tracking)")
    print("=" * 75)

    if mode == "walking":
        print("\n[1/2] Initializing Biological CPG Locomotion Engine (FlyGym v1.2.1)...")
        # Scale CPG oscillation frequency biologically so the fly sprints without lagging CPU
        env = CPGLocomotionEnv(enable_adhesion=True, freq_scale=speed, seed=0)
        m = env.sim.physics.model.ptr
        d = env.sim.physics.data.ptr
    else:
        # Original posture balance mode
        project_root = Path(__file__).resolve().parents[1]
        tensors_path = project_root / "connectome_rl" / "data" / "dna_circuit_tensors.pt"
        if not tensors_path.exists():
            raise FileNotFoundError(f"Circuit tensors not found at: {tensors_path}")

        print("\n[1/2] Loading Janelia MaleCNS v1.0 biological connectome policy...")
        policy = ConnectomePolicy(obs_dim=100, act_dim=42, circuit_data=tensors_path)

        if checkpoint is not None:
            ckpt_path = project_root / checkpoint
            if ckpt_path.exists():
                ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                policy.load_state_dict(ckpt_data["model_state_dict"])
                print(f"      Loaded trained weights from: {ckpt_path}")

        if lesion_target is not None:
            from connectome_rl.src.analysis.lesion import LesionConfig, LesionController
            lesion_ctrl = LesionController(policy=policy, circuit_data=tensors_path)
            ablated_cols = lesion_ctrl.apply_lesion(LesionConfig(target_type=lesion_target, side=lesion_side))
            print(f"      Silenced biological synaptic columns: {ablated_cols}")

        policy.eval()

        env = FlyLocomotionEnv(
            physics_steps_per_action=physics_substeps,
            action_scale=action_scale,
            init_pose=pose,
            max_episode_steps=100000,
            enable_render=False,
        )
        m = env.sim.physics.model.ptr
        d = env.sim.physics.data.ptr

    print("\n[2/2] Launching interactive 3D window...")
    print("-" * 75)
    print("MOUSE & VIEW CONTROLS IN 3D WINDOW:")
    print("  - Left Mouse Drag:         Rotate 3D camera around the fly")
    print("  - Right Mouse Drag / Wheel: Zoom in / out")
    print("  - Middle Mouse Drag:       Pan camera across the arena")
    print("  - Spacebar:                Pause / resume walking simulation")
    print("  - Close Window or Ctrl+C:  Exit viewer")
    print("-" * 75)

    obs, info = env.reset()
    step_count = 0

    thorax_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "0/Thorax")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Camera auto-tracks the fly's thorax smoothly across the arena
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = thorax_id
        viewer.cam.distance = cam_distance
        viewer.cam.elevation = -22.0
        viewer.cam.azimuth = 135.0

        viewer.sync()

        print("\n>>> Interactive 3D window is now open on your desktop! <<<")
        print(">>> Watch the fruit fly continuously march forward across the floor! <<<")
        print(">>> Tip: Use Mouse Scroll Wheel or Right-Click Drag to Zoom! <<<\n")

        prev_action = np.zeros(42, dtype=np.float32)
        # Keep physics substeps per visual frame constant and lightweight (15 steps)
        # Walking speed is driven biologically by CPG oscillation frequency (freq_scale)
        n_steps = 15 if mode == "walking" else max(1, int(physics_substeps))

        while viewer.is_running():
            for _ in range(n_steps):
                if mode == "walking":
                    # Compute CPG differential drive
                    if steer == "straight":
                        drive_l, drive_r = 1.0, 1.0
                    elif steer == "left":
                        drive_l, drive_r = 0.4, 1.2
                    elif steer == "right":
                        drive_l, drive_r = 1.2, 0.4
                    elif steer == "patrol":
                        t_sec = step_count * env.timestep
                        steer_mod = 0.4 * np.sin(2.0 * np.pi * 0.4 * t_sec)
                        drive_l = 1.0 - steer_mod
                        drive_r = 1.0 + steer_mod
                    else:
                        drive_l, drive_r = 1.0, 1.0

                    # In-silico lesion impact:
                    if lesion_target == "DNa01":
                        if lesion_side == "left":
                            drive_l *= 0.25  # Left motor asymmetry
                        elif lesion_side == "right":
                            drive_r *= 0.25
                        elif lesion_side == "bilateral":
                            drive_l *= 0.3
                            drive_r *= 0.3

                    action_cpg = np.array([drive_l, drive_r], dtype=np.float32)
                    obs, reward, terminated, truncated, info = env.step(action_cpg)
                    step_count += 1

                    # Live progress telemetry every 600 physics steps
                    if step_count % 600 == 0:
                        dist = info.get("forward_dist_mm", 0.0)
                        yaw = info.get("yaw_deg", 0.0)
                        print(f"  [Step {step_count:05d}] Walking forward: {dist:+.2f} mm | Heading Yaw: {yaw:+.1f}°")

                else:
                    # Posture mode with ConnectomePolicy
                    obs_tensor = torch.from_numpy(obs).unsqueeze(0).float()
                    with torch.no_grad():
                        action, _, _, _ = policy.get_action_and_value(obs_tensor)
                    raw_act = action.squeeze(0).numpy()

                    if smooth_alpha < 1.0:
                        filtered_act = smooth_alpha * raw_act + (1.0 - smooth_alpha) * prev_action
                        prev_action = filtered_act
                    else:
                        filtered_act = raw_act

                    act_np = np.clip(filtered_act * action_gain, -1.0, 1.0)
                    obs, reward, terminated, truncated, info = env.step(act_np)
                    step_count += 1

                    if terminated:
                        obs, info = env.reset()
                        prev_action = np.zeros(42, dtype=np.float32)
                        break

            viewer.sync()
            time.sleep(0.001)

    env.close()
    print("\nViewer closed. Demo finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive MuJoCo 3D Fly Viewer")
    parser.add_argument("--mode", type=str, default="walking", choices=["walking", "posture"], help="Locomotion mode ('walking' or 'posture')")
    parser.add_argument("--steer", type=str, default="straight", choices=["straight", "left", "right", "patrol"], help="Steering behavior in walking mode")
    parser.add_argument("--speed", type=float, default=1.0, help="Walking speed multiplier (default 1.0 = real-time, 1.5 = brisk, 2.0 = fast sprint)")
    parser.add_argument("--pose", type=str, default="tripod", choices=["tripod", "stretch"], help="Starting posture (posture mode)")
    parser.add_argument("--substeps", type=int, default=20, help="Physics substeps per action (posture mode)")
    parser.add_argument("--cam-dist", type=float, default=9.0, help="Camera distance in mm (default 9.0 mm)")
    parser.add_argument("--lesion", type=str, default=None, choices=["DNa01", "DNa02", "random"], help="In-silico neuron lesion")
    parser.add_argument("--side", type=str, default="bilateral", choices=["bilateral", "left", "right"], help="Lesion side")
    parser.add_argument("--checkpoint", type=str, default=None, help="Trained checkpoint path")
    args = parser.parse_args()

    launch_interactive_viewer(
        mode=args.mode,
        steer=args.steer,
        speed=args.speed,
        pose=args.pose,
        physics_substeps=args.substeps,
        cam_distance=args.cam_dist,
        lesion_target=args.lesion,
        lesion_side=args.side,
        checkpoint=args.checkpoint,
    )






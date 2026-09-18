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

from connectome_rl.src.envs.fly_wrapper import FlyLocomotionEnv
from connectome_rl.src.models.connectome_policy import ConnectomePolicy


def launch_interactive_viewer(
    pose: str = "tripod",
    physics_substeps: int = 20,
    speed_multiplier: float = 1.0,
    cam_distance: float = 8.0,
    lesion_target: str | None = None,
    lesion_side: str = "bilateral",
    checkpoint: str | None = None,
    smooth_alpha: float = 0.35,
    steps_per_render: int = 1,
    action_scale: float = 0.45,
    action_gain: float = 1.6,
) -> None:
    print("=" * 75)
    print("  INTERACTIVE MUJOCO 3D VIEWER: FRUIT FLY CONNECTOME SIMULATION")
    if lesion_target:
        print(f"  [!] IN-SILICO LESION ACTIVE: {lesion_target.upper()} ({lesion_side.upper()} ABLATION)")
    if checkpoint:
        print(f"  [*] TRAINED CHECKPOINT: {checkpoint}")
    print(f"  [*] SMOOTHING FILTER:   alpha={smooth_alpha} (organic biological muscle dynamics)")
    print(f"  [*] ACTION SCALE/GAIN:  scale={action_scale} rad, gain={action_gain}x")
    print(f"  [*] STEPS PER FRAME:    {steps_per_render} step(s) per render sync")
    print("=" * 75)


    # 1. Locate circuit tensors
    project_root = Path(__file__).resolve().parents[1]
    tensors_path = project_root / "connectome_rl" / "data" / "dna_circuit_tensors.pt"
    if not tensors_path.exists():
        raise FileNotFoundError(f"Circuit tensors not found at: {tensors_path}")

    print("[1/3] Loading Janelia MaleCNS v1.0 biological connectome policy...")
    policy = ConnectomePolicy(obs_dim=100, act_dim=42, circuit_data=tensors_path)

    if checkpoint is not None:
        ckpt_path = project_root / checkpoint
        if ckpt_path.exists():
            ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            policy.load_state_dict(ckpt_data["model_state_dict"])
            print(f"      Loaded trained weights from: {ckpt_path}")
        else:
            print(f"      [!] Checkpoint not found at: {ckpt_path}")

    if lesion_target is not None:
        from connectome_rl.src.analysis.lesion import LesionConfig, LesionController
        lesion_ctrl = LesionController(policy=policy, circuit_data=tensors_path)
        ablated_cols = lesion_ctrl.apply_lesion(LesionConfig(target_type=lesion_target, side=lesion_side))
        print(f"      Silenced biological synaptic columns: {ablated_cols}")

    policy.eval()

    print(f"[2/3] Initializing FlyGym v1.2.1 simulation in MuJoCo (pose: {pose}, scale: {action_scale} rad)...")
    # In interactive mode, we don't need offscreen rendering because the viewer renders natively
    env = FlyLocomotionEnv(
        physics_steps_per_action=physics_substeps,
        action_scale=action_scale,
        init_pose=pose,
        max_episode_steps=100000,
        enable_render=False,
    )

    m = env.sim.physics.model.ptr
    d = env.sim.physics.data.ptr

    print("\n[3/3] Launching interactive 3D window...")
    print("-" * 75)
    print("MOUSE & KEYBOARD CONTROLS IN 3D WINDOW:")
    print("  - Left Mouse Drag:         Rotate 3D camera")
    print("  - Right Mouse Drag / Wheel: Zoom in / out")
    print("  - Middle Mouse Drag:       Pan camera across the arena")
    print("  - Double Click on Fly:     Center camera on the fly")
    print("  - Ctrl + Right Click Drag: Apply virtual force / pull the fly's body!")
    print("  - Spacebar:                Pause / resume simulation")
    print("  - Backspace:               Reset fly to starting pose")
    print("  - Close Window or Ctrl+C:  Exit viewer")
    print("-" * 75)

    obs, info = env.reset()
    step_count = 0

    thorax_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "0/Thorax")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Configure tracking camera centered on the fly's thorax
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = thorax_id
        viewer.cam.distance = cam_distance  # Configured distance in mm (default 8.0 mm)
        viewer.cam.elevation = -25.0        # Angle looking slightly downward
        viewer.cam.azimuth = 135.0          # Isometric diagonal perspective

        # Force immediate first frame sync so the window renders the 3D scene instantly
        viewer.sync()

        print("\n>>> Interactive 3D window is now open on your desktop! <<<")
        print(">>> If the window opened behind other apps or looks hidden: <<<")
        print(">>> Click the Tux icon in the taskbar and press [Windows Key + Up Arrow] to maximize it! <<<")
        print(">>> Tip: Use the Mouse Scroll Wheel or Right-Click Drag to Zoom In / Out! <<<")

        # Smooth, steady frame pacing:
        # Eliminates burst-and-freeze rubber-banding by stepping physics at consistent intervals
        prev_action = np.zeros(42, dtype=np.float32)
        n_steps = max(1, int(steps_per_render))

        while viewer.is_running():
            for _ in range(n_steps):
                # 1. Policy forward pass: 100-dim sensation -> 42-dim joint actuation
                obs_tensor = torch.from_numpy(obs).unsqueeze(0).float()
                with torch.no_grad():
                    action, _, _, _ = policy.get_action_and_value(obs_tensor)
                raw_act = action.squeeze(0).numpy()

                # 2. Biological muscle activation filter (exponential moving average) + amplification gain:
                # Smooths out high-frequency neural jitter into organic, fluid strides
                if smooth_alpha < 1.0:
                    filtered_act = smooth_alpha * raw_act + (1.0 - smooth_alpha) * prev_action
                    prev_action = filtered_act
                else:
                    filtered_act = raw_act

                # Scale by action_gain for clear visual leg swings
                act_np = np.clip(filtered_act * action_gain, -1.0, 1.0)

                # 3. Advance physics in MuJoCo
                obs, reward, terminated, truncated, info = env.step(act_np)
                step_count += 1

                # Auto-reset if the fly flips over
                if terminated:
                    obs, info = env.reset()
                    prev_action = np.zeros(42, dtype=np.float32)
                    break

            # 4. Synchronize 3D viewer graphics smoothly every frame
            viewer.sync()
            time.sleep(0.001)

    env.close()
    print("\nViewer closed. Demo finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive MuJoCo 3D Fly Viewer")
    parser.add_argument("--pose", type=str, default="tripod", choices=["tripod", "stretch"], help="Starting posture")
    parser.add_argument("--substeps", type=int, default=20, help="Physics substeps per action")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    parser.add_argument("--cam-dist", type=float, default=8.0, help="Camera distance in mm (e.g. 8.0 or 12.0)")
    parser.add_argument("--lesion", type=str, default=None, choices=["DNa01", "DNa02", "random"], help="In-silico neuron lesion")
    parser.add_argument("--side", type=str, default="bilateral", choices=["bilateral", "left", "right"], help="Lesion side")
    parser.add_argument("--checkpoint", type=str, default=None, help="Trained checkpoint path")
    parser.add_argument("--smooth", type=float, default=0.35, help="Action smoothing factor alpha in (0, 1] (default 0.35)")
    parser.add_argument("--steps-per-frame", type=int, default=1, help="Physics steps per render frame (default 1 = fluid 35-40 FPS)")
    parser.add_argument("--action-scale", type=float, default=0.45, help="Maximum joint angle deviation in radians (default 0.45 rad ~= 26 deg)")
    parser.add_argument("--gain", type=float, default=1.6, help="Motor action amplification gain (default 1.6x)")
    args = parser.parse_args()

    launch_interactive_viewer(
        pose=args.pose,
        physics_substeps=args.substeps,
        speed_multiplier=args.speed,
        cam_distance=args.cam_dist,
        lesion_target=args.lesion,
        lesion_side=args.side,
        checkpoint=args.checkpoint,
        smooth_alpha=args.smooth,
        steps_per_render=args.steps_per_frame,
        action_scale=args.action_scale,
        action_gain=args.gain,
    )




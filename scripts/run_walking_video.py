"""Generate a publication-grade walking demonstration GIF.

Simulates the fruit fly executing continuous biological CPG tripod locomotion,
captures offscreen frames with the tracking camera, and exports an animated GIF:
  - outputs/fly_cpg_walking.gif
"""

from __future__ import annotations

import logging
from pathlib import Path
import numpy as np
from PIL import Image

from flygym import YawOnlyCamera, SingleFlySimulation
from flygym.arena import FlatTerrain
from flygym.examples.locomotion.turning_fly import HybridTurningFly

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CPGWalkingVideo")


def generate_walking_demo_gif(
    out_path: str | Path = "outputs/fly_cpg_walking.gif",
    num_steps: int = 400,
    fps: int = 30,
) -> Path:
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    timestep = 1e-4
    contact_sensors = [
        f"{leg}{segment}"
        for leg in ["LF", "LM", "LH", "RF", "RM", "RH"]
        for segment in ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]
    ]

    logger.info("Initializing HybridTurningFly with CPG tripod stepping...")
    fly = HybridTurningFly(
        enable_adhesion=True,
        draw_adhesion=False,
        contact_sensor_placements=contact_sensors,
        seed=0,
        timestep=timestep,
    )

    cam = YawOnlyCamera(
        attachment_point=fly.model.worldbody,
        camera_name="camera_right",
        targeted_fly_names=fly.name,
        play_speed=0.2,
    )

    sim = SingleFlySimulation(
        fly=fly,
        cameras=[cam],
        timestep=timestep,
        arena=FlatTerrain(),
    )

    obs, info = sim.reset(seed=0)
    start_pos = obs["fly"][0]
    logger.info("Initial position: X=%.2f mm, Y=%.2f mm, Z=%.2f mm", start_pos[0], start_pos[1], start_pos[2])

    frames: list[np.ndarray] = []

    logger.info("Stepping simulation for %d steps with forward CPG drive...", num_steps)
    # Step simulation with forward march drive [1.0, 1.0]
    for step in range(num_steps):
        # Steer slightly to demonstrate agile fluid movement
        t_sec = step * timestep
        steer_mod = 0.2 * np.sin(2.0 * np.pi * 1.5 * t_sec)
        action = np.array([1.0 - steer_mod, 1.0 + steer_mod], dtype=np.float64)

        obs, reward, terminated, truncated, info = sim.step(action)

        # Render offscreen camera frame every 8 physics steps (~80 FPS physics -> 30 FPS video)
        if step % 8 == 0:
            frame = sim.render()[0]
            if frame is not None:
                frames.append(frame)

    end_pos = obs["fly"][0]
    distance_traveled = float(np.linalg.norm(end_pos[:2] - start_pos[:2]))
    logger.info("Final position:   X=%.2f mm, Y=%.2f mm", end_pos[0], end_pos[1])
    logger.info("Total distance:   %.2f mm forward across the arena", distance_traveled)

    sim.close()

    if frames:
        logger.info("Assembling %d frames into animated GIF: %s", len(frames), out_file)
        pil_images = [Image.fromarray(f) for f in frames]
        pil_images[0].save(
            out_file,
            save_all=True,
            append_images=pil_images[1:],
            duration=int(1000 / fps),
            loop=0,
        )
        logger.info("Saved walking demonstration GIF successfully (%d KB)", out_file.stat().st_size // 1024)

    return out_file


if __name__ == "__main__":
    generate_walking_demo_gif()

"""Biological Central Pattern Generator (CPG) Locomotion Environment.

Couples FlyGym's NeuroMechFly HybridTurningFly (6-leg contact sensing, dynamic
adhesion, stumbling and retraction reflexes) with descending motor commands.

Generates continuous forward tripod walking and closed-loop steering:
  - Straight forward march: drive = [1.0, 1.0]
  - Left steer / turn:      drive = [0.4, 1.2]
  - Right steer / turn:     drive = [1.2, 0.4]
  - Stop / hold stance:     drive = [0.0, 0.0]
"""

from __future__ import annotations

from typing import Any
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from flygym import SingleFlySimulation
from flygym.arena import FlatTerrain
from flygym.examples.locomotion.turning_fly import HybridTurningFly


class CPGLocomotionEnv(gym.Env):
    """Continuous locomotion environment driven by an oscillatory CPG network.

    Action Space:
        Box(low=-0.5, high=1.5, shape=(2,), dtype=float32)
        Descending signal controlling [left_drive, right_drive].
        Also accepts 4-dim biological descending commands [DNa01_L, DNa01_R, DNa02_L, DNa02_R],
        which are automatically mapped to differential steering drive.

    Observation Space:
        Box(-inf, inf, shape=(12,), dtype=float32):
          0..2:   Body position [x, y, z] in mm
          3..5:   Body linear velocity [vx, vy, vz] in mm/s
          6:      Yaw heading angle in degrees
          7:      Yaw angular velocity in deg/s
          8..9:   CPG drive state [left_drive, right_drive]
          10..11: Stance duty factor estimates
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        timestep: float = 1e-4,
        spawn_pos: tuple[float, float, float] = (0.0, 0.0, 0.25),
        enable_adhesion: bool = True,
        draw_corrections: bool = False,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.timestep = timestep
        self.spawn_pos = spawn_pos
        self.enable_adhesion = enable_adhesion
        self.draw_corrections = draw_corrections
        self._seed = seed

        # Contact sensors needed for stumbling and retraction reflexes
        self.contact_sensor_placements = [
            f"{leg}{segment}"
            for leg in ["LF", "LM", "LH", "RF", "RM", "RH"]
            for segment in ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]
        ]

        # 1. Build HybridTurningFly with CPG network and reflex arcs
        self.fly = HybridTurningFly(
            enable_adhesion=self.enable_adhesion,
            draw_adhesion=False,
            contact_sensor_placements=self.contact_sensor_placements,
            spawn_pos=self.spawn_pos,
            seed=self._seed,
            draw_corrections=self.draw_corrections,
            timestep=self.timestep,
        )

        # 2. Build MuJoCo physics simulation
        self.sim = SingleFlySimulation(
            fly=self.fly,
            timestep=self.timestep,
            arena=FlatTerrain(),
        )

        # 3. Define Action and Observation Spaces
        self.action_space = spaces.Box(
            low=-0.5,
            high=1.5,
            shape=(2,),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(12,),
            dtype=np.float32,
        )

        self.initial_pos = np.array(spawn_pos, dtype=np.float64)
        self.current_pos = np.array(spawn_pos, dtype=np.float64)
        self.current_yaw_deg = 0.0
        self.step_count = 0

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset fly to starting pose and reinitialize CPG phases."""
        super().reset(seed=seed)
        r_seed = seed if seed is not None else self._seed

        obs_dict, info = self.sim.reset(seed=r_seed)
        fly_pos = obs_dict["fly"][0]
        self.initial_pos = np.copy(fly_pos)
        self.current_pos = np.copy(fly_pos)
        self.current_yaw_deg = 0.0
        self.step_count = 0

        obs_vec = self._build_obs_vector(obs_dict, drive_left=1.0, drive_right=1.0)
        info["position_mm"] = self.current_pos
        info["forward_dist_mm"] = 0.0
        info["yaw_deg"] = 0.0

        return obs_vec, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one simulation step using CPG descending drive.

        Args:
            action: Either 2-dim [drive_left, drive_right] or
                    4-dim biological [DNa01_L, DNa01_R, DNa02_L, DNa02_R].

        Returns:
            (obs, reward, terminated, truncated, info)
        """
        act = np.asarray(action, dtype=np.float32)

        # Map 4D biological connectome descending commands to 2D steering drive:
        if act.shape == (4,):
            # DNa01 controls steering asymmetry (left turning slows left leg)
            # DNa02 controls forward velocity drive
            dna01_l, dna01_r, dna02_l, dna02_r = act
            base_drive = 1.0 + 0.3 * (dna02_l + dna02_r)
            drive_left = base_drive - 0.4 * dna01_l + 0.2 * dna01_r
            drive_right = base_drive - 0.4 * dna01_r + 0.2 * dna01_l
            cpg_action = np.clip([drive_left, drive_right], -0.5, 1.5).astype(np.float64)
        elif act.shape == (2,):
            cpg_action = np.clip(act, -0.5, 1.5).astype(np.float64)
        else:
            raise ValueError(f"Expected action shape (2,) or (4,), got {act.shape}")

        # Step FlyGym simulation
        obs_dict, reward, terminated, truncated, info = self.sim.step(cpg_action)
        self.step_count += 1

        fly_pos = obs_dict["fly"][0]
        self.current_pos = np.copy(fly_pos)

        # Compute heading orientation
        fly_rot = obs_dict["fly"][1]  # rotation matrix or euler
        if fly_rot.shape == (3, 3):
            # Yaw from rotation matrix R[1, 0] / R[0, 0]
            self.current_yaw_deg = float(np.degrees(np.arctan2(fly_rot[1, 0], fly_rot[0, 0])))
        elif len(fly_rot) == 3:
            self.current_yaw_deg = float(np.degrees(fly_rot[2]))

        forward_dist = float(self.current_pos[0] - self.initial_pos[0])

        # Dense locomotion reward
        step_reward = forward_dist * 10.0 - 0.1 * abs(self.current_yaw_deg)

        obs_vec = self._build_obs_vector(obs_dict, cpg_action[0], cpg_action[1])

        info["position_mm"] = self.current_pos
        info["forward_dist_mm"] = forward_dist
        info["yaw_deg"] = self.current_yaw_deg

        return obs_vec, step_reward, terminated, truncated, info

    def _build_obs_vector(
        self,
        obs_dict: dict[str, Any],
        drive_left: float,
        drive_right: float,
    ) -> np.ndarray:
        """Flatten sensory state into a compact 12-dimensional vector."""
        pos = obs_dict["fly"][0]
        vel = obs_dict["fly"][2] if len(obs_dict["fly"]) > 2 else np.zeros(3)

        obs = np.array([
            pos[0], pos[1], pos[2],
            vel[0], vel[1], vel[2],
            self.current_yaw_deg,
            0.0,  # yaw velocity
            drive_left, drive_right,
            0.5, 0.5,  # stance duty factors
        ], dtype=np.float32)

        return obs

    def close(self) -> None:
        """Safely shut down physics simulation."""
        self.sim.close()

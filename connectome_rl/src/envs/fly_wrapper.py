"""Gymnasium Environment Wrapper for FlyGym v1.2.1 Biomechanical Simulation.

Bridges the MuJoCo 3.x physics simulation of Drosophila melanogaster with
PyTorch Reinforcement Learning and the MaleCNS biological connectome.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from flygym import Fly, SingleFlySimulation, Camera
from flygym.arena import FlatTerrain
from connectome_rl.src.envs.rewards import LocomotionRewardCalculator, LocomotionRewardWeights


class FlyLocomotionEnv(gym.Env):
    """Gymnasium environment wrapping FlyGym SingleFlySimulation for locomotion learning.
    
    Action Space:
        Box(-1.0, 1.0, shape=(42,), dtype=float32)
        Continuous normalized target angles for the 42 actuated leg joints (6 legs × 7 DOFs).
        Actions are centered around the biological standing posture (KinematicPose).
        
    Observation Space:
        Box(-inf, inf, shape=(100,), dtype=float32)
        Continuous proprioceptive and vestibular features:
          - 0..41:   Joint angle errors relative to neutral posture (42 dims)
          - 42..83:  Joint angular velocities (42 dims)
          - 84..86:  Thorax orientation Euler angles [roll, pitch, yaw] (3 dims)
          - 87..89:  Thorax linear velocity [vx, vy, vz] (3 dims)
          - 90..92:  Thorax angular velocity [wx, wy, wz] (3 dims)
          - 93..98:  Per-leg ground contact force magnitudes [LF, LM, LH, RF, RM, RH] (6 dims)
          - 99:      Thorax height above ground [z] in mm (1 dim)
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        physics_steps_per_action: int = 20,
        action_scale: float = 0.25,
        max_episode_steps: int = 500,
        spawn_pos: tuple[float, float, float] = (0.0, 0.0, 0.5),
        init_pose: str = "tripod",
        enable_render: bool = False,
        render_camera_name: str = "camera_top",
        reward_weights: LocomotionRewardWeights | None = None,
    ) -> None:
        """Initialize the FlyGym locomotion environment.
        
        Args:
            physics_steps_per_action: Number of MuJoCo physics steps per policy decision.
                                      (20 steps × 0.0001s = 0.002s = 500 Hz control).
            action_scale: Maximum joint angle deviation from neutral pose in radians (~14.3 deg).
            max_episode_steps: Maximum decision steps before episode truncation.
            spawn_pos: Initial (x, y, z) spawn coordinate of the fly thorax in mm.
            init_pose: Starting kinematic posture ('tripod' for 6-leg stance, 'stretch' for forward reach).
            enable_render: Whether to attach an offscreen camera for visual rendering.
            render_camera_name: Camera perspective ('camera_top', 'camera_side', etc.).
            reward_weights: Optional custom reward weighting hyperparameters.
        """
        super().__init__()
        self.physics_steps_per_action = physics_steps_per_action
        self.action_scale = action_scale
        self.max_episode_steps = max_episode_steps
        self.spawn_pos = spawn_pos
        self.init_pose = init_pose
        self.enable_render = enable_render
        self.render_camera_name = render_camera_name

        self.reward_calculator = LocomotionRewardCalculator(weights=reward_weights)

        # 1. Define Gymnasium Action and Observation Spaces
        self.num_joints = 42
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_joints,),
            dtype=np.float32,
        )

        self.obs_dim = 100
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        # 2. Build FlyGym Simulation & Fly
        self.fly = Fly(spawn_pos=self.spawn_pos, init_pose=self.init_pose)
        self.arena = FlatTerrain()

        # Extract biological default standing joint angles
        self.default_joint_angles = np.array(
            [self.fly.init_pose.joint_pos[joint_name] for joint_name in self.fly.actuated_joints],
            dtype=np.float32,
        )

        # Optional camera attachment with smooth Center-of-Mass tracking (no tilt shake)
        self.camera = None
        cameras = None
        if self.enable_render:
            camera_params = {
                "mode": "trackcom",
                "pos": [0.0, 0.0, 8.0],
                "euler": [0.0, 0.0, 0.0],
            }
            self.camera = Camera(
                attachment_point=self.fly.model.worldbody,
                camera_name=self.render_camera_name,
                camera_parameters=camera_params,
            )
            cameras = [self.camera]

        self.sim = SingleFlySimulation(
            fly=self.fly,
            arena=self.arena,
            cameras=cameras,
        )

        # Step counter and state tracking
        self.current_step = 0
        self.prev_thorax_pos = np.zeros(3, dtype=np.float32)

    def _extract_observation(self, raw_obs: dict[str, np.ndarray]) -> np.ndarray:
        """Parse raw FlyGym simulation dictionary into a 100-dim continuous feature vector."""
        # 1. Joint angles and angular velocities
        joint_angles = raw_obs["joints"][0].astype(np.float32)
        joint_velocities = raw_obs["joints"][1].astype(np.float32)
        joint_angle_errors = joint_angles - self.default_joint_angles  # 42 dims

        # 2. Thorax linear position and velocities
        thorax_pos = raw_obs["fly"][0].astype(np.float32)       # (3,) [x, y, z]
        thorax_lin_vel = raw_obs["fly"][1].astype(np.float32)   # (3,) [vx, vy, vz]
        thorax_ang_vel = raw_obs["fly"][3].astype(np.float32)   # (3,) [wx, wy, wz]
        thorax_orientation = raw_obs["fly_orientation"].astype(np.float32)  # (3,) [roll, pitch, yaw]

        # 3. Ground contact forces: sum 5 tarsus segments for each of the 6 legs
        contact_forces = raw_obs["contact_forces"]  # Shape: (30, 3)
        leg_contacts = np.zeros(6, dtype=np.float32)
        for leg_idx in range(6):
            # 5 tarsus segments per leg: [leg_idx*5 : (leg_idx+1)*5]
            segment_forces = contact_forces[leg_idx * 5 : (leg_idx + 1) * 5]
            # Normal force magnitude sum (in mN)
            leg_contacts[leg_idx] = float(np.linalg.norm(segment_forces, axis=1).sum())

        # 4. Assemble 100-dim vector
        obs_vec = np.concatenate([
            joint_angle_errors,          # 42 dims [0..41]
            joint_velocities,            # 42 dims [42..83]
            thorax_orientation,          # 3 dims  [84..86]
            thorax_lin_vel,              # 3 dims  [87..89]
            thorax_ang_vel,              # 3 dims  [90..92]
            leg_contacts,                # 6 dims  [93..98]
            np.array([thorax_pos[2]], dtype=np.float32),  # 1 dim [99] (height z)
        ]).astype(np.float32)

        return obs_vec

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the simulation to initial state."""
        super().reset(seed=seed)
        self.current_step = 0

        raw_obs, info = self.sim.reset()
        self.prev_thorax_pos = raw_obs["fly"][0].copy()

        obs = self._extract_observation(raw_obs)
        info["step"] = self.current_step
        info["thorax_pos"] = self.prev_thorax_pos
        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance the physics simulation by one policy decision step.
        
        Args:
            action: Normalized continuous joint actions in [-1.0, 1.0] of shape (42,).
            
        Returns:
            observation: 100-dimensional sensory observation vector.
            reward: Scalar locomotion reward.
            terminated: True if the fly flipped or collapsed.
            truncated: True if maximum episode duration reached.
            info: Diagnostics dictionary.
        """
        self.current_step += 1
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        # Convert normalized actions [-1, 1] to target joint angles in radians
        target_joint_angles = self.default_joint_angles + (action * self.action_scale)
        flygym_action = {"joints": target_joint_angles}

        # Step MuJoCo physics for substeps
        raw_obs = None
        for _ in range(self.physics_steps_per_action):
            raw_obs, _, _, _, _ = self.sim.step(flygym_action)

        assert raw_obs is not None
        current_thorax_pos = raw_obs["fly"][0].copy()
        current_cardinal_vectors = raw_obs["cardinal_vectors"]

        # Check termination (falling or flipping over onto back)
        terminated = self.reward_calculator.check_termination(
            curr_pos=current_thorax_pos,
            curr_cardinal_vectors=current_cardinal_vectors,
        )

        # Compute reward
        reward, reward_breakdown = self.reward_calculator.compute_reward(
            prev_pos=self.prev_thorax_pos,
            curr_pos=current_thorax_pos,
            curr_cardinal_vectors=current_cardinal_vectors,
            action=action,
            is_terminated=terminated,
        )

        # Check truncation (time limit)
        truncated = self.current_step >= self.max_episode_steps

        # Extract observation
        obs = self._extract_observation(raw_obs)

        # Update previous position
        self.prev_thorax_pos = current_thorax_pos

        # Diagnostics info
        info = {
            "step": self.current_step,
            "thorax_pos": current_thorax_pos,
            **reward_breakdown,
        }

        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        """Render RGB image of current simulation state."""
        if not self.enable_render or self.camera is None:
            return None
        frames = self.sim.render()
        if frames and len(frames) > 0:
            return frames[0]
        return None

    def close(self) -> None:
        """Clean up simulation resources."""
        if hasattr(self.sim, "close"):
            self.sim.close()

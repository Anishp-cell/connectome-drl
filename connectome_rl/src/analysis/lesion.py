"""In-Silico Targeted Lesion and Optogenetic Ablation Suite.

Enables precision neuroscience experiments on trained embodied connectome policies:
  1. Bilateral Ablation: Silencing both left and right homologous neurons (e.g. all DNa01).
  2. Unilateral Ablation: Silencing only one hemisphere (e.g. left DNa01 vs right DNa01)
     to measure asymmetric steering bias and rotational torque.
  3. Acute Mid-Episode Clamping: Silencing neurons mid-stride to measure kinematic
     perturbation, stumbling, and compensatory recovery.
  4. Random Ablation Controls: Silencing random sets of neurons to prove functional
     specificity versus generic network degradation.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any, Generator, Iterator

import numpy as np
import torch
import torch.nn as nn

from connectome_rl.src.connectome.graph_utils import ConnectomeCircuitData, load_circuit_data
from connectome_rl.src.envs.fly_wrapper import FlyLocomotionEnv

logger = logging.getLogger(__name__)

# Biological mapping in Janelia MaleCNS v1.0
DN_METADATA: dict[str, dict[str, int]] = {
    "DNa01": {"L": 10442, "R": 10760},
    "DNa02": {"L": 523769, "R": 10360},
}


@dataclass
class LesionConfig:
    """Configuration for an in-silico ablation experiment."""

    target_type: str = "DNa01"  # "DNa01", "DNa02", "all_dns", "random", "custom"
    side: str = "bilateral"     # "bilateral", "left", "right"
    target_indices: list[int] | None = None  # Explicit neuron tensor indices if custom
    num_random_nodes: int = 2   # Number of nodes if target_type == "random"
    intensity: float = 1.0      # 1.0 = total knockout (0 weight), 0.5 = 50% knockdown
    start_step: int = 0         # Timestep to begin ablation (0 = whole episode)
    end_step: int | None = None # Timestep to end ablation (None = until episode end)


@dataclass
class LesionResult:
    """Quantitative behavioral metrics recorded under an in-silico lesion."""

    condition: str
    target_type: str
    side: str
    episode_return: float
    forward_distance: float       # Net forward travel along initial heading (mm)
    lateral_drift: float          # Net sideways deviation (mm)
    net_yaw_rotation: float       # Cumulative body turning angle (degrees)
    mean_speed: float             # Average forward velocity (mm/s)
    stability_score: float        # Mean dorsal upright alignment dot product
    trajectory_x: list[float] = field(default_factory=list)
    trajectory_y: list[float] = field(default_factory=list)
    step_rewards: list[float] = field(default_factory=list)


class LesionController:
    """Precision in-silico ablation controller for neural and connectome policies."""

    def __init__(
        self,
        policy: nn.Module,
        circuit_data: ConnectomeCircuitData | Path | str | None = None,
    ) -> None:
        """Initialize the lesion controller.

        Args:
            policy: Trained PyTorch policy module.
            circuit_data: ConnectomeCircuitData instance or path to cached .pt file.
        """
        self.policy = policy
        self._circuit_data: ConnectomeCircuitData | None = None

        if circuit_data is not None:
            if isinstance(circuit_data, (str, Path)):
                self._circuit_data = load_circuit_data(circuit_data)
            else:
                self._circuit_data = circuit_data
        elif hasattr(policy, "circuit_data") and policy.circuit_data is not None:
            self._circuit_data = policy.circuit_data

        # Build neuron bodyId -> 0-indexed DN tensor position mapping
        self.dn_tensor_map: dict[int, int] = {}
        if self._circuit_data is not None:
            for idx in self._circuit_data.dn_indices:
                body_id = self._circuit_data.idx_to_node.get(idx)
                if body_id is not None:
                    # Map bodyId to column index in hop1_layer (0 to num_dn-1)
                    self.dn_tensor_map[body_id] = idx

        # Backup store for non-destructive weight restoration
        self._saved_weights: dict[str, torch.Tensor] = {}
        self._is_lesioned: bool = False

    def resolve_target_indices(self, config: LesionConfig) -> list[int]:
        """Convert biological neuron type and side to 0-indexed tensor column indices."""
        if config.target_indices is not None:
            return config.target_indices

        target = config.target_type.upper()
        side = config.side.lower()

        if target in ["INTACT", "NONE", "CONTROL"]:
            return []

        if target in ["DNA01", "DNA02"]:
            # Retrieve biological body IDs for the specified type and side
            type_key = "DNa01" if target == "DNA01" else "DNa02"
            body_ids: list[int] = []
            if side in ["bilateral", "both"]:
                body_ids = list(DN_METADATA[type_key].values())
            elif side in ["l", "left"]:
                body_ids = [DN_METADATA[type_key]["L"]]
            elif side in ["r", "right"]:
                body_ids = [DN_METADATA[type_key]["R"]]
            else:
                raise ValueError(f"Unknown ablation side '{config.side}'. Choose from ['bilateral', 'left', 'right'].")

            # Map body IDs to tensor column indices in the policy
            indices = [self.dn_tensor_map[bid] for bid in body_ids if bid in self.dn_tensor_map]
            if not indices:
                # Fallback to default index positions: DNa01=[1,2], DNa02=[0,3]
                if target == "DNA01":
                    indices = [1, 2] if side == "bilateral" else ([1] if side in ["l", "left"] else [2])
                else:
                    indices = [0, 3] if side == "bilateral" else ([3] if side in ["l", "left"] else [0])
            return indices

        elif target in ["ALL_DNS", "ALL_DN"]:
            if self._circuit_data is not None:
                return list(range(len(self._circuit_data.dn_indices)))
            return [0, 1, 2, 3]

        elif target == "RANDOM":
            total_dns = len(self._circuit_data.dn_indices) if self._circuit_data else 4
            num_k = min(config.num_random_nodes, total_dns)
            perm = np.random.permutation(total_dns)
            return perm[:num_k].tolist()

        else:
            raise ValueError(f"Unsupported lesion target type: '{config.target_type}'")

    def apply_lesion(self, config: LesionConfig) -> list[int]:
        """Apply targeted ablation to policy weights non-destructively.

        Args:
            config: LesionConfig specifying neuron targets and ablation intensity.

        Returns:
            List of 0-indexed tensor positions that were ablated.
        """
        if self._is_lesioned:
            self.restore()

        indices = self.resolve_target_indices(config)
        scale = 1.0 - config.intensity  # 0.0 for complete knockout

        # 1. ConnectomePolicy: hop1_layer is MaskedLinear(num_dn -> num_in)
        if hasattr(self.policy, "hop1_layer") and hasattr(self.policy.hop1_layer, "weight"):
            w = self.policy.hop1_layer.weight
            self._saved_weights["hop1"] = w.clone()
            with torch.no_grad():
                for col in indices:
                    if col < w.shape[1]:
                        w[:, col] *= scale

        # 2. HierarchicalPolicy: vnc.hop1_layer is MaskedLinear(num_dn -> num_in)
        elif hasattr(self.policy, "vnc") and hasattr(self.policy.vnc, "hop1_layer"):
            w = self.policy.vnc.hop1_layer.weight
            self._saved_weights["vnc_hop1"] = w.clone()
            with torch.no_grad():
                for col in indices:
                    if col < w.shape[1]:
                        w[:, col] *= scale

        # 3. ConnectomeRNNPolicy: rnn_cell has input weights
        elif hasattr(self.policy, "rnn_cell") and hasattr(self.policy.rnn_cell, "w_in"):
            w = self.policy.rnn_cell.w_in
            self._saved_weights["rnn_w_in"] = w.clone()
            with torch.no_grad():
                for col in indices:
                    if col < w.shape[1]:
                        w[:, col] *= scale

        # 4. SynapticGNNPolicy: input_proj or edge features
        elif hasattr(self.policy, "input_proj") and hasattr(self.policy.input_proj, "weight"):
            w = self.policy.input_proj.weight
            self._saved_weights["gnn_input"] = w.clone()
            with torch.no_grad():
                for col in indices:
                    if col < w.shape[1]:
                        w[:, col] *= scale

        # 5. Generic / MLP baseline policy fallback: ablate first hidden layer features
        elif hasattr(self.policy, "actor") and isinstance(self.policy.actor, nn.Sequential):
            layer0 = self.policy.actor[0]
            if isinstance(layer0, nn.Linear):
                w = layer0.weight
                self._saved_weights["mlp_actor"] = w.clone()
                with torch.no_grad():
                    for col in indices:
                        if col < w.shape[0]:
                            w[col, :] *= scale

        self._is_lesioned = True
        return indices

    def restore(self) -> None:
        """Restore all policy weights to their intact, unlesioned state."""
        if not self._is_lesioned:
            return

        with torch.no_grad():
            if "hop1" in self._saved_weights and hasattr(self.policy, "hop1_layer"):
                self.policy.hop1_layer.weight.copy_(self._saved_weights["hop1"])
            elif "vnc_hop1" in self._saved_weights and hasattr(self.policy, "vnc"):
                self.policy.vnc.hop1_layer.weight.copy_(self._saved_weights["vnc_hop1"])
            elif "rnn_w_in" in self._saved_weights and hasattr(self.policy, "rnn_cell"):
                self.policy.rnn_cell.w_in.copy_(self._saved_weights["rnn_w_in"])
            elif "gnn_input" in self._saved_weights and hasattr(self.policy, "input_proj"):
                self.policy.input_proj.weight.copy_(self._saved_weights["gnn_input"])
            elif "mlp_actor" in self._saved_weights and hasattr(self.policy, "actor"):
                self.policy.actor[0].weight.copy_(self._saved_weights["mlp_actor"])

        self._saved_weights.clear()
        self._is_lesioned = False

    restore_all = restore

    @contextmanager
    def active_lesion(self, config: LesionConfig) -> Generator[list[int], None, None]:
        """Context manager to apply a lesion temporarily and automatically restore on exit."""
        ablated = self.apply_lesion(config)
        try:
            yield ablated
        finally:
            self.restore()

    def evaluate_lesion(
        self,
        env: FlyLocomotionEnv,
        config: LesionConfig,
        num_steps: int = 200,
        seed: int = 42,
    ) -> LesionResult:
        """Run a closed-loop evaluation episode under lesion or mid-episode acute clamping.

        Args:
            env: Biomechanical FlyLocomotionEnv simulation.
            config: Lesion configuration.
            num_steps: Episode evaluation length in control steps.
            seed: Random seed for environment reset.

        Returns:
            LesionResult containing quantitative behavioral metrics and trajectory.
        """
        obs, info = env.reset(seed=seed)
        device = next(self.policy.parameters()).device

        traj_x: list[float] = []
        traj_y: list[float] = []
        step_rewards: list[float] = []

        total_return = 0.0
        init_pos = env.sim.physics.data.qpos[:3].copy()
        init_heading = obs[84:87].copy()

        lesion_active = False

        # If static ablation (start_step == 0 and end_step is None), apply immediately
        if config.start_step == 0 and config.end_step is None:
            self.apply_lesion(config)
            lesion_active = True

        for step in range(num_steps):
            # Dynamic mid-episode lesion activation / deactivation
            if config.start_step > 0 or config.end_step is not None:
                should_be_active = (step >= config.start_step) and (
                    config.end_step is None or step < config.end_step
                )
                if should_be_active and not lesion_active:
                    self.apply_lesion(config)
                    lesion_active = True
                elif not should_be_active and lesion_active:
                    self.restore()
                    lesion_active = False

            # Model inference
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).float().to(device)
            with torch.no_grad():
                out = self.policy.get_action_and_value(obs_tensor)
                act = out[0].squeeze(0).cpu().numpy()

            next_obs, reward, terminated, truncated, step_info = env.step(act)
            obs = next_obs
            total_return += reward
            step_rewards.append(reward)

            # Record trajectory
            cur_pos = env.sim.physics.data.qpos[:3].copy()
            traj_x.append(float(cur_pos[0]))
            traj_y.append(float(cur_pos[1]))

            if terminated or truncated:
                break

        # Ensure policy is restored after evaluation
        if lesion_active:
            self.restore()

        final_pos = env.sim.physics.data.qpos[:3].copy()
        final_heading = obs[84:87].copy()

        # Displacement in arena frame (mm)
        dx = float(final_pos[0] - init_pos[0]) * 1000.0  # Convert m to mm
        dy = float(final_pos[1] - init_pos[1]) * 1000.0

        angle_init = np.arctan2(init_heading[1], init_heading[0])
        angle_final = np.arctan2(final_heading[1], final_heading[0])
        angle_diff = float(np.arctan2(np.sin(angle_final - angle_init), np.cos(angle_final - angle_init)))
        dyaw = float(np.degrees(angle_diff))  # Degrees of yaw turning


        dt = env.physics_steps_per_action * env.sim.timestep
        total_time = max(len(step_rewards) * dt, 1e-6)
        mean_speed = (np.sqrt(dx**2 + dy**2)) / total_time

        condition_name = f"{config.target_type}_{config.side}"

        return LesionResult(
            condition=condition_name,
            target_type=config.target_type,
            side=config.side,
            episode_return=float(total_return),
            forward_distance=float(dx),
            lateral_drift=float(dy),
            net_yaw_rotation=float(dyaw),
            mean_speed=float(mean_speed),
            stability_score=float(np.mean([info.get("upright_score", 1.0)])),
            trajectory_x=traj_x,
            trajectory_y=traj_y,
            step_rewards=step_rewards,
        )

    def run_standard_lesion_battery(
        self,
        env: FlyLocomotionEnv,
        num_steps: int = 150,
        seed: int = 42,
    ) -> dict[str, LesionResult]:
        """Execute the standard biological lesion experiment battery.

        Runs 6 targeted conditions:
          1. Intact Control (unlesioned baseline)
          2. DNa01 Bilateral (steering circuit ablation)
          3. DNa01 Unilateral Left (asymmetric steering knockout -> rightward turn)
          4. DNa01 Unilateral Right (asymmetric steering knockout -> leftward turn)
          5. DNa02 Bilateral (speed circuit ablation)
          6. Random Control (2 random neurons ablated as nonspecific control)

        Returns:
            Dictionary mapping condition name to LesionResult.
        """
        results: dict[str, LesionResult] = {}

        # 1. Intact Control
        intact_res = self.evaluate_lesion(
            env,
            LesionConfig(target_type="DNa01", intensity=0.0),
            num_steps=num_steps,
            seed=seed,
        )
        intact_res.condition = "Intact_Control"
        results["Intact_Control"] = intact_res

        # 2. DNa01 Bilateral
        results["DNa01_Bilateral"] = self.evaluate_lesion(
            env,
            LesionConfig(target_type="DNa01", side="bilateral"),
            num_steps=num_steps,
            seed=seed,
        )

        # 3. DNa01 Left (Unilateral)
        results["DNa01_Left"] = self.evaluate_lesion(
            env,
            LesionConfig(target_type="DNa01", side="left"),
            num_steps=num_steps,
            seed=seed,
        )

        # 4. DNa01 Right (Unilateral)
        results["DNa01_Right"] = self.evaluate_lesion(
            env,
            LesionConfig(target_type="DNa01", side="right"),
            num_steps=num_steps,
            seed=seed,
        )

        # 5. DNa02 Bilateral
        results["DNa02_Bilateral"] = self.evaluate_lesion(
            env,
            LesionConfig(target_type="DNa02", side="bilateral"),
            num_steps=num_steps,
            seed=seed,
        )

        # 6. Random Control
        results["Random_Control"] = self.evaluate_lesion(
            env,
            LesionConfig(target_type="random", num_random_nodes=2),
            num_steps=num_steps,
            seed=seed,
        )

        return results

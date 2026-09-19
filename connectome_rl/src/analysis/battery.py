"""Systematic In-Silico Lesion Battery Suite for Drosophila Connectomics.

Executes rigorous multi-seed experimental ablation batteries:
  1. Intact Control (unlesioned biological baseline)
  2. DNa01 Left (unilateral knockout: steering asymmetry hypothesis)
  3. DNa01 Right (unilateral knockout: contralateral steering asymmetry)
  4. DNa01 Bilateral (bilateral knockout: bilateral turning degradation)
  5. DNa02 Bilateral (bilateral knockout: forward speed control degradation)
  6. Random Knockdown Control (functional specificity control: generic network perturbation)

Computes statistical means, standard deviations, and exports publication-grade error-bar charts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from connectome_rl.src.analysis.benchmark import extract_deterministic_action, compute_energy_and_cot
from connectome_rl.src.analysis.kinematics import KinematicAnalyzer
from connectome_rl.src.analysis.lesion import LesionConfig, LesionController
from connectome_rl.src.connectome.graph_utils import ConnectomeCircuitData, load_circuit_data
from connectome_rl.src.envs.fly_wrapper import FlyLocomotionEnv


@dataclass
class LesionBatteryTrial:
    """Quantitative result of a single in-silico ablation trial under a specific random seed."""

    condition_name: str
    target_type: str
    side: str
    seed: int
    episode_return: float
    forward_distance_mm: float
    mean_speed_mm_s: float
    lateral_drift_mm: float
    net_yaw_deg: float
    energy_expenditure: float
    cost_of_transport: float
    tripod_coordination_index: float
    straightness_index: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ConditionSummaryStats:
    """Aggregated statistical metrics (mean, std, sample size) for a single experimental condition."""

    condition_name: str
    sample_size: int
    return_mean: float
    return_std: float
    forward_dist_mean: float
    forward_dist_std: float
    speed_mean: float
    speed_std: float
    lateral_drift_mean: float
    lateral_drift_std: float
    yaw_deg_mean: float
    yaw_deg_std: float
    energy_mean: float
    energy_std: float
    cot_mean: float
    cot_std: float
    tci_mean: float
    tci_std: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SystematicLesionBattery:
    """Orchestrator for multi-seed systematic in-silico neuroscience lesion batteries."""

    DEFAULT_BATTERY_SPECS: list[dict[str, Any]] = [
        {"name": "Intact Control", "target": "intact", "side": "both", "intensity": 1.0},
        {"name": "DNa01 Left Knockout", "target": "DNa01", "side": "left", "intensity": 1.0},
        {"name": "DNa01 Right Knockout", "target": "DNa01", "side": "right", "intensity": 1.0},
        {"name": "DNa01 Bilateral Knockout", "target": "DNa01", "side": "bilateral", "intensity": 1.0},
        {"name": "DNa02 Bilateral Knockout", "target": "DNa02", "side": "bilateral", "intensity": 1.0},
        {"name": "Random Knockdown Control", "target": "random", "side": "bilateral", "intensity": 0.5},
    ]

    def __init__(
        self,
        policy: nn.Module,
        circuit_data: ConnectomeCircuitData | Path | str | None = None,
        custom_specs: list[dict[str, Any]] | None = None,
    ) -> None:
        self.policy = policy
        self.circuit_data = circuit_data
        self.controller = LesionController(policy=policy, circuit_data=circuit_data)
        self.battery_specs = custom_specs if custom_specs is not None else self.DEFAULT_BATTERY_SPECS

    def run_single_trial(
        self,
        env: FlyLocomotionEnv,
        spec: dict[str, Any],
        seed: int = 42,
        num_steps: int = 50,
        dt: float = 0.002,
    ) -> LesionBatteryTrial:
        """Execute one simulation rollout under an active in-silico lesion configuration."""
        cfg = LesionConfig(
            target_type=spec["target"],
            side=spec.get("side", "both"),
            intensity=spec.get("intensity", 1.0),
        )

        obs, _ = env.reset(seed=seed)
        device = next(self.policy.parameters()).device if list(self.policy.parameters()) else torch.device("cpu")

        positions: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        contacts: list[np.ndarray] = []
        yaws: list[float] = []

        init_heading = obs[84:87].copy()
        total_reward = 0.0

        # Apply targeted in-silico lesion
        self.controller.apply_lesion(cfg)

        try:
            for _ in range(num_steps):
                cur_pos = env.sim.physics.data.qpos[:3].copy() * 1000.0
                positions.append(cur_pos)
                contacts.append(obs[93:99].copy())

                curr_heading = obs[84:87]
                yaw_rad = np.arctan2(
                    init_heading[0] * curr_heading[1] - init_heading[1] * curr_heading[0],
                    init_heading[0] * curr_heading[0] + init_heading[1] * curr_heading[1],
                )
                yaws.append(float(np.degrees(yaw_rad)))

                obs_t = torch.from_numpy(obs).unsqueeze(0).float().to(device)
                act = extract_deterministic_action(self.policy, obs_t)
                act = np.clip(act, -1.0, 1.0)
                actions.append(act)

                next_obs, reward, done, truncated, _ = env.step(act)
                total_reward += reward
                obs = next_obs
                if done or truncated:
                    break
        finally:
            # Ensure policy is restored to 100% intact state
            self.controller.restore()

        analyzer = KinematicAnalyzer(contact_force_threshold=0.5)
        pos_arr = np.array(positions)
        contact_arr = np.array(contacts)
        act_arr = np.array(actions)
        yaw_arr = np.array(yaws)

        gait = analyzer.analyze_rollout(
            contact_forces=contact_arr,
            positions=pos_arr,
            yaws=yaw_arr,
            dt=dt,
        )

        net_fwd = gait.net_forward_displacement_mm
        energy, cot = compute_energy_and_cot(act_arr, net_fwd)
        lateral_drift = float(pos_arr[-1, 1] - pos_arr[0, 1]) if len(pos_arr) > 1 else 0.0

        return LesionBatteryTrial(
            condition_name=spec["name"],
            target_type=spec["target"],
            side=spec.get("side", "both"),
            seed=seed,
            episode_return=float(total_reward),
            forward_distance_mm=float(net_fwd),
            mean_speed_mm_s=float(gait.mean_forward_velocity_mm_s),
            lateral_drift_mm=lateral_drift,
            net_yaw_deg=float(gait.net_yaw_angle_deg),
            energy_expenditure=float(energy),
            cost_of_transport=float(cot),
            tripod_coordination_index=float(gait.tripod_index),
            straightness_index=float(gait.straightness_index),
        )

    def run_battery(
        self,
        env: FlyLocomotionEnv,
        seeds: Sequence[int] = (42, 43, 44),
        num_steps: int = 50,
        dt: float = 0.002,
    ) -> tuple[list[LesionBatteryTrial], list[ConditionSummaryStats]]:
        """Run the full systematic lesion battery across multiple random seeds."""
        all_trials: list[LesionBatteryTrial] = []
        summaries: list[ConditionSummaryStats] = []

        for spec in self.battery_specs:
            condition_trials: list[LesionBatteryTrial] = []
            for seed in seeds:
                trial = self.run_single_trial(
                    env=env,
                    spec=spec,
                    seed=seed,
                    num_steps=num_steps,
                    dt=dt,
                )
                condition_trials.append(trial)
                all_trials.append(trial)

            n = len(condition_trials)
            returns = [t.episode_return for t in condition_trials]
            fwds = [t.forward_distance_mm for t in condition_trials]
            speeds = [t.mean_speed_mm_s for t in condition_trials]
            drifts = [t.lateral_drift_mm for t in condition_trials]
            yaws = [abs(t.net_yaw_deg) for t in condition_trials]
            energies = [t.energy_expenditure for t in condition_trials]
            cots = [t.cost_of_transport for t in condition_trials]
            tcis = [t.tripod_coordination_index for t in condition_trials]

            summary = ConditionSummaryStats(
                condition_name=spec["name"],
                sample_size=n,
                return_mean=float(np.mean(returns)),
                return_std=float(np.std(returns)),
                forward_dist_mean=float(np.mean(fwds)),
                forward_dist_std=float(np.std(fwds)),
                speed_mean=float(np.mean(speeds)),
                speed_std=float(np.std(speeds)),
                lateral_drift_mean=float(np.mean(drifts)),
                lateral_drift_std=float(np.std(drifts)),
                yaw_deg_mean=float(np.mean(yaws)),
                yaw_deg_std=float(np.std(yaws)),
                energy_mean=float(np.mean(energies)),
                energy_std=float(np.std(energies)),
                cot_mean=float(np.mean(cots)),
                cot_std=float(np.std(cots)),
                tci_mean=float(np.mean(tcis)),
                tci_std=float(np.std(tcis)),
            )
            summaries.append(summary)

        return all_trials, summaries


def plot_lesion_battery_results(
    summaries: Sequence[ConditionSummaryStats],
    title: str = "Systematic In-Silico Lesion Battery Impact (Janelia MaleCNS v1.0)",
    out_path: str | Path | None = "outputs/systematic_lesion_battery.png",
) -> plt.Figure:
    """Generate a 4-panel publication-grade figure with standard deviation error bars.

    Panels:
      - Panel A: Net Forward Distance (mm) [mean ± std]
      - Panel B: Steering Asymmetry (|Yaw| deg) [mean ± std]
      - Panel C: Locomotion Speed (mm/s) [mean ± std]
      - Panel D: Tripod Coordination Index (TCI) [mean ± std]
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=150)
    plt.subplots_adjust(hspace=0.45, wspace=0.30)

    names = [s.condition_name for s in summaries]
    # Shorten condition names for clean x-axis ticks
    short_names = [
        n.replace(" Knockout", "").replace(" Control", "")
        for n in names
    ]
    x = np.arange(len(names))
    width = 0.55

    # Palette
    bar_colors = [
        "#27ae60",  # Intact: Green
        "#9b59b6",  # DNa01 Left: Purple
        "#8e44ad",  # DNa01 Right: Dark Purple
        "#c0392b",  # DNa01 Bilateral: Red
        "#2980b9",  # DNa02 Bilateral: Blue
        "#7f8c8d",  # Random Control: Gray
    ]
    if len(bar_colors) < len(names):
        bar_colors = bar_colors + ["#34495e"] * (len(names) - len(bar_colors))

    # -------------------------------------------------------------
    # Panel A: Forward Travel Distance
    # -------------------------------------------------------------
    ax_a = axes[0, 0]
    fwd_m = [s.forward_dist_mean for s in summaries]
    fwd_s = [s.forward_dist_std for s in summaries]
    ax_a.bar(x, fwd_m, yerr=fwd_s, capsize=4, width=width, color=bar_colors, edgecolor="black", linewidth=0.8)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(short_names, rotation=22, ha="right", fontsize=9, fontweight="bold")
    ax_a.set_ylabel("Forward Distance (mm)", fontsize=10)
    ax_a.set_title("A. Net Forward Travel (mean ± std)", fontsize=11, fontweight="bold")
    ax_a.grid(True, linestyle="--", alpha=0.4, axis="y")

    # -------------------------------------------------------------
    # Panel B: Steering Asymmetry (Yaw)
    # -------------------------------------------------------------
    ax_b = axes[0, 1]
    yaw_m = [s.yaw_deg_mean for s in summaries]
    yaw_s = [s.yaw_deg_std for s in summaries]
    ax_b.bar(x, yaw_m, yerr=yaw_s, capsize=4, width=width, color=bar_colors, edgecolor="black", linewidth=0.8)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(short_names, rotation=22, ha="right", fontsize=9, fontweight="bold")
    ax_b.set_ylabel("Steering Deflection (|Yaw| deg)", fontsize=10)
    ax_b.set_title("B. Steering Asymmetry Deflection (mean ± std)", fontsize=11, fontweight="bold")
    ax_b.grid(True, linestyle="--", alpha=0.4, axis="y")

    # -------------------------------------------------------------
    # Panel C: Locomotion Speed
    # -------------------------------------------------------------
    ax_c = axes[1, 0]
    spd_m = [s.speed_mean for s in summaries]
    spd_s = [s.speed_std for s in summaries]
    ax_c.bar(x, spd_m, yerr=spd_s, capsize=4, width=width, color=bar_colors, edgecolor="black", linewidth=0.8)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(short_names, rotation=22, ha="right", fontsize=9, fontweight="bold")
    ax_c.set_ylabel("Speed (mm/s)", fontsize=10)
    ax_c.set_title("C. Locomotion Speed (mean ± std)", fontsize=11, fontweight="bold")
    ax_c.grid(True, linestyle="--", alpha=0.4, axis="y")

    # -------------------------------------------------------------
    # Panel D: Tripod Coordination Index
    # -------------------------------------------------------------
    ax_d = axes[1, 1]
    tci_m = [s.tci_mean for s in summaries]
    tci_s = [s.tci_std for s in summaries]
    ax_d.bar(x, tci_m, yerr=tci_s, capsize=4, width=width, color=bar_colors, edgecolor="black", linewidth=0.8)
    ax_d.axhline(0.50, color="gray", linestyle="--", alpha=0.6, label="Ideal Alternation (0.50)")
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(short_names, rotation=22, ha="right", fontsize=9, fontweight="bold")
    ax_d.set_ylabel("TCI Score", fontsize=10)
    ax_d.set_title("D. Tripod Coordination Index (mean ± std)", fontsize=11, fontweight="bold")
    ax_d.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax_d.legend(loc="upper right", fontsize=8)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)

    if out_path is not None:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


def save_battery_results(
    trials: Sequence[LesionBatteryTrial],
    summaries: Sequence[ConditionSummaryStats],
    out_path: str | Path = "outputs/systematic_lesion_battery.json",
) -> Path:
    """Save raw trials and statistical summaries to a JSON document."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summaries": [s.to_dict() for s in summaries],
        "trials": [t.to_dict() for t in trials],
    }
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return p

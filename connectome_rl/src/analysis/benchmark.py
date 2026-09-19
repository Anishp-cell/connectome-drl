"""Scientific Benchmarking Engine for Drosophila Embodied Locomotion.

Provides quantitative head-to-head comparison between policy architectures:
  1. Biological Connectome Policy (Janelia MaleCNS v1.0 constrained)
  2. Unconstrained MLP Baseline (Experimental Control)

Evaluates three primary scientific pillars:
  - Pillar 1: Sample Efficiency (learning curve progression, episodic returns per env step).
  - Pillar 2: Energy Efficiency (cumulative joint action effort and mechanical Cost of Transport).
  - Pillar 3: Gait Coordination (Tripod Coordination Index, step frequencies, bilateral symmetry).
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

from connectome_rl.src.analysis.kinematics import KinematicAnalyzer, LEG_NAMES
from connectome_rl.src.envs.fly_wrapper import FlyLocomotionEnv


@dataclass
class ModelBenchmarkResult:
    """Quantitative performance and biomechanical scorecard for a policy architecture."""

    model_name: str
    seed: int
    episode_return: float
    forward_distance_mm: float
    mean_speed_mm_s: float
    lateral_drift_mm: float
    net_yaw_deg: float
    energy_expenditure: float           # Cumulative squared action norm: sum(||a_t||^2)
    cost_of_transport: float            # Energy per mm traveled: Energy / max(0.01, forward_distance)
    tripod_coordination_index: float    # TCI in [0, 1] measuring alternating tripod gait
    mean_step_frequency_hz: float       # Average stepping rate in Hz
    straightness_index: float           # Net displacement / Total path length in [0, 1]
    learning_steps: list[int] = field(default_factory=list)
    learning_returns: list[float] = field(default_factory=list)
    trajectory_x: list[float] = field(default_factory=list)
    trajectory_y: list[float] = field(default_factory=list)
    duty_factors: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert result scorecard to serializable dictionary."""
        return asdict(self)


def compute_energy_and_cot(actions: np.ndarray, forward_distance_mm: float) -> tuple[float, float]:
    """Compute cumulative joint effort and mechanical Cost of Transport (CoT).

    Args:
        actions: Array of shape (T, 42) containing joint torque targets in [-1, 1].
        forward_distance_mm: Net forward distance traveled in millimeters.

    Returns:
        (total_energy, cost_of_transport): Total squared effort and effort per mm.
    """
    if len(actions) == 0:
        return 0.0, 0.0

    # Total squared action norm across all 42 DOFs: sum_{t=1}^T sum_{i=1}^{42} a_{t,i}^2
    total_energy = float(np.sum(np.square(actions)))
    effective_dist = max(0.01, forward_distance_mm)
    cot = total_energy / effective_dist
    return total_energy, cot


def extract_deterministic_action(policy: nn.Module, obs_tensor: torch.Tensor) -> np.ndarray:
    """Extract deterministic action vector from any supported policy architecture."""
    with torch.no_grad():
        if hasattr(policy, "forward_actor"):
            action_mean, _ = policy.forward_actor(obs_tensor)
            return action_mean.squeeze(0).cpu().numpy()
        elif hasattr(policy, "actor") and isinstance(policy.actor, nn.Sequential):
            return policy.actor(obs_tensor).squeeze(0).cpu().numpy()
        elif hasattr(policy, "get_action_and_value"):
            out = policy.get_action_and_value(obs_tensor)
            return out[0].squeeze(0).cpu().numpy()
        else:
            raise ValueError(f"Unable to extract action from policy of type {type(policy)}")


def evaluate_policy_locomotion(
    policy: nn.Module,
    env: FlyLocomotionEnv,
    model_name: str = "model",
    num_steps: int = 100,
    seed: int = 42,
    dt: float = 0.002,
) -> ModelBenchmarkResult:
    """Run a standardized closed-loop locomotion evaluation trial.

    Args:
        policy: PyTorch policy module.
        env: FlyLocomotionEnv biomechanical simulation.
        model_name: Human-readable model label (e.g. 'Connectome' or 'MLP Baseline').
        num_steps: Number of policy control steps to evaluate.
        seed: Random seed for environment initialization.
        dt: Control step physics interval in seconds.

    Returns:
        ModelBenchmarkResult populated with kinematic, energetic, and coordination metrics.
    """
    policy.eval()
    obs, _ = env.reset(seed=seed)
    device = next(policy.parameters()).device if list(policy.parameters()) else torch.device("cpu")

    positions: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    contacts: list[np.ndarray] = []
    yaws: list[float] = []

    init_heading = obs[84:87].copy()
    init_pos = env.sim.physics.data.qpos[:3].copy() * 1000.0  # mm
    total_reward = 0.0

    for _ in range(num_steps):
        # Record fly center of mass position (mm)
        cur_pos = env.sim.physics.data.qpos[:3].copy() * 1000.0
        positions.append(cur_pos)

        # Record foot contact forces [LF, LM, LH, RF, RM, RH]
        contact_forces = obs[93:99].copy()
        contacts.append(contact_forces)

        # Record heading yaw
        curr_heading = obs[84:87]
        yaw_rad = np.arctan2(
            init_heading[0] * curr_heading[1] - init_heading[1] * curr_heading[0],
            init_heading[0] * curr_heading[0] + init_heading[1] * curr_heading[1],
        )
        yaws.append(float(np.degrees(yaw_rad)))

        # Compute action
        obs_t = torch.from_numpy(obs).unsqueeze(0).float().to(device)
        act = extract_deterministic_action(policy, obs_t)
        act = np.clip(act, -1.0, 1.0)
        actions.append(act)

        # Step physics
        next_obs, reward, done, truncated, _ = env.step(act)
        total_reward += reward
        obs = next_obs
        if done or truncated:
            break

    # Analyze Kinematics
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

    # Energetics & Cost of Transport
    net_fwd_dist = gait.net_forward_displacement_mm
    total_energy, cot = compute_energy_and_cot(act_arr, net_fwd_dist)
    mean_freq = float(np.mean(list(gait.step_frequencies_hz.values()))) if gait.step_frequencies_hz else 0.0
    lateral_drift = float(pos_arr[-1, 1] - pos_arr[0, 1]) if len(pos_arr) > 1 else 0.0

    return ModelBenchmarkResult(
        model_name=model_name,
        seed=seed,
        episode_return=float(total_reward),
        forward_distance_mm=float(net_fwd_dist),
        mean_speed_mm_s=float(gait.mean_forward_velocity_mm_s),
        lateral_drift_mm=lateral_drift,
        net_yaw_deg=float(gait.net_yaw_angle_deg),
        energy_expenditure=float(total_energy),
        cost_of_transport=float(cot),
        tripod_coordination_index=float(gait.tripod_index),
        mean_step_frequency_hz=mean_freq,
        straightness_index=float(gait.straightness_index),
        trajectory_x=pos_arr[:, 0].tolist(),
        trajectory_y=pos_arr[:, 1].tolist(),
        duty_factors=gait.duty_factors,
    )


def plot_model_comparison(
    results: Sequence[ModelBenchmarkResult],
    title: str = "Biological Connectome vs MLP Baseline Comparison",
    out_path: str | Path | None = "outputs/model_comparison.png",
) -> plt.Figure:
    """Generate a multi-panel scientific figure comparing Connectome vs MLP baseline.

    Panel Layout:
      - Panel A (Top-Left): Sample Efficiency / Learning Returns.
      - Panel B (Top-Right): Mechanical Cost of Transport (Energy Efficiency).
      - Panel C (Bottom-Left): Tripod Coordination Index (Gait Coordination).
      - Panel D (Bottom-Right): Forward Locomotion Velocity vs Lateral Slip.

    Args:
        results: Sequence of ModelBenchmarkResult scorecards.
        title: Figure title.
        out_path: Optional destination path to save image file.

    Returns:
        Matplotlib Figure object.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=150)
    plt.subplots_adjust(hspace=0.35, wspace=0.28)

    model_names = [r.model_name for r in results]
    colors = ["#2ecc71" if "connectome" in m.lower() else "#e74c3c" for m in model_names]
    # Default fallback color if other models
    for i, c in enumerate(colors):
        if "connectome" not in model_names[i].lower() and "mlp" not in model_names[i].lower():
            colors[i] = "#3498db"

    x_pos = np.arange(len(model_names))
    bar_width = 0.45

    # -------------------------------------------------------------
    # Panel A: Sample Efficiency / Returns
    # -------------------------------------------------------------
    ax_a = axes[0, 0]
    has_learning_curves = any(len(r.learning_returns) > 0 for r in results)

    if has_learning_curves:
        for r in results:
            if len(r.learning_returns) > 0:
                steps = r.learning_steps if r.learning_steps else list(range(len(r.learning_returns)))
                c = "#2ecc71" if "connectome" in r.model_name.lower() else "#e74c3c"
                ax_a.plot(steps, r.learning_returns, label=r.model_name, color=c, linewidth=2.2, marker="o", markersize=4)
        ax_a.set_xlabel("Environment Steps", fontsize=10)
        ax_a.set_ylabel("Episodic Return", fontsize=10)
        ax_a.set_title("A. Sample Efficiency (Learning Curves)", fontsize=11, fontweight="bold")
        ax_a.legend(loc="lower right", fontsize=9)
    else:
        returns = [r.episode_return for r in results]
        ax_a.bar(x_pos, returns, width=bar_width, color=colors, edgecolor="black", linewidth=0.8)
        ax_a.set_xticks(x_pos)
        ax_a.set_xticklabels(model_names, fontsize=10, fontweight="bold")
        ax_a.set_ylabel("Episodic Return", fontsize=10)
        ax_a.set_title("A. Task Reward Performance", fontsize=11, fontweight="bold")

    ax_a.grid(True, linestyle="--", alpha=0.4, axis="y")

    # -------------------------------------------------------------
    # Panel B: Energy Efficiency (Mechanical Cost of Transport)
    # -------------------------------------------------------------
    ax_b = axes[0, 1]
    cots = [r.cost_of_transport for r in results]
    bars_b = ax_b.bar(x_pos, cots, width=bar_width, color=colors, edgecolor="black", linewidth=0.8)
    ax_b.set_xticks(x_pos)
    ax_b.set_xticklabels(model_names, fontsize=10, fontweight="bold")
    ax_b.set_ylabel("Cost of Transport (Energy / mm)", fontsize=10)
    ax_b.set_title("B. Energy Efficiency (Lower is More Efficient)", fontsize=11, fontweight="bold")
    ax_b.grid(True, linestyle="--", alpha=0.4, axis="y")

    # Add numeric labels on top of bars
    for bar in bars_b:
        h = bar.get_height()
        ax_b.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                      xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    # -------------------------------------------------------------
    # Panel C: Gait Coordination (Tripod Coordination Index)
    # -------------------------------------------------------------
    ax_c = axes[1, 0]
    tcis = [r.tripod_coordination_index for r in results]
    bars_c = ax_c.bar(x_pos, tcis, width=bar_width, color=colors, edgecolor="black", linewidth=0.8)
    ax_c.axhline(0.50, color="gray", linestyle="--", alpha=0.6, label="Tripod Threshold (0.50)")
    ax_c.set_ylim(0.0, 1.0)
    ax_c.set_xticks(x_pos)
    ax_c.set_xticklabels(model_names, fontsize=10, fontweight="bold")
    ax_c.set_ylabel("Tripod Coordination Index (TCI)", fontsize=10)
    ax_c.set_title("C. Gait Coordination (Biological Alternation)", fontsize=11, fontweight="bold")
    ax_c.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax_c.legend(loc="upper right", fontsize=8)

    for bar in bars_c:
        h = bar.get_height()
        ax_c.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                      xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    # -------------------------------------------------------------
    # Panel D: Forward Velocity vs Lateral Drift
    # -------------------------------------------------------------
    ax_d = axes[1, 1]
    speeds = [r.mean_speed_mm_s for r in results]
    drifts = [abs(r.lateral_drift_mm) for r in results]

    w_sub = 0.28
    ax_d.bar(x_pos - w_sub / 2, speeds, width=w_sub, label="Forward Speed (mm/s)", color="#2980b9", edgecolor="black", linewidth=0.8)
    ax_d.bar(x_pos + w_sub / 2, drifts, width=w_sub, label="Lateral Drift (|mm|)", color="#d35400", edgecolor="black", linewidth=0.8)
    ax_d.set_xticks(x_pos)
    ax_d.set_xticklabels(model_names, fontsize=10, fontweight="bold")
    ax_d.set_ylabel("Metric Value", fontsize=10)
    ax_d.set_title("D. Speed vs Straight-Line Stability", fontsize=11, fontweight="bold")
    ax_d.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax_d.legend(loc="best", fontsize=8)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)

    if out_path is not None:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


def save_benchmark_results(
    results: Sequence[ModelBenchmarkResult],
    out_path: str | Path = "outputs/model_comparison.json",
) -> Path:
    """Save benchmark scorecard data to structured JSON file."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = [r.to_dict() for r in results]
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return p

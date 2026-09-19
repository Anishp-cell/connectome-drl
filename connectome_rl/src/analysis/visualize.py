"""Visualization Suite for Drosophila Biomechanical Locomotion and Connectome Lesions.

Provides publication-grade scientific figures and video export utilities:
  1. Multi-panel locomotion dashboard:
     - 2D trajectory arena tracking with heading orientation.
     - 6-leg stance/swing footfall raster diagram (tripod alternation).
     - Quantitative bar charts (forward speed, lateral drift, yaw turning asymmetry).
     - Bilateral leg duty factor symmetry profiles.
  2. Side-by-side animated video composition (.gif / .mp4):
     - Synchronized comparison of intact vs lesioned fly.
     - Dynamic HUD telemetry banner overlays (step counter, live yaw angle, forward speed).
  3. Lesion battery benchmark comparison summaries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import imageio
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from connectome_rl.src.analysis.kinematics import (
    GaitMetrics,
    KinematicAnalyzer,
    LEG_NAMES,
    TRIPOD_1_INDICES,
    TRIPOD_2_INDICES,
)
from connectome_rl.src.analysis.lesion import LesionResult


# Standard high-contrast scientific palette
PALETTE = {
    "intact": "#2ca02c",      # Forest Green
    "dna01_left": "#9467bd",  # Purple
    "dna01_both": "#d62728",  # Crimson Red
    "dna02_both": "#1f77b4",  # Royal Blue
    "random": "#7f7f7f",      # Charcoal Gray
    "tripod_1": "#1f77b4",    # Blue
    "tripod_2": "#ff7f0e",    # Orange
}


def draw_hud_banner(
    frame: np.ndarray,
    label_left: str,
    label_right: str,
    step_num: int,
    telemetry_left: dict[str, Any] | None = None,
    telemetry_right: dict[str, Any] | None = None,
) -> np.ndarray:
    """Overlay a modern dark HUD banner on top of side-by-side video frames.

    Args:
        frame: Combined RGB image array of shape (H, W, 3).
        label_left: Condition name for the left fly.
        label_right: Condition name for the right fly.
        step_num: Current simulation step index.
        telemetry_left: Optional live metrics for the left fly (e.g. yaw, speed).
        telemetry_right: Optional live metrics for the right fly.

    Returns:
        Annotated RGB image array of the same shape.
    """
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    w, h = img.size
    mid_x = w // 2

    # Draw semi-transparent header bar
    banner_height = 46
    draw.rectangle([(0, 0), (w, banner_height)], fill=(16, 20, 28, 230))
    # Vertical dividing line between left and right views
    draw.line([(mid_x, 0), (mid_x, h)], fill=(220, 220, 220), width=2)

    # Left condition text
    yaw_l = telemetry_left.get("yaw_deg", 0.0) if telemetry_left else 0.0
    text_l = f"{label_left} | Yaw: {yaw_l:+.1f}°"
    draw.text((16, 14), text_l, fill=(80, 210, 255))

    # Right condition text
    yaw_r = telemetry_right.get("yaw_deg", 0.0) if telemetry_right else 0.0
    text_r = f"{label_right} | Yaw: {yaw_r:+.1f}°"
    draw.text((mid_x + 16, 14), text_r, fill=(255, 120, 90))

    # Center step counter
    step_text = f"Step {step_num:03d}"
    draw.text((mid_x - 34, 14), step_text, fill=(230, 230, 230))

    return np.array(img)


def compose_side_by_side_video(
    frames_left: Sequence[np.ndarray],
    frames_right: Sequence[np.ndarray],
    label_left: str = "INTACT CONTROL",
    label_right: str = "LESIONED FLY",
    yaws_left: Sequence[float] | None = None,
    yaws_right: Sequence[float] | None = None,
    fps: int = 8,
    out_path: str | Path = "outputs/side_by_side_comparison.gif",
) -> Path:
    """Compose and save a synchronized side-by-side comparison video with telemetry HUD.

    Args:
        frames_left: List of RGB frames for condition A.
        frames_right: List of RGB frames for condition B.
        label_left: Label text for condition A.
        label_right: Label text for condition B.
        yaws_left: Optional series of heading yaw angles for condition A.
        yaws_right: Optional series of heading yaw angles for condition B.
        fps: Playback frame rate (default 8 for slow-motion stride inspection).
        out_path: Destination file path (.gif or .mp4).

    Returns:
        Resolved Path to the saved video file.
    """
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    min_frames = min(len(frames_left), len(frames_right))
    if min_frames == 0:
        raise ValueError("Cannot compose video: frame sequence is empty.")

    annotated_frames: list[np.ndarray] = []

    for i in range(min_frames):
        fl = frames_left[i]
        fr = frames_right[i]
        combined = np.concatenate([fl, fr], axis=1)

        t_l = {"yaw_deg": yaws_left[i]} if yaws_left and i < len(yaws_left) else None
        t_r = {"yaw_deg": yaws_right[i]} if yaws_right and i < len(yaws_right) else None

        annotated = draw_hud_banner(
            frame=combined,
            label_left=label_left,
            label_right=label_right,
            step_num=i + 1,
            telemetry_left=t_l,
            telemetry_right=t_r,
        )
        annotated_frames.append(annotated)

    # Save to disk
    if out_file.suffix.lower() == ".gif":
        # Duration per frame in seconds for imageio v3 / pillow plugin
        duration_ms = 1000.0 / max(1, fps)
        imageio.mimsave(str(out_file), annotated_frames, duration=duration_ms, loop=0)
    else:
        imageio.mimsave(str(out_file), annotated_frames, fps=fps)

    return out_file



def plot_publication_dashboard(
    trajectories: dict[str, np.ndarray],
    footfall_contacts: np.ndarray | None = None,
    dt: float = 0.002,
    metrics_by_condition: dict[str, GaitMetrics] | None = None,
    title: str = "Drosophila Biomechanical Locomotion & Connectome Analysis",
    out_path: str | Path | None = None,
) -> plt.Figure:
    """Generate a multi-panel publication-grade scientific figure.

    Panel Layout:
      - Panel A (Top-Left): 2D Locomotion Trajectories in MuJoCo arena.
      - Panel B (Top-Right): 6-Leg Stance/Swing Footfall Diagram (Tripod Alternation).
      - Panel C (Bottom-Left): Forward Velocity & Net Heading Turn Comparison.
      - Panel D (Bottom-Right): Per-Leg Duty Factor Symmetry Profiles.

    Args:
        trajectories: Mapping from condition name to (T, 2) or (T, 3) CoM positions.
        footfall_contacts: Optional (T, 6) contact force matrix for the footfall raster.
        dt: Physics step interval in seconds.
        metrics_by_condition: Mapping from condition name to GaitMetrics scorecard.
        title: Master figure title.
        out_path: Destination path to save the generated image.

    Returns:
        Matplotlib Figure instance.
    """
    fig = plt.figure(figsize=(14, 9), dpi=150)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.25)

    # -------------------------------------------------------------
    # Panel A: 2D Locomotion Trajectories
    # -------------------------------------------------------------
    ax_traj = fig.add_subplot(gs[0, 0])
    color_map = {
        "Intact": PALETTE["intact"],
        "DNa01 Left": PALETTE["dna01_left"],
        "DNa01 Bilateral": PALETTE["dna01_both"],
        "DNa02 Bilateral": PALETTE["dna02_both"],
        "Random Control": PALETTE["random"],
    }

    for name, pos in trajectories.items():
        p = np.asarray(pos, dtype=np.float32)
        if len(p) == 0:
            continue
        c = color_map.get(name, "#333333")
        # Plot trajectory line
        ax_traj.plot(p[:, 0], p[:, 1], label=name, color=c, linewidth=2.2, alpha=0.85)
        # Start marker
        ax_traj.scatter(p[0, 0], p[0, 1], color=c, marker="o", s=36, zorder=5)
        # End marker
        ax_traj.scatter(p[-1, 0], p[-1, 1], color=c, marker="x", s=50, linewidths=2.0, zorder=5)

    ax_traj.set_title("A. 2D Locomotion Trajectories (MuJoCo Arena)", fontsize=11, fontweight="bold")
    ax_traj.set_xlabel("Forward X Position (mm)", fontsize=10)
    ax_traj.set_ylabel("Lateral Y Position (mm)", fontsize=10)
    ax_traj.grid(True, linestyle="--", alpha=0.5)
    ax_traj.legend(loc="best", fontsize=9, framealpha=0.9)

    # -------------------------------------------------------------
    # Panel B: Footfall Stance/Swing Raster Diagram
    # -------------------------------------------------------------
    ax_foot = fig.add_subplot(gs[0, 1])
    if footfall_contacts is not None and len(footfall_contacts) > 0:
        analyzer = KinematicAnalyzer()
        binary = analyzer.threshold_contacts(footfall_contacts)
        tci = analyzer.compute_tripod_index(binary)
        freqs = analyzer.compute_step_frequencies(binary, dt=dt)
        mean_freq = float(np.mean(list(freqs.values())))

        total_frames = binary.shape[0]
        time_axis = np.arange(total_frames) * dt
        leg_colors = [
            PALETTE["tripod_1"], PALETTE["tripod_2"], PALETTE["tripod_1"],
            PALETTE["tripod_2"], PALETTE["tripod_1"], PALETTE["tripod_2"]
        ]

        for i, leg_name in enumerate(LEG_NAMES):
            stance_mask = binary[:, i]
            color = leg_colors[i]
            starts = np.where(np.diff(np.pad(stance_mask.astype(int), (1, 1))) == 1)[0]
            ends = np.where(np.diff(np.pad(stance_mask.astype(int), (1, 1))) == -1)[0]

            for s, e in zip(starts, ends):
                ax_foot.barh(
                    y=5 - i,
                    width=(e - s) * dt,
                    left=s * dt,
                    height=0.62,
                    color=color,
                    edgecolor="none",
                    align="center",
                )

        ax_foot.set_yticks(range(6))
        ax_foot.set_yticklabels(list(reversed(LEG_NAMES)), fontsize=9, fontweight="bold")
        ax_foot.set_xlabel("Time (seconds)", fontsize=10)
        ax_foot.set_ylabel("Leg", fontsize=10)
        ax_foot.set_title(
            f"B. 6-Leg Gait Stance Phase (TCI: {tci:.2f} | Freq: {mean_freq:.1f} Hz)",
            fontsize=11,
            fontweight="bold",
        )
        ax_foot.grid(True, linestyle="--", alpha=0.5, axis="x")
        ax_foot.set_xlim(0, max(0.01, time_axis[-1] if len(time_axis) > 0 else 1.0))

        # Legend
        handles = [
            Patch(facecolor=PALETTE["tripod_1"], label="Tripod 1 (LF, RM, LH)"),
            Patch(facecolor=PALETTE["tripod_2"], label="Tripod 2 (RF, LM, RH)"),
        ]
        ax_foot.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)
    else:
        ax_foot.text(0.5, 0.5, "Footfall Telemetry Not Provided", ha="center", va="center", color="gray")
        ax_foot.set_title("B. 6-Leg Gait Stance Phase", fontsize=11, fontweight="bold")

    # -------------------------------------------------------------
    # Panel C: Forward Velocity vs Steering Asymmetry
    # -------------------------------------------------------------
    ax_bars = fig.add_subplot(gs[1, 0])
    if metrics_by_condition is not None and len(metrics_by_condition) > 0:
        cond_names = list(metrics_by_condition.keys())
        v_forwards = [m.mean_forward_velocity_mm_s for m in metrics_by_condition.values()]
        yaws = [abs(m.net_yaw_angle_deg) for m in metrics_by_condition.values()]

        x_indices = np.arange(len(cond_names))
        bar_w = 0.35

        b1 = ax_bars.bar(x_indices - bar_w / 2, v_forwards, width=bar_w, label="Forward Velocity (mm/s)", color="#2b5c8f")
        ax_bars_right = ax_bars.twinx()
        b2 = ax_bars_right.bar(x_indices + bar_w / 2, yaws, width=bar_w, label="Heading Yaw (|deg|)", color="#c0392b")

        ax_bars.set_xticks(x_indices)
        ax_bars.set_xticklabels(cond_names, rotation=18, ha="right", fontsize=9)
        ax_bars.set_ylabel("Velocity (mm/s)", fontsize=10, color="#2b5c8f")
        ax_bars_right.set_ylabel("Net Yaw Angle (|deg|)", fontsize=10, color="#c0392b")
        ax_bars.set_title("C. Forward Speed vs Turning Asymmetry", fontsize=11, fontweight="bold")
        ax_bars.grid(True, linestyle="--", alpha=0.3, axis="y")
    else:
        ax_bars.text(0.5, 0.5, "Telemetry Metrics Not Provided", ha="center", va="center", color="gray")
        ax_bars.set_title("C. Forward Speed vs Turning Asymmetry", fontsize=11, fontweight="bold")

    # -------------------------------------------------------------
    # Panel D: Per-Leg Duty Factor Profiles (Left vs Right)
    # -------------------------------------------------------------
    ax_duty = fig.add_subplot(gs[1, 1])
    if metrics_by_condition is not None and len(metrics_by_condition) > 0:
        first_m = next(iter(metrics_by_condition.values()))
        dfs = first_m.duty_factors

        leg_order = ["LF", "LM", "LH", "RF", "RM", "RH"]
        df_vals = [dfs.get(l, 0.0) for l in leg_order]
        bar_colors = ["#3498db", "#3498db", "#3498db", "#e67e22", "#e67e22", "#e67e22"]

        ax_duty.bar(leg_order, df_vals, color=bar_colors, edgecolor="black", linewidth=0.8, alpha=0.85)
        ax_duty.axhline(0.5, color="red", linestyle="--", alpha=0.6, label="Ideal Running Stance (0.50)")
        ax_duty.set_ylim(0.0, 1.0)
        ax_duty.set_ylabel("Duty Factor (Stance Ratio)", fontsize=10)
        ax_duty.set_xlabel("Leg (Left: Blue | Right: Orange)", fontsize=10)
        ax_duty.set_title("D. Bilateral Stance Symmetry Profile", fontsize=11, fontweight="bold")
        ax_duty.grid(True, linestyle="--", alpha=0.4, axis="y")
        ax_duty.legend(loc="upper right", fontsize=8)
    else:
        ax_duty.text(0.5, 0.5, "Duty Factor Telemetry Not Provided", ha="center", va="center", color="gray")
        ax_duty.set_title("D. Bilateral Stance Symmetry Profile", fontsize=11, fontweight="bold")

    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)

    if out_path is not None:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_connectome_lesion_summary(
    results: Sequence[LesionResult],
    out_path: str | Path = "outputs/lesion_impact_summary.png",
) -> Path:
    """Generate a multi-panel scientific scorecard comparing in-silico lesion conditions.

    Args:
        results: Sequence of LesionResult objects from in-silico ablation trials.
        out_path: Destination path for the saved plot image.

    Returns:
        Resolved Path to the saved figure file.
    """
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    conditions = [r.condition for r in results]
    displacements = [r.forward_distance for r in results]
    yaws = [abs(r.net_yaw_rotation) for r in results]
    speeds = [r.mean_speed for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), dpi=150)

    # 1. Forward Displacement
    axes[0].bar(conditions, displacements, color="#2ecc71", edgecolor="black", linewidth=0.8)
    axes[0].set_title("Forward Travel (mm)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Distance (mm)", fontsize=10)
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].grid(True, linestyle="--", alpha=0.4, axis="y")

    # 2. Turning Asymmetry (Yaw)
    axes[1].bar(conditions, yaws, color="#e74c3c", edgecolor="black", linewidth=0.8)
    axes[1].set_title("Steering Asymmetry (|Yaw| deg)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Turn Angle (degrees)", fontsize=10)
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(True, linestyle="--", alpha=0.4, axis="y")

    # 3. Locomotion Speed
    axes[2].bar(conditions, speeds, color="#3498db", edgecolor="black", linewidth=0.8)
    axes[2].set_title("Locomotion Speed (mm/s)", fontsize=11, fontweight="bold")
    axes[2].set_ylabel("Speed (mm/s)", fontsize=10)
    axes[2].tick_params(axis="x", rotation=25)
    axes[2].grid(True, linestyle="--", alpha=0.4, axis="y")


    fig.suptitle("Janelia MaleCNS v1.0 In-Silico Lesion Battery Impact", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return out_file

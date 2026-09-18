"""Kinematics and Gait Analysis Suite for Drosophila Biomechanical Locomotion.

Provides quantitative neuroethological and biomechanical metrics:
  - Stance and swing phase segmentation across all 6 legs (LF, LM, LH, RF, RM, RH).
  - Duty factor computation per leg (fraction of time in ground contact).
  - Step cycle frequency (Hz) and bout duration analysis.
  - Tripod Coordination Index (TCI): quantifying canonical insect alternating tripod gait.
  - Tetrapod and wave gait coordination metrics.
  - Inter-leg phase relationships and contralateral / ipsilateral coupling.
  - Trajectory tortuosity (straightness index), forward speed, lateral slip, and yaw dynamics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np


# Standard biological naming and indices for Drosophila legs in FlyGym:
# Index 0: Left Front (LF)
# Index 1: Left Middle (LM)
# Index 2: Left Hind (LH)
# Index 3: Right Front (RF)
# Index 4: Right Middle (RM)
# Index 5: Right Hind (RH)
LEG_NAMES: tuple[str, ...] = ("LF", "LM", "LH", "RF", "RM", "RH")

# Canonical insect alternating tripod leg groupings:
# Tripod 1 (T1): Left Front (0), Right Middle (4), Left Hind (2)
# Tripod 2 (T2): Right Front (3), Left Middle (1), Right Hind (5)
TRIPOD_1_INDICES: tuple[int, int, int] = (0, 4, 2)
TRIPOD_2_INDICES: tuple[int, int, int] = (3, 1, 5)

# Ipsilateral pairs:
LEFT_LEGS: tuple[int, int, int] = (0, 1, 2)
RIGHT_LEGS: tuple[int, int, int] = (3, 4, 5)

# Contralateral pairs:
FRONT_LEGS: tuple[int, int] = (0, 3)
MIDDLE_LEGS: tuple[int, int] = (1, 4)
HIND_LEGS: tuple[int, int] = (2, 5)


@dataclass
class StepBout:
    """Individual step phase event (stance or swing) for a single leg."""

    leg_idx: int
    leg_name: str
    phase_type: str  # 'stance' or 'swing'
    start_time: float
    end_time: float
    duration: float
    num_frames: int


@dataclass
class GaitMetrics:
    """Comprehensive quantitative locomotion and gait scorecard."""

    # 1. Per-leg duty factors (fraction of time in ground stance)
    duty_factors: dict[str, float] = field(default_factory=dict)
    mean_duty_factor: float = 0.0

    # 2. Stepping frequencies (Hz) and durations
    step_frequencies_hz: dict[str, float] = field(default_factory=dict)
    mean_step_frequency_hz: float = 0.0
    mean_stance_duration_sec: dict[str, float] = field(default_factory=dict)
    mean_swing_duration_sec: dict[str, float] = field(default_factory=dict)

    # 3. Inter-leg Coordination Indices
    tripod_index: float = 0.0
    tetrapod_index: float = 0.0
    wave_gait_index: float = 0.0
    tripod_1_synchrony: float = 0.0
    tripod_2_synchrony: float = 0.0
    inter_tripod_opposition: float = 0.0

    # 4. Trajectory and Heading Kinematics
    mean_forward_velocity_mm_s: float = 0.0
    mean_lateral_velocity_mm_s: float = 0.0
    total_distance_mm: float = 0.0
    net_forward_displacement_mm: float = 0.0
    straightness_index: float = 0.0  # Net / Total path length [0..1]
    net_yaw_angle_deg: float = 0.0
    mean_yaw_rate_deg_s: float = 0.0
    left_right_stance_asymmetry: float = 0.0  # abs(mean_duty_left - mean_duty_right)

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to a plain dictionary for logging and serialization."""
        return asdict(self)

    def summary(self) -> str:
        """Generate a formatted human-readable telemetry summary."""
        lines = [
            "=" * 70,
            "              BIOMECHANICAL GAIT & KINEMATICS SCORECARD",
            "=" * 70,
            f"  Tripod Coordination Index (TCI):  {self.tripod_index:.3f} / 1.000",
            f"  Mean Stepping Frequency:          {self.mean_step_frequency_hz:.2f} Hz",
            f"  Mean Duty Factor (Stance Ratio):  {self.mean_duty_factor:.3f}",
            f"  Left/Right Stance Asymmetry:       {self.left_right_stance_asymmetry:.3f}",
            "-" * 70,
            "  Per-Leg Duty Factors:",
            "    " + " | ".join(f"{leg}: {self.duty_factors.get(leg, 0.0):.2f}" for leg in LEG_NAMES),
            "  Per-Leg Step Frequencies (Hz):",
            "    " + " | ".join(f"{leg}: {self.step_frequencies_hz.get(leg, 0.0):.1f}" for leg in LEG_NAMES),
            "-" * 70,
            f"  Forward Velocity:                 {self.mean_forward_velocity_mm_s:+.2f} mm/s",
            f"  Lateral Drift Velocity:           {self.mean_lateral_velocity_mm_s:+.2f} mm/s",
            f"  Total Distance Traveled:          {self.total_distance_mm:.2f} mm",
            f"  Straightness Index (0 to 1):      {self.straightness_index:.3f}",
            f"  Net Heading Turn (Yaw):           {self.net_yaw_angle_deg:+.2f}°",
            "=" * 70,
        ]
        return "\n".join(lines)


class KinematicAnalyzer:
    """Analyzer for insect biomechanical locomotion and gait coordination."""

    def __init__(self, contact_force_threshold: float = 0.5) -> None:
        """Initialize the KinematicAnalyzer.

        Args:
            contact_force_threshold: Threshold in mN to classify ground contact as stance (1) vs swing (0).
        """
        self.contact_force_threshold = float(contact_force_threshold)

    def threshold_contacts(self, contact_forces: np.ndarray) -> np.ndarray:
        """Convert continuous leg contact forces to binary stance states.

        Args:
            contact_forces: Array of shape (T, 6) or (6,) with ground contact force magnitudes.

        Returns:
            Binary boolean array of the same shape where True = stance, False = swing.
        """
        forces = np.asarray(contact_forces, dtype=np.float32)
        return (forces >= self.contact_force_threshold).astype(bool)

    def compute_duty_factors(
        self,
        contact_binary: np.ndarray,
    ) -> dict[str, float]:
        """Compute duty factor (fraction of time in stance) for each leg.

        Args:
            contact_binary: Boolean array of shape (T, 6).

        Returns:
            Dictionary mapping leg name (e.g. 'LF') to duty factor in [0.0, 1.0].
        """
        if contact_binary.ndim == 1:
            contact_binary = contact_binary[np.newaxis, :]

        total_steps = contact_binary.shape[0]
        if total_steps == 0:
            return {name: 0.0 for name in LEG_NAMES}

        duty_factors: dict[str, float] = {}
        for i, name in enumerate(LEG_NAMES):
            stance_count = int(np.sum(contact_binary[:, i]))
            duty_factors[name] = float(stance_count / total_steps)

        return duty_factors

    def segment_step_bouts(
        self,
        contact_binary: np.ndarray,
        dt: float,
    ) -> dict[str, list[StepBout]]:
        """Segment continuous contact series into discrete stance and swing bouts.

        Args:
            contact_binary: Boolean array of shape (T, 6).
            dt: Time step duration in seconds between consecutive frames.

        Returns:
            Dictionary mapping leg names to lists of StepBout objects.
        """
        if contact_binary.ndim == 1:
            contact_binary = contact_binary[np.newaxis, :]

        total_frames = contact_binary.shape[0]
        bouts_by_leg: dict[str, list[StepBout]] = {name: [] for name in LEG_NAMES}

        for i, name in enumerate(LEG_NAMES):
            leg_series = contact_binary[:, i]
            if total_frames == 0:
                continue

            current_phase = bool(leg_series[0])
            start_frame = 0

            for frame in range(1, total_frames):
                phase = bool(leg_series[frame])
                if phase != current_phase:
                    # Bout finished
                    duration = (frame - start_frame) * dt
                    bouts_by_leg[name].append(
                        StepBout(
                            leg_idx=i,
                            leg_name=name,
                            phase_type="stance" if current_phase else "swing",
                            start_time=start_frame * dt,
                            end_time=frame * dt,
                            duration=duration,
                            num_frames=frame - start_frame,
                        )
                    )
                    current_phase = phase
                    start_frame = frame

            # Final ongoing bout
            duration = (total_frames - start_frame) * dt
            bouts_by_leg[name].append(
                StepBout(
                    leg_idx=i,
                    leg_name=name,
                    phase_type="stance" if current_phase else "swing",
                    start_time=start_frame * dt,
                    end_time=total_frames * dt,
                    duration=duration,
                    num_frames=total_frames - start_frame,
                )
            )

        return bouts_by_leg

    def compute_step_frequencies(
        self,
        contact_binary: np.ndarray,
        dt: float,
    ) -> dict[str, float]:
        """Compute stepping cycle frequency in Hz for each of the 6 legs.

        A step cycle is defined from the start of a stance phase to the start of the next stance phase.

        Args:
            contact_binary: Boolean array of shape (T, 6).
            dt: Time step duration in seconds.

        Returns:
            Dictionary mapping leg name to frequency in Hz.
        """
        bouts = self.segment_step_bouts(contact_binary, dt)
        frequencies: dict[str, float] = {}

        for name, leg_bouts in bouts.items():
            # Find all stance start times (touchdowns)
            touchdowns = [b.start_time for b in leg_bouts if b.phase_type == "stance"]
            if len(touchdowns) < 2:
                # Less than one full cycle observed
                frequencies[name] = 0.0
            else:
                cycle_periods = np.diff(touchdowns)
                # Filter out spurious zero periods if any
                valid_periods = cycle_periods[cycle_periods > 0]
                if len(valid_periods) > 0:
                    mean_period = float(np.mean(valid_periods))
                    frequencies[name] = float(1.0 / mean_period) if mean_period > 0 else 0.0
                else:
                    frequencies[name] = 0.0

        return frequencies

    def compute_tripod_index(self, contact_binary: np.ndarray) -> float:
        """Compute the Tripod Coordination Index (TCI).

        The canonical insect tripod gait requires:
          - Tripod 1 (LF, RM, LH) moves synchronously in stance while Tripod 2 (RF, LM, RH) is in swing.
          - Tripod 2 moves synchronously in stance while Tripod 1 is in swing.

        The Tripod Index combines:
          1. In-group synchrony of T1: pairwise agreement among LF, RM, LH.
          2. In-group synchrony of T2: pairwise agreement among RF, LM, RH.
          3. Anti-group opposition: penalty for both tripods being simultaneously planted or both in flight.

        Args:
            contact_binary: Boolean array of shape (T, 6).

        Returns:
            Normalized score in [0.0, 1.0], where 1.0 = perfect alternating tripod gait.
        """
        if contact_binary.ndim == 1:
            contact_binary = contact_binary[np.newaxis, :]

        total_frames = contact_binary.shape[0]
        if total_frames == 0:
            return 0.0

        # Cast to float in {0.0, 1.0}
        c = contact_binary.astype(np.float32)

        # Tripod 1 legs: LF (0), RM (4), LH (2)
        t1_legs = c[:, TRIPOD_1_INDICES]  # (T, 3)
        # Tripod 2 legs: RF (3), LM (1), RH (5)
        t2_legs = c[:, TRIPOD_2_INDICES]  # (T, 3)

        # In-group synchrony: average pairwise agreement (1 - |a - b|)
        # For 3 legs: pairs (0,1), (0,2), (1,2)
        def group_synchrony(g: np.ndarray) -> float:
            sync_01 = 1.0 - np.abs(g[:, 0] - g[:, 1])
            sync_02 = 1.0 - np.abs(g[:, 0] - g[:, 2])
            sync_12 = 1.0 - np.abs(g[:, 1] - g[:, 2])
            return float(np.mean((sync_01 + sync_02 + sync_12) / 3.0))

        sync_t1 = group_synchrony(t1_legs)
        sync_t2 = group_synchrony(t2_legs)

        # Group activation: average fraction of legs active in T1 and T2
        t1_active = np.mean(t1_legs, axis=1)  # (T,) in [0, 1]
        t2_active = np.mean(t2_legs, axis=1)  # (T,) in [0, 1]

        # Opposition: T1 and T2 should alternate (|t1_active - t2_active|)
        # When one is in stance (1.0) and other is in swing (0.0), difference is 1.0
        # When both are stance (all 6 legs down), difference is 0.0
        opposition = float(np.mean(np.abs(t1_active - t2_active)))

        # Also verify that stepping is occurring (not standing perfectly still with 0 Hz)
        # Variance of contact over time indicates dynamic stepping
        has_stepping = float(np.clip(np.mean(np.var(c, axis=0)) * 4.0, 0.0, 1.0))

        # Combined Tripod Coordination Index:
        # High synchrony within T1 and T2 + High opposition between T1 and T2 + Stepping activity
        raw_tci = ((sync_t1 + sync_t2) / 2.0) * opposition

        # Scale by stepping activity factor so a motionless standing fly (all 6 feet down)
        # does not falsely score as an alternating tripod
        final_tci = float(np.clip(raw_tci * (0.5 + 0.5 * has_stepping), 0.0, 1.0))
        return final_tci

    def compute_coordination_breakdown(self, contact_binary: np.ndarray) -> dict[str, float]:
        """Detailed breakdown of gait coordination components."""
        if contact_binary.ndim == 1:
            contact_binary = contact_binary[np.newaxis, :]

        c = contact_binary.astype(np.float32)
        t1_legs = c[:, TRIPOD_1_INDICES]
        t2_legs = c[:, TRIPOD_2_INDICES]

        def group_sync(g: np.ndarray) -> float:
            return float(np.mean((
                (1.0 - np.abs(g[:, 0] - g[:, 1])) +
                (1.0 - np.abs(g[:, 0] - g[:, 2])) +
                (1.0 - np.abs(g[:, 1] - g[:, 2]))
            ) / 3.0))

        t1_sync = group_sync(t1_legs)
        t2_sync = group_sync(t2_legs)

        t1_active = np.mean(t1_legs, axis=1)
        t2_active = np.mean(t2_legs, axis=1)
        opposition = float(np.mean(np.abs(t1_active - t2_active)))

        # Tetrapod index: 4 legs on ground at any time (common in slower insect walking)
        legs_on_ground = np.sum(c, axis=1)  # (T,)
        tetrapod_ratio = float(np.mean(legs_on_ground == 4))

        # Wave gait index: 5 legs on ground at any time
        wave_ratio = float(np.mean(legs_on_ground == 5))

        return {
            "tripod_index": self.compute_tripod_index(contact_binary),
            "tripod_1_synchrony": t1_sync,
            "tripod_2_synchrony": t2_sync,
            "inter_tripod_opposition": opposition,
            "tetrapod_index": tetrapod_ratio,
            "wave_gait_index": wave_ratio,
        }

    def compute_trajectory_metrics(
        self,
        positions: np.ndarray,
        yaws: np.ndarray | None = None,
        dt: float = 0.002,
    ) -> dict[str, float]:
        """Compute spatial trajectory kinematics and heading stability.

        Args:
            positions: Array of shape (T, 2) or (T, 3) containing fly CoM [X, Y, (Z)] coordinates in mm.
            yaws: Optional array of shape (T,) containing fly yaw heading angles in degrees.
            dt: Time step duration in seconds.

        Returns:
            Dictionary with forward velocity, lateral drift, path tortuosity, and yaw rates.
        """
        pos = np.asarray(positions, dtype=np.float32)
        total_frames = pos.shape[0]

        if total_frames < 2:
            return {
                "mean_forward_velocity_mm_s": 0.0,
                "mean_lateral_velocity_mm_s": 0.0,
                "total_distance_mm": 0.0,
                "net_forward_displacement_mm": 0.0,
                "straightness_index": 1.0,
                "net_yaw_angle_deg": 0.0,
                "mean_yaw_rate_deg_s": 0.0,
            }

        # Step displacements in mm
        d_pos = np.diff(pos[:, :2], axis=0)  # (T-1, 2) [dx, dy]
        step_distances = np.linalg.norm(d_pos, axis=1)
        total_distance = float(np.sum(step_distances))

        # Net displacement from start to finish
        net_vec = pos[-1, :2] - pos[0, :2]
        net_distance = float(np.linalg.norm(net_vec))
        net_forward = float(net_vec[0])
        net_lateral = float(net_vec[1])

        total_time = (total_frames - 1) * dt
        v_forward = net_forward / total_time if total_time > 0 else 0.0
        v_lateral = net_lateral / total_time if total_time > 0 else 0.0

        # Straightness index: ratio of net Euclidean displacement to total path length
        # S = 1.0 indicates a perfectly straight path; S ~ 0 indicates looping or meandering
        straightness = float(net_distance / (total_distance + 1e-6))
        straightness = float(np.clip(straightness, 0.0, 1.0))

        # Yaw turning dynamics
        net_yaw = 0.0
        mean_yaw_rate = 0.0
        if yaws is not None and len(yaws) >= 2:
            y_arr = np.asarray(yaws, dtype=np.float32)
            net_yaw = float(y_arr[-1] - y_arr[0])
            dyaws = np.abs(np.diff(y_arr))
            mean_yaw_rate = float(np.mean(dyaws) / dt) if dt > 0 else 0.0

        return {
            "mean_forward_velocity_mm_s": v_forward,
            "mean_lateral_velocity_mm_s": v_lateral,
            "total_distance_mm": total_distance,
            "net_forward_displacement_mm": net_forward,
            "straightness_index": straightness,
            "net_yaw_angle_deg": net_yaw,
            "mean_yaw_rate_deg_s": mean_yaw_rate,
        }

    def analyze_rollout(
        self,
        contact_forces: np.ndarray,
        positions: np.ndarray,
        yaws: np.ndarray | None = None,
        dt: float = 0.002,
    ) -> GaitMetrics:
        """Run complete end-to-end kinematic analysis on a simulated rollout.

        Args:
            contact_forces: Array of shape (T, 6) with ground contact forces.
            positions: Array of shape (T, 2) or (T, 3) with fly CoM positions in mm.
            yaws: Optional array of shape (T,) with fly yaw angles in degrees.
            dt: Time step duration in seconds.

        Returns:
            GaitMetrics instance with full quantitative locomotion telemetry.
        """
        contact_binary = self.threshold_contacts(contact_forces)
        duty_factors = self.compute_duty_factors(contact_binary)
        step_freqs = self.compute_step_frequencies(contact_binary, dt)
        bouts = self.segment_step_bouts(contact_binary, dt)
        coord = self.compute_coordination_breakdown(contact_binary)
        traj = self.compute_trajectory_metrics(positions, yaws, dt)

        # Mean stance and swing durations per leg
        mean_stance: dict[str, float] = {}
        mean_swing: dict[str, float] = {}
        for leg, leg_bouts in bouts.items():
            st_durs = [b.duration for b in leg_bouts if b.phase_type == "stance"]
            sw_durs = [b.duration for b in leg_bouts if b.phase_type == "swing"]
            mean_stance[leg] = float(np.mean(st_durs)) if st_durs else 0.0
            mean_swing[leg] = float(np.mean(sw_durs)) if sw_durs else 0.0

        mean_df = float(np.mean(list(duty_factors.values())))
        mean_freq = float(np.mean(list(step_freqs.values())))

        # Left vs right duty asymmetry
        left_df = np.mean([duty_factors[l] for l in ("LF", "LM", "LH")])
        right_df = np.mean([duty_factors[r] for r in ("RF", "RM", "RH")])
        asym = float(abs(left_df - right_df))

        return GaitMetrics(
            duty_factors=duty_factors,
            mean_duty_factor=mean_df,
            step_frequencies_hz=step_freqs,
            mean_step_frequency_hz=mean_freq,
            mean_stance_duration_sec=mean_stance,
            mean_swing_duration_sec=mean_swing,
            tripod_index=coord["tripod_index"],
            tetrapod_index=coord["tetrapod_index"],
            wave_gait_index=coord["wave_gait_index"],
            tripod_1_synchrony=coord["tripod_1_synchrony"],
            tripod_2_synchrony=coord["tripod_2_synchrony"],
            inter_tripod_opposition=coord["inter_tripod_opposition"],
            mean_forward_velocity_mm_s=traj["mean_forward_velocity_mm_s"],
            mean_lateral_velocity_mm_s=traj["mean_lateral_velocity_mm_s"],
            total_distance_mm=traj["total_distance_mm"],
            net_forward_displacement_mm=traj["net_forward_displacement_mm"],
            straightness_index=traj["straightness_index"],
            net_yaw_angle_deg=traj["net_yaw_angle_deg"],
            mean_yaw_rate_deg_s=traj["mean_yaw_rate_deg_s"],
            left_right_stance_asymmetry=asym,
        )

    def plot_footfall_diagram(
        self,
        contact_forces: np.ndarray,
        dt: float = 0.002,
        title: str | None = None,
        out_path: str | Path | None = None,
    ) -> plt.Figure:
        """Generate a publication-grade biological 6-leg stance/swing footfall diagram.

        Args:
            contact_forces: Array of shape (T, 6) with ground contact force values.
            dt: Time step duration in seconds.
            title: Optional plot title.
            out_path: Optional file path to save the generated figure.

        Returns:
            Matplotlib Figure object.
        """
        contact_binary = self.threshold_contacts(contact_forces)
        tci = self.compute_tripod_index(contact_binary)
        freqs = self.compute_step_frequencies(contact_binary, dt)
        mean_freq = float(np.mean(list(freqs.values())))

        total_frames = contact_binary.shape[0]
        time_axis = np.arange(total_frames) * dt

        fig, ax = plt.subplots(figsize=(10, 4.5), dpi=150)

        # Plot bars for stance phases for each leg
        # Tripod 1 colors vs Tripod 2 colors
        c_t1 = "#1f77b4"  # Blue for Tripod 1 (LF, RM, LH)
        c_t2 = "#ff7f0e"  # Orange for Tripod 2 (RF, LM, RH)

        leg_colors = [c_t1, c_t2, c_t1, c_t2, c_t1, c_t2]

        for i, leg_name in enumerate(LEG_NAMES):
            stance_mask = contact_binary[:, i]
            color = leg_colors[i]

            # Plot continuous segments
            starts = np.where(np.diff(np.pad(stance_mask.astype(int), (1, 1))) == 1)[0]
            ends = np.where(np.diff(np.pad(stance_mask.astype(int), (1, 1))) == -1)[0]

            for s, e in zip(starts, ends):
                t_start = s * dt
                t_end = e * dt
                ax.barh(
                    y=5 - i,
                    width=(t_end - t_start),
                    left=t_start,
                    height=0.6,
                    color=color,
                    edgecolor="none",
                    align="center",
                )

        ax.set_yticks(range(6))
        ax.set_yticklabels(list(reversed(LEG_NAMES)), fontsize=11, fontweight="bold")
        ax.set_xlabel("Time (seconds)", fontsize=11)
        ax.set_ylabel("Leg", fontsize=11)

        plot_title = (
            title
            if title
            else f"Biological 6-Leg Stance-Phase Footfall Diagram\nTripod Index: {tci:.2f} | Step Freq: {mean_freq:.1f} Hz"
        )
        ax.set_title(plot_title, fontsize=12, fontweight="bold", pad=12)
        ax.grid(True, linestyle="--", alpha=0.5, axis="x")
        ax.set_xlim(0, max(0.01, time_axis[-1] if len(time_axis) > 0 else 1.0))

        # Custom legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=c_t1, label="Tripod 1 (LF, RM, LH)"),
            Patch(facecolor=c_t2, label="Tripod 2 (RF, LM, RH)"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", framealpha=0.9)

        plt.tight_layout()

        if out_path is not None:
            p = Path(out_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(p, dpi=150)
            plt.close(fig)

        return fig

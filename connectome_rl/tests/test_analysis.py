"""Unit and integration tests for in-silico neuroscience analysis and lesion suite.

Verifies:
  - TestLesionController: DNa01/DNa02 resolution, non-destructive weight ablation & restoration,
    context manager, partial knockdown, policy architectures (Connectome, Hierarchical, MLP),
    closed-loop rollout evaluation.
  - TestKinematics: Contact thresholding, step frequency, tripod coordination index (TCI),
    trajectory tortuosity, gait scorecard metrics, footfall raster diagram.
  - TestVisualization: Telemetry HUD banner, synchronized side-by-side video, 4-panel
    publication dashboard, lesion impact summary scorecard.
  - TestBenchmark: Actuator effort & Cost of Transport (CoT), deterministic action extraction,
    ModelBenchmarkResult serialization, 4-panel model comparison figure.
  - TestLesionBattery: Systematic multi-seed lesion battery execution, statistical aggregation
    (mean ± std), error-bar figure generation, JSON persistence.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch
import torch.nn as nn

from connectome_rl.src.analysis.battery import (
    ConditionSummaryStats,
    LesionBatteryTrial,
    SystematicLesionBattery,
    plot_lesion_battery_results,
    save_battery_results,
)
from connectome_rl.src.analysis.benchmark import (
    ModelBenchmarkResult,
    compute_energy_and_cot,
    extract_deterministic_action,
    plot_model_comparison,
    save_benchmark_results,
)
from connectome_rl.src.analysis.kinematics import (
    GaitMetrics,
    KinematicAnalyzer,
    LEG_NAMES,
    TRIPOD_1_INDICES,
    TRIPOD_2_INDICES,
)
from connectome_rl.src.analysis.lesion import (
    LesionConfig,
    LesionController,
    LesionResult,
)
from connectome_rl.src.analysis.visualize import (
    compose_side_by_side_video,
    draw_hud_banner,
    plot_connectome_lesion_summary,
    plot_publication_dashboard,
)
from connectome_rl.src.analysis.statistics import (
    StatisticalAnalyzer,
    StatisticalComparison,
    compute_cohens_d,
    compute_welch_t_test,
    format_significance_stars,
)
from connectome_rl.src.connectome.graph_utils import ConnectomeCircuitData, load_circuit_data
from connectome_rl.src.envs.fly_wrapper import FlyLocomotionEnv
from connectome_rl.src.models.connectome_policy import ConnectomePolicy
from connectome_rl.src.models.hierarchical_policy import HierarchicalPolicy
from connectome_rl.src.models.mlp_policy import MLPPolicy


@pytest.fixture
def circuit_data() -> ConnectomeCircuitData:
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "connectome_rl" / "data" / "dna_circuit_tensors.pt"
    if not data_path.exists():
        pytest.skip(f"Biological circuit data not found at {data_path}")
    return load_circuit_data(data_path)


@pytest.fixture
def connectome_policy(circuit_data) -> ConnectomePolicy:
    return ConnectomePolicy(obs_dim=100, act_dim=42, circuit_data=circuit_data)


# ==============================================================================
# 1. TestLesionController
# ==============================================================================
class TestLesionController:
    """Test suite for in-silico ablation controller."""

    def test_target_resolution_dna01_and_dna02(self, connectome_policy, circuit_data) -> None:
        controller = LesionController(policy=connectome_policy, circuit_data=circuit_data)

        # Intact control should return empty indices
        assert controller.resolve_target_indices(LesionConfig(target_type="intact")) == []
        assert controller.resolve_target_indices(LesionConfig(target_type="none")) == []

        # DNa01 Left / Right / Bilateral
        left_idx = controller.resolve_target_indices(LesionConfig(target_type="DNa01", side="left"))
        right_idx = controller.resolve_target_indices(LesionConfig(target_type="DNa01", side="right"))
        both_idx = controller.resolve_target_indices(LesionConfig(target_type="DNa01", side="bilateral"))

        assert len(left_idx) == 1
        assert len(right_idx) == 1
        assert len(both_idx) == 2
        assert set(both_idx) == set(left_idx + right_idx)

    def test_connectome_ablation_and_restoration(self, connectome_policy, circuit_data) -> None:
        controller = LesionController(policy=connectome_policy, circuit_data=circuit_data)
        orig_weights = connectome_policy.hop1_layer.weight.clone()

        ablated = controller.apply_lesion(LesionConfig(target_type="DNa01", side="bilateral"))
        for col in ablated:
            assert torch.all(connectome_policy.hop1_layer.weight[:, col] == 0.0)

        controller.restore()
        assert torch.equal(connectome_policy.hop1_layer.weight, orig_weights)

    def test_active_lesion_context_manager(self, connectome_policy, circuit_data) -> None:
        controller = LesionController(policy=connectome_policy, circuit_data=circuit_data)
        orig_weights = connectome_policy.hop1_layer.weight.clone()

        with controller.active_lesion(LesionConfig(target_type="DNa02", side="bilateral")):
            assert not torch.equal(connectome_policy.hop1_layer.weight, orig_weights)

        assert torch.equal(connectome_policy.hop1_layer.weight, orig_weights)

    def test_partial_knockdown_intensity(self, connectome_policy, circuit_data) -> None:
        controller = LesionController(policy=connectome_policy, circuit_data=circuit_data)
        orig_col = connectome_policy.hop1_layer.weight[:, 1].clone()

        with controller.active_lesion(LesionConfig(target_type="custom", target_indices=[1], intensity=0.5)):
            scaled_col = connectome_policy.hop1_layer.weight[:, 1]
            assert torch.allclose(scaled_col, orig_col * 0.5, atol=1e-5)

        assert torch.equal(connectome_policy.hop1_layer.weight[:, 1], orig_col)

    def test_hierarchical_policy_support(self, circuit_data) -> None:
        h_policy = HierarchicalPolicy(obs_dim=100, act_dim=42, circuit_data=circuit_data)
        controller = LesionController(policy=h_policy, circuit_data=circuit_data)

        orig_w = h_policy.vnc.hop1_layer.weight.clone()
        with controller.active_lesion(LesionConfig(target_type="DNa02", side="bilateral")):
            assert not torch.equal(h_policy.vnc.hop1_layer.weight, orig_w)

        assert torch.equal(h_policy.vnc.hop1_layer.weight, orig_w)

    def test_mlp_baseline_policy_support(self) -> None:
        mlp = MLPPolicy(obs_dim=100, act_dim=42)
        controller = LesionController(policy=mlp)

        orig_w = mlp.actor[0].weight.clone()
        with controller.active_lesion(LesionConfig(target_type="random", num_random_nodes=2)):
            assert not torch.equal(mlp.actor[0].weight, orig_w)

        assert torch.equal(mlp.actor[0].weight, orig_w)

    def test_closed_loop_evaluate_lesion(self, connectome_policy, circuit_data) -> None:
        env = FlyLocomotionEnv(physics_steps_per_action=20, max_episode_steps=15, enable_render=False)
        controller = LesionController(policy=connectome_policy, circuit_data=circuit_data)

        cfg = LesionConfig(target_type="DNa01", side="left")
        result = controller.evaluate_lesion(env=env, config=cfg, num_steps=10)
        env.close()

        assert isinstance(result, LesionResult)
        assert len(result.trajectory_x) > 0


# ==============================================================================
# 2. TestKinematics
# ==============================================================================
class TestKinematics:
    """Test suite for locomotion kinematics, duty factors, and TCI."""

    def test_contact_thresholding_and_duty_factor(self) -> None:
        analyzer = KinematicAnalyzer(contact_force_threshold=0.5)
        raw_forces = np.array([
            [1.0, 0.2, 0.8, 0.1, 0.9, 0.0],
            [0.6, 0.1, 0.7, 0.2, 0.8, 0.1],
            [0.1, 0.8, 0.2, 0.7, 0.1, 0.9],
            [0.0, 0.9, 0.1, 0.8, 0.0, 0.8],
        ])
        binary = analyzer.threshold_contacts(raw_forces)
        assert binary.shape == raw_forces.shape
        assert binary[0, 0] is True or binary[0, 0] == 1
        assert binary[0, 1] is False or binary[0, 1] == 0

        dfs = analyzer.compute_duty_factors(binary)
        assert len(dfs) == 6
        assert dfs["LF"] == 0.50

    def test_step_frequencies_periodic_signal(self) -> None:
        analyzer = KinematicAnalyzer()
        dt = 0.01
        t = 200
        binary = np.zeros((t, 6), dtype=bool)
        for frame in range(t):
            if (frame // 10) % 2 == 0:
                binary[frame, :] = True

        freqs = analyzer.compute_step_frequencies(binary, dt=dt)
        for name in freqs:
            assert pytest.approx(freqs[name], rel=1e-2) == 5.0

    def test_tripod_coordination_index_ideal_vs_static(self) -> None:
        analyzer = KinematicAnalyzer()
        t = 100
        ideal_tripod = np.zeros((t, 6), dtype=bool)
        for frame in range(t):
            if (frame // 10) % 2 == 0:
                ideal_tripod[frame, TRIPOD_1_INDICES] = True
            else:
                ideal_tripod[frame, TRIPOD_2_INDICES] = True

        tci_ideal = analyzer.compute_tripod_index(ideal_tripod)
        assert tci_ideal > 0.85

        static_standing = np.ones((t, 6), dtype=bool)
        tci_static = analyzer.compute_tripod_index(static_standing)
        assert tci_static == 0.0

    def test_trajectory_metrics_straight_vs_circle(self) -> None:
        analyzer = KinematicAnalyzer()
        n = 100
        x_straight = np.linspace(0.0, 10.0, n)
        y_straight = np.zeros(n)
        pos_straight = np.column_stack([x_straight, y_straight])

        res_straight = analyzer.compute_trajectory_metrics(pos_straight, dt=0.01)
        assert pytest.approx(res_straight["straightness_index"], rel=1e-3) == 1.0

        angles = np.linspace(0, 2 * np.pi, n)
        pos_circle = np.column_stack([5.0 * np.cos(angles), 5.0 * np.sin(angles)])
        res_circle = analyzer.compute_trajectory_metrics(pos_circle, dt=0.01)
        assert res_circle["straightness_index"] < 0.05

    def test_end_to_end_analyze_rollout(self) -> None:
        analyzer = KinematicAnalyzer(contact_force_threshold=0.5)
        t = 120
        forces = np.abs(np.random.randn(t, 6)).astype(np.float32)
        positions = np.cumsum(np.abs(np.random.randn(t, 3) * 0.05), axis=0).astype(np.float32)
        yaws = np.linspace(0.0, 15.0, t).astype(np.float32)

        metrics = analyzer.analyze_rollout(
            contact_forces=forces,
            positions=positions,
            yaws=yaws,
            dt=0.002,
        )
        assert isinstance(metrics, GaitMetrics)

    def test_plot_footfall_diagram_generation(self, tmp_path) -> None:
        analyzer = KinematicAnalyzer()
        t = 50
        forces = np.abs(np.random.rand(t, 6)).astype(np.float32)
        out_fig = tmp_path / "footfall_test.png"
        analyzer.plot_footfall_diagram(forces, dt=0.002, out_path=out_fig)
        assert out_fig.exists()
        assert out_fig.stat().st_size > 1000


# ==============================================================================
# 3. TestVisualization
# ==============================================================================
class TestVisualization:
    """Test suite for HUD banner, side-by-side video, and publication figures."""

    def test_draw_hud_banner(self) -> None:
        frame = np.zeros((240, 640, 3), dtype=np.uint8)
        annotated = draw_hud_banner(
            frame=frame,
            label_left="INTACT",
            label_right="DNa01-L",
            step_num=12,
            telemetry_left={"yaw_deg": 1.5},
            telemetry_right={"yaw_deg": -18.2},
        )
        assert annotated.shape == frame.shape
        assert not np.array_equal(annotated, frame)

    def test_compose_side_by_side_video(self, tmp_path) -> None:
        f_left = [np.full((120, 160, 3), 50, dtype=np.uint8) for _ in range(5)]
        f_right = [np.full((120, 160, 3), 100, dtype=np.uint8) for _ in range(5)]
        out_gif = tmp_path / "test_side_by_side.gif"

        res = compose_side_by_side_video(
            frames_left=f_left,
            frames_right=f_right,
            label_left="LEFT",
            label_right="RIGHT",
            out_path=out_gif,
        )
        assert res.exists()
        assert res.stat().st_size > 500

    def test_plot_publication_dashboard(self, tmp_path) -> None:
        trajectories = {
            "Intact": np.array([[0, 0], [5, 1], [10, 2]]),
            "DNa01 Left": np.array([[0, 0], [3, -4], [6, -9]]),
        }
        contacts = np.random.rand(40, 6) > 0.5
        out_img = tmp_path / "test_dashboard.png"

        plot_publication_dashboard(
            trajectories=trajectories,
            footfall_contacts=contacts,
            dt=0.002,
            out_path=out_img,
        )
        assert out_img.exists()
        assert out_img.stat().st_size > 5000

    def test_plot_connectome_lesion_summary(self, tmp_path) -> None:
        results = [
            LesionResult(
                condition="Intact",
                target_type="none",
                side="none",
                episode_return=15.0,
                forward_distance=10.0,
                lateral_drift=0.2,
                net_yaw_rotation=0.5,
                mean_speed=12.0,
                stability_score=0.98,
            ),
            LesionResult(
                condition="DNa01_Left",
                target_type="DNa01",
                side="left",
                episode_return=8.5,
                forward_distance=6.2,
                lateral_drift=-3.5,
                net_yaw_rotation=-32.0,
                mean_speed=7.5,
                stability_score=0.92,
            ),
        ]
        out_summary = tmp_path / "lesion_summary.png"
        res = plot_connectome_lesion_summary(results=results, out_path=out_summary)
        assert res.exists()
        assert res.stat().st_size > 5000


# ==============================================================================
# 4. TestBenchmark
# ==============================================================================
class TestBenchmark:
    """Test suite for scientific model comparison (Connectome vs MLP baseline)."""

    def test_compute_energy_and_cot(self) -> None:
        e0, cot0 = compute_energy_and_cot(np.zeros((0, 42)), forward_distance_mm=10.0)
        assert e0 == 0.0
        assert cot0 == 0.0

        actions = np.full((10, 42), 0.5)
        dist = 20.0
        energy, cot = compute_energy_and_cot(actions, forward_distance_mm=dist)
        assert pytest.approx(energy, rel=1e-3) == 105.0
        assert pytest.approx(cot, rel=1e-3) == 5.25

    def test_extract_deterministic_action(self, circuit_data) -> None:
        obs = torch.randn(1, 100)
        cp = ConnectomePolicy(obs_dim=100, act_dim=42, circuit_data=circuit_data)
        act_cp = extract_deterministic_action(cp, obs)
        assert act_cp.shape == (42,)

        mlp = MLPPolicy(obs_dim=100, act_dim=42)
        act_mlp = extract_deterministic_action(mlp, obs)
        assert act_mlp.shape == (42,)

    def test_model_benchmark_result_and_json_serialization(self, tmp_path) -> None:
        res = ModelBenchmarkResult(
            model_name="Connectome (MaleCNS)",
            seed=42,
            episode_return=25.4,
            forward_distance_mm=18.5,
            mean_speed_mm_s=12.2,
            lateral_drift_mm=0.3,
            net_yaw_deg=0.8,
            energy_expenditure=95.0,
            cost_of_transport=5.13,
            tripod_coordination_index=0.78,
            mean_step_frequency_hz=14.5,
            straightness_index=0.96,
        )
        out_json = tmp_path / "benchmark_test.json"
        saved_file = save_benchmark_results([res], out_path=out_json)
        assert saved_file.exists()
        with open(saved_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["model_name"] == "Connectome (MaleCNS)"

    def test_plot_model_comparison_generation(self, tmp_path) -> None:
        res_conn = ModelBenchmarkResult(
            model_name="Connectome Policy",
            seed=42,
            episode_return=30.0,
            forward_distance_mm=20.0,
            mean_speed_mm_s=15.0,
            lateral_drift_mm=0.2,
            net_yaw_deg=0.5,
            energy_expenditure=110.0,
            cost_of_transport=5.5,
            tripod_coordination_index=0.82,
            mean_step_frequency_hz=16.0,
            straightness_index=0.98,
        )
        res_mlp = ModelBenchmarkResult(
            model_name="MLP Baseline",
            seed=42,
            episode_return=12.0,
            forward_distance_mm=7.0,
            mean_speed_mm_s=4.5,
            lateral_drift_mm=3.5,
            net_yaw_deg=22.0,
            energy_expenditure=210.0,
            cost_of_transport=30.0,
            tripod_coordination_index=0.18,
            mean_step_frequency_hz=9.0,
            straightness_index=0.48,
        )
        out_fig = tmp_path / "model_comp.png"
        fig = plot_model_comparison(results=[res_conn, res_mlp], out_path=out_fig)
        assert out_fig.exists()
        assert out_fig.stat().st_size > 5000


# ==============================================================================
# 5. TestLesionBattery (Phase 6: Step 6.2)
# ==============================================================================
class TestLesionBattery:
    """Test suite for systematic multi-seed lesion battery and error-bar statistics."""

    def test_battery_single_trial_execution(self, connectome_policy, circuit_data) -> None:
        env = FlyLocomotionEnv(physics_steps_per_action=20, max_episode_steps=15, enable_render=False)
        battery = SystematicLesionBattery(policy=connectome_policy, circuit_data=circuit_data)

        spec = {"name": "DNa01 Left Test", "target": "DNa01", "side": "left", "intensity": 1.0}
        trial = battery.run_single_trial(env=env, spec=spec, seed=42, num_steps=10)
        env.close()

        assert isinstance(trial, LesionBatteryTrial)
        assert trial.condition_name == "DNa01 Left Test"
        assert trial.target_type == "DNa01"
        assert trial.seed == 42
        assert not np.isnan(trial.episode_return)

    def test_battery_multi_seed_aggregation(self, connectome_policy, circuit_data) -> None:
        env = FlyLocomotionEnv(physics_steps_per_action=20, max_episode_steps=12, enable_render=False)
        custom_specs = [
            {"name": "Intact", "target": "intact", "side": "both", "intensity": 1.0},
            {"name": "DNa01 Bilateral", "target": "DNa01", "side": "bilateral", "intensity": 1.0},
        ]
        battery = SystematicLesionBattery(
            policy=connectome_policy,
            circuit_data=circuit_data,
            custom_specs=custom_specs,
        )

        trials, summaries = battery.run_battery(env=env, seeds=[42, 43], num_steps=8)
        env.close()

        assert len(trials) == 4  # 2 conditions * 2 seeds
        assert len(summaries) == 2  # 2 conditions
        assert summaries[0].sample_size == 2
        assert summaries[1].sample_size == 2
        assert summaries[0].condition_name == "Intact"
        assert summaries[1].condition_name == "DNa01 Bilateral"
        assert not np.isnan(summaries[0].forward_dist_mean)
        assert not np.isnan(summaries[0].forward_dist_std)

    def test_plot_lesion_battery_results(self, tmp_path) -> None:
        summaries = [
            ConditionSummaryStats(
                condition_name="Intact Control",
                sample_size=3,
                return_mean=35.0, return_std=2.1,
                forward_dist_mean=25.0, forward_dist_std=1.5,
                speed_mean=18.0, speed_std=1.2,
                lateral_drift_mean=0.2, lateral_drift_std=0.05,
                yaw_deg_mean=0.8, yaw_deg_std=0.2,
                energy_mean=80.0, energy_std=5.0,
                cot_mean=3.2, cot_std=0.3,
                tci_mean=0.75, tci_std=0.04,
            ),
            ConditionSummaryStats(
                condition_name="DNa01 Left",
                sample_size=3,
                return_mean=18.0, return_std=3.4,
                forward_dist_mean=12.0, forward_dist_std=2.1,
                speed_mean=9.5, speed_std=1.8,
                lateral_drift_mean=-4.2, lateral_drift_std=0.8,
                yaw_deg_mean=28.5, yaw_deg_std=4.2,
                energy_mean=110.0, energy_std=8.0,
                cot_mean=9.1, cot_std=1.2,
                tci_mean=0.35, tci_std=0.06,
            ),
        ]

        out_fig = tmp_path / "battery_errorbars.png"
        fig = plot_lesion_battery_results(summaries=summaries, out_path=out_fig)
        assert out_fig.exists()
        assert out_fig.stat().st_size > 5000

    def test_save_battery_results(self, tmp_path) -> None:
        trial = LesionBatteryTrial(
            condition_name="Intact",
            target_type="intact",
            side="both",
            seed=42,
            episode_return=30.0,
            forward_distance_mm=20.0,
            mean_speed_mm_s=15.0,
            lateral_drift_mm=0.1,
            net_yaw_deg=0.5,
            energy_expenditure=75.0,
            cost_of_transport=3.75,
            tripod_coordination_index=0.80,
            straightness_index=0.95,
        )
        summary = ConditionSummaryStats(
            condition_name="Intact",
            sample_size=1,
            return_mean=30.0, return_std=0.0,
            forward_dist_mean=20.0, forward_dist_std=0.0,
            speed_mean=15.0, speed_std=0.0,
            lateral_drift_mean=0.1, lateral_drift_std=0.0,
            yaw_deg_mean=0.5, yaw_deg_std=0.0,
            energy_mean=75.0, energy_std=0.0,
            cot_mean=3.75, cot_std=0.0,
            tci_mean=0.80, tci_std=0.0,
        )

        out_json = tmp_path / "battery_data.json"
        saved = save_battery_results(trials=[trial], summaries=[summary], out_path=out_json)
        assert saved.exists()

        with open(saved, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "summaries" in data
        assert "trials" in data
        assert len(data["summaries"]) == 1
        assert len(data["trials"]) == 1


# ==============================================================================
# 6. TestStatistics (Phase 6: Step 6.3)
# ==============================================================================
class TestStatistics:
    """Test suite for statistical hypothesis testing and publication tables."""

    def test_format_significance_stars(self) -> None:
        assert format_significance_stars(0.0001) == "***"
        assert format_significance_stars(0.005) == "**"
        assert format_significance_stars(0.03) == "*"
        assert format_significance_stars(0.05) == "ns"
        assert format_significance_stars(0.42) == "ns"
        assert format_significance_stars(float("nan")) == "ns"

    def test_compute_cohens_d(self) -> None:
        # Identical groups -> 0 effect
        d_zero = compute_cohens_d([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert pytest.approx(d_zero, abs=1e-5) == 0.0

        # High separation -> positive effect
        d_pos = compute_cohens_d([10.0, 11.0, 12.0], [0.0, 1.0, 2.0])
        assert d_pos > 2.0  # Very large effect

        # Empty group handling
        assert compute_cohens_d([], [1.0, 2.0]) == 0.0
        assert compute_cohens_d([1.0], []) == 0.0

    def test_compute_welch_t_test(self) -> None:
        # Identical samples
        t_stat, p_val = compute_welch_t_test([5.0, 5.0, 5.0], [5.0, 5.0, 5.0])
        assert t_stat == 0.0
        assert p_val == 1.0

        # Significantly separated samples
        t_stat_sep, p_val_sep = compute_welch_t_test([100.0, 102.0, 101.0, 103.0], [1.0, 2.0, 1.5, 2.5])
        assert t_stat_sep > 50.0
        assert p_val_sep < 0.001

        # Insufficient samples (< 2)
        t_small, p_small = compute_welch_t_test([1.0], [2.0, 3.0])
        assert t_small == 0.0
        assert p_small == 1.0

    def test_statistical_analyzer_compare_metric(self) -> None:
        trials_by_condition = {
            "Intact Control": [
                {"forward_distance_mm": 20.0, "mean_speed_mm_s": 15.0},
                {"forward_distance_mm": 22.0, "mean_speed_mm_s": 16.0},
                {"forward_distance_mm": 21.0, "mean_speed_mm_s": 15.5},
            ],
            "DNa01 Left": [
                {"forward_distance_mm": 5.0, "mean_speed_mm_s": 4.0},
                {"forward_distance_mm": 4.5, "mean_speed_mm_s": 3.8},
                {"forward_distance_mm": 5.5, "mean_speed_mm_s": 4.2},
            ],
        }

        analyzer = StatisticalAnalyzer(trials_by_condition=trials_by_condition)
        comparisons = analyzer.compare_metric("forward_distance_mm", metric_display_name="Forward Dist (mm)")

        assert len(comparisons) == 1
        comp = comparisons[0]
        assert comp.condition_name == "DNa01 Left"
        assert comp.metric_name == "Forward Dist (mm)"
        assert comp.baseline_mean == 21.0
        assert comp.lesion_mean == 5.0
        assert comp.percentage_change < -70.0  # Significant drop
        assert comp.cohens_d < -10.0  # Massive negative effect size
        assert comp.p_value < 0.01
        assert comp.significance_stars in ["**", "***"]

    def test_markdown_and_latex_and_json_export(self, tmp_path) -> None:
        comp = StatisticalComparison(
            condition_name="DNa01_Left",
            metric_name="Forward Distance (mm)",
            baseline_mean=21.3,
            baseline_std=0.8,
            lesion_mean=5.1,
            lesion_std=0.5,
            percentage_change=-76.1,
            cohens_d=-24.1,
            t_statistic=-30.5,
            p_value=0.00004,
            significance_stars="***",
        )
        analyzer = StatisticalAnalyzer(trials_by_condition={})

        md = analyzer.generate_markdown_table([comp])
        assert "| Experimental Condition |" in md
        assert "DNa01_Left" in md
        assert "-76.1%" in md
        assert "***" in md

        tex = analyzer.generate_latex_table([comp])
        assert r"\begin{table}" in tex
        assert r"\bottomrule" in tex
        assert "DNa01\\_Left" in tex

        md_p, tex_p, json_p = analyzer.export_all([comp], out_dir=tmp_path)
        assert md_p.exists()
        assert tex_p.exists()
        assert json_p.exists()

        with open(json_p, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["condition_name"] == "DNa01_Left"


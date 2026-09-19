from connectome_rl.src.analysis.kinematics import (
    GaitMetrics,
    KinematicAnalyzer,
    StepBout,
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
from connectome_rl.src.analysis.benchmark import (
    ModelBenchmarkResult,
    compute_energy_and_cot,
    evaluate_policy_locomotion,
    plot_model_comparison,
    save_benchmark_results,
)
from connectome_rl.src.analysis.battery import (
    ConditionSummaryStats,
    LesionBatteryTrial,
    SystematicLesionBattery,
    plot_lesion_battery_results,
    save_battery_results,
)
from connectome_rl.src.analysis.statistics import (
    StatisticalAnalyzer,
    StatisticalComparison,
    compute_cohens_d,
    compute_welch_t_test,
    format_significance_stars,
)

__all__ = [
    "LesionConfig",
    "LesionController",
    "LesionResult",
    "GaitMetrics",
    "KinematicAnalyzer",
    "StepBout",
    "LEG_NAMES",
    "TRIPOD_1_INDICES",
    "TRIPOD_2_INDICES",
    "compose_side_by_side_video",
    "draw_hud_banner",
    "plot_connectome_lesion_summary",
    "plot_publication_dashboard",
    "ModelBenchmarkResult",
    "compute_energy_and_cot",
    "evaluate_policy_locomotion",
    "plot_model_comparison",
    "save_benchmark_results",
    "ConditionSummaryStats",
    "LesionBatteryTrial",
    "SystematicLesionBattery",
    "plot_lesion_battery_results",
    "save_battery_results",
    "StatisticalAnalyzer",
    "StatisticalComparison",
    "compute_cohens_d",
    "compute_welch_t_test",
    "format_significance_stars",
]



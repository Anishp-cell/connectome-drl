"""Run Statistical Publication Analysis for Drosophila Connectomics.

Loads the multi-seed systematic lesion battery results and generates formal
scientific publication tables:
  1. Percentage change (% relative to intact control baseline).
  2. Effect size (Cohen's d).
  3. Welch's unequal variance two-sample t-test (t-stat, p-value, significance stars).
  4. Camera-ready exports:
     - outputs/statistical_significance_table.md (GitHub Markdown)
     - outputs/statistical_significance_table.tex (Publication LaTeX booktabs)
     - outputs/statistical_summary.json (Structured JSON)

Usage:
  python scripts/run_statistical_analysis.py [--battery-json outputs/systematic_lesion_battery.json]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from connectome_rl.src.analysis.statistics import (
    StatisticalAnalyzer,
    StatisticalComparison,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("StatisticalAnalysis")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate statistical significance tables for Connectome lesion experiments."
    )
    parser.add_argument(
        "--battery-json",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "systematic_lesion_battery.json"),
        help="Path to systematic lesion battery trial results JSON.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs"),
        help="Directory to save statistical tables and JSON.",
    )
    args = parser.parse_args()

    json_path = Path(args.battery_json)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not json_path.exists():
        logger.error("Battery JSON not found at %s. Please run scripts/run_systematic_lesion_battery.py first.", json_path)
        sys.exit(1)

    logger.info("Loading lesion battery trial data from %s...", json_path)
    analyzer = StatisticalAnalyzer.from_json(json_path)

    metrics_to_test = [
        ("forward_distance_mm", "Forward Distance (mm)"),
        ("mean_speed_mm_s", "Mean Speed (mm/s)"),
        ("net_yaw_deg", "Net Yaw Turning (deg)"),
        ("tripod_coordination_index", "Tripod Coordination Index (TCI)"),
        ("cost_of_transport", "Cost of Transport (CoT)"),
    ]

    all_comparisons: list[StatisticalComparison] = []

    for metric_key, display_name in metrics_to_test:
        logger.info("Performing Welch's t-test and Cohen's d for: %s", display_name)
        comps = analyzer.compare_metric(metric_key=metric_key, metric_display_name=display_name)
        all_comparisons.extend(comps)

    # Export all tables (Markdown, LaTeX booktabs, JSON)
    md_path, tex_path, json_out_path = analyzer.export_all(all_comparisons, out_dir=out_dir)

    print("\n" + "=" * 90)
    print("DROSOPHILA CONNECTOME LESION BATTERY: STATISTICAL SIGNIFICANCE SUMMARY")
    print("=" * 90)
    print(analyzer.generate_markdown_table(all_comparisons))
    print("\nSignificance stars: *** p < 0.001 | ** p < 0.01 | * p < 0.05 | ns (not significant)")
    print("=" * 90 + "\n")

    logger.info("Saved camera-ready Markdown table to: %s", md_path)
    logger.info("Saved publication LaTeX table to:     %s", tex_path)
    logger.info("Saved structured summary JSON to:       %s", json_out_path)


if __name__ == "__main__":
    main()

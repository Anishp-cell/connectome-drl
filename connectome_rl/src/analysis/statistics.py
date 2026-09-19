"""Statistical Significance and Publication Tables Suite for Drosophila Connectomics.

Performs formal statistical hypothesis testing on in-silico ablation batteries:
  1. Percentage changes (% drop / % increase relative to intact baseline).
  2. Effect sizes (Cohen's d: quantifying biological impact magnitude).
  3. Hypothesis testing (Welch's two-sample t-test with unequal variances).
  4. Significance threshold mapping (*** p < 0.001, ** p < 0.01, * p < 0.05, ns).
  5. Multi-format camera-ready export:
     - outputs/statistical_significance_table.md (GitHub Markdown)
     - outputs/statistical_significance_table.tex (Publication LaTeX booktabs)
     - outputs/statistical_summary.json (Structured JSON)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy import stats


@dataclass
class StatisticalComparison:
    """Quantitative statistical comparison between an experimental condition and baseline."""

    condition_name: str
    metric_name: str
    baseline_mean: float
    baseline_std: float
    lesion_mean: float
    lesion_std: float
    percentage_change: float
    cohens_d: float
    t_statistic: float
    p_value: float
    significance_stars: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def format_significance_stars(p_value: float) -> str:
    """Convert a p-value into standard scientific significance star notation."""
    if np.isnan(p_value):
        return "ns"
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return "ns"


def compute_cohens_d(group1: Sequence[float], group2: Sequence[float]) -> float:
    """Compute Cohen's d effect size between two independent groups.

    d = (mean1 - mean2) / s_pooled

    Args:
        group1: Experimental sample values.
        group2: Baseline sample values.

    Returns:
        Cohen's d effect size (positive indicates group1 > group2).
    """
    g1 = np.asarray(group1, dtype=np.float64)
    g2 = np.asarray(group2, dtype=np.float64)

    n1, n2 = len(g1), len(g2)
    if n1 < 1 or n2 < 1:
        return 0.0

    mean1, mean2 = float(np.mean(g1)), float(np.mean(g2))
    var1 = float(np.var(g1, ddof=1)) if n1 > 1 else 0.0
    var2 = float(np.var(g2, ddof=1)) if n2 > 1 else 0.0

    # Pooled standard deviation
    dof = (n1 - 1) + (n2 - 1)
    if dof > 0:
        s_pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / dof)
    else:
        s_pooled = 0.0

    if s_pooled < 1e-9:
        # Fallback if both groups have zero variance
        diff = mean1 - mean2
        if abs(diff) < 1e-9:
            return 0.0
        return float(np.sign(diff) * 10.0)  # Large effect if completely separated

    return float((mean1 - mean2) / s_pooled)


def compute_welch_t_test(group1: Sequence[float], group2: Sequence[float]) -> tuple[float, float]:
    """Execute Welch's two-sample t-test (two-tailed, unequal variance).

    Args:
        group1: Experimental sample values.
        group2: Baseline control sample values.

    Returns:
        (t_statistic, p_value): Test statistic and two-tailed probability.
    """
    g1 = np.asarray(group1, dtype=np.float64)
    g2 = np.asarray(group2, dtype=np.float64)

    if len(g1) < 2 or len(g2) < 2:
        # Not enough samples for a valid two-sample t-test
        diff = float(np.mean(g1) - np.mean(g2)) if len(g1) > 0 and len(g2) > 0 else 0.0
        return 0.0, 1.0

    # Check for zero variance
    if np.var(g1) < 1e-12 and np.var(g2) < 1e-12:
        if np.isclose(np.mean(g1), np.mean(g2)):
            return 0.0, 1.0
        else:
            # Deterministic separation
            return float(np.sign(np.mean(g1) - np.mean(g2)) * 100.0), 0.0001

    res = stats.ttest_ind(g1, g2, equal_var=False)
    t_stat = float(res.statistic) if not np.isnan(res.statistic) else 0.0
    p_val = float(res.pvalue) if not np.isnan(res.pvalue) else 1.0
    return t_stat, p_val


class StatisticalAnalyzer:
    """Scientific statistical analysis engine for comparing experimental lesion groups."""

    def __init__(self, trials_by_condition: dict[str, list[dict[str, Any]]]) -> None:
        """Initialize with trials grouped by condition name.

        Args:
            trials_by_condition: Dict mapping condition name to list of trial dicts.
        """
        self.trials = trials_by_condition
        # Identify baseline control condition
        self.baseline_key = "Intact Control"
        for k in self.trials:
            if "intact" in k.lower() or "control" in k.lower() and "random" not in k.lower():
                self.baseline_key = k
                break

    @classmethod
    def from_json(cls, json_path: str | Path) -> StatisticalAnalyzer:
        """Instantiate analyzer from a saved systematic lesion battery JSON file."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        raw_trials = data.get("trials", [])
        grouped: dict[str, list[dict[str, Any]]] = {}
        for t in raw_trials:
            c = t["condition_name"]
            grouped.setdefault(c, []).append(t)

        return cls(trials_by_condition=grouped)

    def compare_metric(
        self,
        metric_key: str,
        metric_display_name: str | None = None,
    ) -> list[StatisticalComparison]:
        """Perform statistical hypothesis testing across all conditions against baseline."""
        disp_name = metric_display_name if metric_display_name is not None else metric_key
        baseline_trials = self.trials.get(self.baseline_key, [])
        baseline_vals = [float(t.get(metric_key, 0.0)) for t in baseline_trials]

        base_mean = float(np.mean(baseline_vals)) if baseline_vals else 0.0
        base_std = float(np.std(baseline_vals, ddof=1)) if len(baseline_vals) > 1 else 0.0

        comparisons: list[StatisticalComparison] = []

        for cond_name, trials in self.trials.items():
            if cond_name == self.baseline_key:
                continue

            vals = [float(t.get(metric_key, 0.0)) for t in trials]
            m_mean = float(np.mean(vals)) if vals else 0.0
            m_std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

            # Percentage change relative to baseline
            if abs(base_mean) > 1e-6:
                pct_change = ((m_mean - base_mean) / abs(base_mean)) * 100.0
            else:
                pct_change = 0.0

            d = compute_cohens_d(vals, baseline_vals)
            t_stat, p_val = compute_welch_t_test(vals, baseline_vals)
            stars = format_significance_stars(p_val)

            comparisons.append(
                StatisticalComparison(
                    condition_name=cond_name,
                    metric_name=disp_name,
                    baseline_mean=base_mean,
                    baseline_std=base_std,
                    lesion_mean=m_mean,
                    lesion_std=m_std,
                    percentage_change=float(pct_change),
                    cohens_d=float(d),
                    t_statistic=float(t_stat),
                    p_value=float(p_val),
                    significance_stars=stars,
                )
            )

        return comparisons

    def generate_markdown_table(self, comparisons: Sequence[StatisticalComparison]) -> str:
        """Generate a GitHub-flavored Markdown publication table."""
        lines = [
            "| Experimental Condition | Metric | Baseline (Intact) | Lesion (Ablated) | Δ (%) | Cohen's d | t-stat | p-value | Sig. |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        for c in comparisons:
            b_str = f"{c.baseline_mean:+.2f} ± {c.baseline_std:.2f}"
            l_str = f"{c.lesion_mean:+.2f} ± {c.lesion_std:.2f}"
            pct_str = f"{c.percentage_change:+.1f}%"
            d_str = f"{c.cohens_d:+.2f}"
            t_str = f"{c.t_statistic:+.2f}"
            p_str = f"{c.p_value:.4f}" if c.p_value >= 0.0001 else "<0.0001"
            lines.append(
                f"| **{c.condition_name}** | {c.metric_name} | {b_str} | {l_str} | {pct_str} | {d_str} | {t_str} | {p_str} | {c.significance_stars} |"
            )
        return "\n".join(lines)

    def generate_latex_table(
        self,
        comparisons: Sequence[StatisticalComparison],
        caption: str = "Statistical significance and effect sizes across in-silico lesion battery conditions relative to intact baseline.",
        label: str = "tab:lesion_significance",
    ) -> str:
        """Generate a publication-grade LaTeX booktabs table."""
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\begin{tabular}{llcccccc}",
            r"\toprule",
            r"\textbf{Condition} & \textbf{Metric} & \textbf{Intact Control} & \textbf{Lesioned Fly} & \textbf{$\Delta$ (\%)} & \textbf{Cohen's $d$} & \textbf{$p$-value} & \textbf{Sig.} \\",
            r"\midrule",
        ]

        for c in comparisons:
            clean_name = c.condition_name.replace("_", r"\_")
            b_str = f"{c.baseline_mean:+.2f} $\\pm$ {c.baseline_std:.2f}"
            l_str = f"{c.lesion_mean:+.2f} $\\pm$ {c.lesion_std:.2f}"
            pct_str = f"{c.percentage_change:+.1f}\\%"
            d_str = f"{c.cohens_d:+.2f}"
            p_str = f"{c.p_value:.4f}" if c.p_value >= 0.0001 else r"$<0.0001$"
            lines.append(
                rf"{clean_name} & {c.metric_name} & {b_str} & {l_str} & {pct_str} & {d_str} & {p_str} & \textbf{{{c.significance_stars}}} \\"
            )

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])
        return "\n".join(lines)

    def export_all(
        self,
        comparisons: Sequence[StatisticalComparison],
        out_dir: str | Path = "outputs",
    ) -> tuple[Path, Path, Path]:
        """Export Markdown, LaTeX, and JSON summary tables to disk."""
        p_dir = Path(out_dir)
        p_dir.mkdir(parents=True, exist_ok=True)

        # 1. Markdown table
        md_file = p_dir / "statistical_significance_table.md"
        with open(md_file, "w", encoding="utf-8") as f:
            f.write("# Drosophila Connectomics: In-Silico Lesion Statistical Significance\n\n")
            f.write(self.generate_markdown_table(comparisons))
            f.write("\n\n*Significance: `***` p < 0.001 | `**` p < 0.01 | `*` p < 0.05 | `ns` not significant.*\n")

        # 2. LaTeX table
        tex_file = p_dir / "statistical_significance_table.tex"
        with open(tex_file, "w", encoding="utf-8") as f:
            f.write(self.generate_latex_table(comparisons) + "\n")

        # 3. JSON data
        json_file = p_dir / "statistical_summary.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump([c.to_dict() for c in comparisons], f, indent=2)

        return md_file, tex_file, json_file

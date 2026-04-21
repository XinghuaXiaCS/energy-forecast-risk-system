from __future__ import annotations

from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_plots(scored_df: pd.DataFrame, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    plot_dir = output_dir / "plots"
    _ensure_dir(plot_dir)

    heat = scored_df.pivot_table(index="region", columns="risk_band", values="risk_score", aggfunc="count", fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(heat.values)
    ax.set_xticks(range(len(heat.columns)))
    ax.set_xticklabels(heat.columns)
    ax.set_yticks(range(len(heat.index)))
    ax.set_yticklabels(heat.index)
    ax.set_title("Risk band counts by region")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            ax.text(j, i, str(int(heat.iloc[i, j])), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(plot_dir / "risk_heatmap.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    scored_df["confidence_level"].value_counts().reindex(["high", "medium", "low"]).fillna(0).plot(kind="bar", ax=ax)
    ax.set_title("Confidence distribution")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(plot_dir / "confidence_distribution.png", dpi=150)
    plt.close(fig)

    reason_series = scored_df["reason_codes"].str.split("|").explode().value_counts().head(8)
    fig, ax = plt.subplots(figsize=(8, 4))
    reason_series.sort_values().plot(kind="barh", ax=ax)
    ax.set_title("Root cause breakdown")
    ax.set_xlabel("Count")
    fig.tight_layout()
    fig.savefig(plot_dir / "root_cause_breakdown.png", dpi=150)
    plt.close(fig)


def write_monitoring_summary(summary: dict, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    _ensure_dir(output_dir)
    with open(output_dir / "monitoring_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def write_markdown_report(scored_df: pd.DataFrame, summary: dict, output_dir: str | Path, top_n: int = 10) -> None:
    output_dir = Path(output_dir)
    _ensure_dir(output_dir)
    top = scored_df.sort_values(["expected_impact_nzd", "risk_score"], ascending=False).head(top_n)
    lines = [
        "# Forecast Risk Daily Report",
        "",
        "## Summary",
        f"- Rows scored: {len(scored_df)}",
        f"- Critical alerts: {(scored_df['risk_band'] == 'Critical').sum()}",
        f"- High alerts: {(scored_df['risk_band'] == 'High').sum()}",
        f"- Medium alerts: {(scored_df['risk_band'] == 'Medium').sum()}",
        "",
        "## Monitoring",
        f"- AUC: {summary['model']['auc']:.3f}",
        f"- PR-AUC: {summary['model']['pr_auc']:.3f}",
        f"- Brier: {summary['model']['brier']:.3f}",
        f"- Severity RMSE: {summary['model']['severity_rmse']:.2f}",
        f"- Trigger retraining: {summary['lifecycle']['trigger_retraining']}",
        "",
        "## Top flagged rows",
        "",
        "| forecast_date | region | horizon | risk_band | risk_score | predicted_severity | expected_impact_nzd | reason_codes | recommended_action |",
        "|---|---|---:|---|---:|---:|---:|---|---|",
    ]
    for _, row in top.iterrows():
        lines.append(
            f"| {row['forecast_date'].date()} | {row['region']} | {int(row['horizon_hours'])} | {row['risk_band']} | {row['risk_score']:.3f} | {row['predicted_severity']:.1f} | {row['expected_impact_nzd']:.0f} | {row['reason_codes']} | {row['recommended_action']} |"
        )
    lines.extend([
        "",
        "## Plots",
        "",
        "- `plots/risk_heatmap.png`",
        "- `plots/confidence_distribution.png`",
        "- `plots/root_cause_breakdown.png`",
    ])
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")

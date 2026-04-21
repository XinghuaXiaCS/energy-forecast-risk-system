from __future__ import annotations

from pathlib import Path
import pandas as pd

from yes_forecast_risk.data.warehouse import read_table
from yes_forecast_risk.models.inference import load_bundle, score_rows
from yes_forecast_risk.monitoring.drift import compute_drift
from yes_forecast_risk.monitoring.metrics import model_metrics, business_metrics, data_metrics, lifecycle_flags
from yes_forecast_risk.monitoring.reports import save_plots, write_markdown_report, write_monitoring_summary
from yes_forecast_risk.decision.policy import apply_decision_policy
from yes_forecast_risk.features.engineering import split_train_valid_test


def run_batch(cfg: dict, output_dir: str | Path | None = None) -> Path:
    output_dir = Path(output_dir or cfg["report_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_mart = read_table(cfg["warehouse_path"], "feature_mart")
    feature_mart["forecast_date"] = pd.to_datetime(feature_mart["forecast_date"])
    _, _, test_df = split_train_valid_test(feature_mart)
    bundle = load_bundle(cfg["model_dir"])
    scored = score_rows(test_df, bundle)

    train_df, _, _ = split_train_valid_test(feature_mart)
    drift_summary = compute_drift(train_df, scored)
    max_psi = max(v["psi"] for v in drift_summary.values()) if drift_summary else 0.0
    decided = apply_decision_policy(scored, cfg, max_feature_psi=max_psi)

    model_summary = model_metrics(decided)
    business_summary = business_metrics(decided)
    data_summary = data_metrics(decided)
    lifecycle = lifecycle_flags(model_summary, drift_summary, cfg)
    summary = {
        "model": model_summary,
        "business": business_summary,
        "data": data_summary,
        "drift": drift_summary,
        "lifecycle": lifecycle,
    }

    # Outputs
    daily_cols = [
        "forecast_date", "region", "horizon_hours", "risk_score", "predicted_severity", "predicted_risk_type",
        "expected_impact_nzd", "risk_band", "recommended_action", "escalation_priority",
        "confidence_level", "reason_codes", "reason_messages", "suggested_owner", "workflow_step", "decision_mode"
    ]
    alert_bands = cfg["batch"].get("alert_queue_bands", ["Medium", "High", "Critical"])
    decided.sort_values(["forecast_date", "expected_impact_nzd"], ascending=[True, False]).to_csv(
        output_dir / "daily_risk_table.csv", index=False
    )
    decided[decided["risk_band"].isin(alert_bands)].sort_values(
        ["expected_impact_nzd", "risk_score"], ascending=False
    ).to_csv(output_dir / "alert_queue.csv", index=False)
    decided[daily_cols].sort_values(["expected_impact_nzd", "risk_score"], ascending=False).to_csv(
        output_dir / "decision_log.csv", index=False
    )
    save_plots(decided, output_dir)
    write_monitoring_summary(summary, output_dir)
    write_markdown_report(decided, summary, output_dir, top_n=cfg["batch"]["report_top_n"])
    return output_dir

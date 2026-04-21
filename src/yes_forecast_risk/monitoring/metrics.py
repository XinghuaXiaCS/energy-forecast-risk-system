from __future__ import annotations

from typing import Dict
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, mean_absolute_error, mean_squared_error


def model_metrics(df: pd.DataFrame) -> Dict[str, float]:
    y_true = df["effective_label"].astype(int)
    y_prob = df["risk_score"].astype(float)
    sev_true = df["target_severity"].astype(float)
    sev_pred = df["predicted_severity"].astype(float)
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "severity_rmse": float(np.sqrt(mean_squared_error(sev_true, sev_pred))),
        "severity_mae": float(mean_absolute_error(sev_true, sev_pred)),
    }


def business_metrics(df: pd.DataFrame) -> Dict[str, float]:
    df = df.copy()
    df["alerted"] = df["risk_band"].isin(["High", "Critical"]).astype(int)
    true_high = df["effective_label"].astype(int)
    alerted = df["alerted"].astype(int)
    tp = int(((true_high == 1) & (alerted == 1)).sum())
    fp = int(((true_high == 0) & (alerted == 1)).sum())
    escalated_days = max(int(alerted.sum()), 1)
    return {
        "alert_precision_high_plus": float(tp / max(tp + fp, 1)),
        "false_alert_burden": float(fp / max(len(df), 1)),
        "escalated_hit_rate": float(tp / escalated_days),
        "analyst_review_load": int((df["risk_band"] == "Medium").sum() + alerted.sum()),
        "critical_count": int((df["risk_band"] == "Critical").sum()),
    }


def data_metrics(df: pd.DataFrame) -> Dict[str, float]:
    return {
        "missingness_mean": float(df["missingness_pct"].mean()),
        "latency_p95": float(df["data_latency_min"].quantile(0.95)),
        "source_health_mean": float(df["source_health"].mean()),
        "low_source_health_rate": float((df["source_health"] < 0.82).mean()),
    }


def lifecycle_flags(metric_summary: Dict[str, float], drift_summary: Dict[str, Dict[str, float]], cfg: dict) -> Dict[str, bool]:
    max_psi = max(v["psi"] for v in drift_summary.values()) if drift_summary else 0.0
    return {
        "trigger_retraining": bool(
            metric_summary.get("auc", 1.0) < cfg["retraining"]["auc_floor"]
            or metric_summary.get("pr_auc", 1.0) < cfg["retraining"]["pr_auc_floor"]
            or max_psi > cfg["retraining"]["max_population_stability_index"]
        ),
        "review_now": bool(max_psi > 0.18 or metric_summary.get("brier", 0.0) > 0.20),
    }

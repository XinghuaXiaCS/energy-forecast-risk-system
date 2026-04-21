from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from yes_forecast_risk.explain.reason_codes import build_reason_codes


REGION_OWNER = {
    "CNI": "Forecast Operations Lead",
    "UNI": "Market Analytics Analyst",
    "LNI": "Forecast Operations Lead",
    "LSI": "Regional Forecast Analyst",
    "USI": "Regional Forecast Analyst",
}

# Default fallback weights used only if the config does not supply them. Kept here as
# a transparent fallback so the module still runs if an older config is loaded.
_DEFAULT_FALLBACK_WEIGHTS = {
    "model_disagreement": 0.24,
    "weather_uncertainty": 0.18,
    "load_ramp": 0.16,
    "input_drift_score": 0.14,
    "analog_miss_rate": 0.12,
    "forecast_spread": 0.08,
    "missingness_pct": 0.08,
}

_DEFAULT_CONFIDENCE = {
    "high": {"model_agreement": 0.80, "data_health": 0.88},
    "medium": {"model_agreement": 0.62, "data_health": 0.74},
}


def fallback_score(row: pd.Series, cfg: Dict[str, Any] | None = None) -> float:
    """Rule-based risk score used when input data is too degraded to trust the model."""
    weights = (cfg or {}).get("fallback_weights", _DEFAULT_FALLBACK_WEIGHTS)
    components = []
    for feat, w in weights.items():
        val = float(row.get(feat, 0.0))
        if feat == "missingness_pct":
            # Missingness is on a narrow [0, 0.35] scale operationally; scale to [0, 1].
            val = min(val * 4, 1.0)
        else:
            val = min(max(val, 0.0), 1.0)
        components.append(w * val)
    return max(0.0, min(sum(components), 1.0))


def use_fallback(row: pd.Series, cfg: Dict[str, Any], max_feature_psi: float | None = None) -> bool:
    fb = cfg["fallback"]
    min_source_health = fb.get("min_source_health", 0.75)
    return (
        row.get("missingness_pct", 0) > fb["max_missingness_pct"]
        or row.get("data_latency_min", 0) > fb["max_latency_minutes"]
        or row.get("input_drift_score", 0) > fb["max_input_drift_score"]
        or row.get("source_health", 1) < min_source_health
        or (max_feature_psi is not None and max_feature_psi > fb["max_feature_psi"])
    )


def impact_weighted_priority(prob: float, severity: float, exposure_nzd: float) -> float:
    return float(prob * severity * max(exposure_nzd, 1) / 100.0)


def confidence_level(row: pd.Series, cfg: Dict[str, Any] | None = None) -> str:
    thresholds = (cfg or {}).get("confidence_thresholds", _DEFAULT_CONFIDENCE)
    agreement = float(row.get("model_agreement", 0.5))
    data_health = float(row.get("data_health_score", 0.75))
    high = thresholds["high"]
    medium = thresholds["medium"]
    if agreement >= high["model_agreement"] and data_health >= high["data_health"]:
        return "high"
    if agreement >= medium["model_agreement"] and data_health >= medium["data_health"]:
        return "medium"
    return "low"


def assign_band(prob: float, impact_nzd: float, cfg: Dict[str, Any]) -> str:
    composite = 0.65 * prob + 0.35 * min(impact_nzd / cfg["impact_thresholds_nzd"]["critical"], 1.0)
    if composite >= cfg["risk_thresholds"]["critical"] or impact_nzd >= cfg["impact_thresholds_nzd"]["critical"]:
        return "Critical"
    if composite >= cfg["risk_thresholds"]["high"] or impact_nzd >= cfg["impact_thresholds_nzd"]["high"]:
        return "High"
    if composite >= cfg["risk_thresholds"]["medium"] or impact_nzd >= cfg["impact_thresholds_nzd"]["medium"]:
        return "Medium"
    return "Low"


def action_bundle(row: pd.Series, band: str) -> Dict[str, str]:
    owner = REGION_OWNER.get(row.get("region"), "Forecast Analyst")
    if band == "Critical":
        return {
            "recommended_action": "Open cross-functional ops review and validate same-day forecast assumptions.",
            "escalation_priority": "P1",
            "suggested_owner": "Forecast Operations Lead",
            "workflow_step": "critical_incident_review",
        }
    if band == "High":
        return {
            "recommended_action": "Escalate for analyst and forecast-ops review before market close.",
            "escalation_priority": "P2",
            "suggested_owner": owner,
            "workflow_step": "escalate_to_ops_lead",
        }
    if band == "Medium":
        return {
            "recommended_action": "Queue analyst review and confirm drivers before noon.",
            "escalation_priority": "P3",
            "suggested_owner": "Market Analytics Analyst",
            "workflow_step": "analyst_review_queue",
        }
    return {
        "recommended_action": "Passive monitoring only; no immediate intervention.",
        "escalation_priority": "P4",
        "suggested_owner": "Monitoring Only",
        "workflow_step": "monitor_only",
    }


def apply_decision_policy(
    df: pd.DataFrame, cfg: Dict[str, Any], max_feature_psi: float | None = None
) -> pd.DataFrame:
    rows = []
    multipliers = cfg.get("region_exposure_multiplier", {})
    for _, row in df.iterrows():
        row = row.copy()
        if use_fallback(row, cfg, max_feature_psi=max_feature_psi):
            risk_score = fallback_score(row, cfg)
            decision_mode = "fallback_rule"
        else:
            risk_score = float(row.get("risk_score", 0.0))
            decision_mode = "selected_model"
        severity = float(row.get("predicted_severity", 0.0))
        exposure = float(row.get("exposure_nzd", 0.0)) * float(multipliers.get(row.get("region"), 1.0))
        expected_impact_nzd = impact_weighted_priority(risk_score, severity, exposure)
        band = assign_band(risk_score, expected_impact_nzd, cfg)
        reason_codes, reason_messages = build_reason_codes(row)
        bundle = action_bundle(row, band)
        row["risk_score"] = round(risk_score, 5)
        row["expected_impact_nzd"] = round(expected_impact_nzd, 2)
        row["risk_band"] = band
        row["confidence_level"] = confidence_level(row, cfg)
        row["decision_mode"] = decision_mode
        row["reason_codes"] = "|".join(reason_codes)
        row["reason_messages"] = "|".join(reason_messages)
        for k, v in bundle.items():
            row[k] = v
        rows.append(row)
    return pd.DataFrame(rows)

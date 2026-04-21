import pandas as pd
from yes_forecast_risk.decision.policy import apply_decision_policy
from yes_forecast_risk.config import load_config


def test_policy_uses_fallback_when_data_is_bad():
    cfg = load_config("configs/base.yaml")
    df = pd.DataFrame([{
        "forecast_date": "2025-06-01",
        "region": "CNI",
        "horizon_hours": 24,
        "risk_score": 0.91,
        "predicted_severity": 60.0,
        "weather_uncertainty": 0.2,
        "load_ramp": 0.3,
        "model_disagreement": 0.4,
        "calendar_anomaly": 0.1,
        "holiday_flag": 0,
        "input_drift_score": 0.9,
        "analog_miss_rate": 0.3,
        "missingness_pct": 0.2,
        "data_latency_min": 150,
        "prior_abs_error": 5.0,
        "forecast_spread": 0.2,
        "price_spike_proxy": 0.2,
        "exposure_nzd": 22000,
        "source_health": 0.6,
        "data_health_score": 0.5,
        "model_agreement": 0.2,
    }])
    out = apply_decision_policy(df, cfg)
    assert out.iloc[0]["decision_mode"] == "fallback_rule"
    assert out.iloc[0]["risk_band"] in {"Medium", "High", "Critical"}

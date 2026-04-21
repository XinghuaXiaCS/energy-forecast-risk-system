import pandas as pd
from yes_forecast_risk.explain.reason_codes import build_reason_codes


def test_reason_codes_include_expected_flags():
    row = pd.Series({
        "weather_uncertainty": 0.9,
        "load_ramp": 0.8,
        "model_disagreement": 0.85,
        "calendar_anomaly": 0.05,
        "holiday_flag": 0,
        "input_drift_score": 0.7,
        "analog_miss_rate": 0.8,
        "forecast_spread": 0.72,
        "prior_abs_error": 28,
        "data_latency_min": 90,
        "source_health": 0.7,
        "missingness_pct": 0.02,
    })
    codes, messages = build_reason_codes(row)
    assert "RC01" in codes
    assert "RC02" in codes or "RC03" in codes
    assert len(messages) >= 1

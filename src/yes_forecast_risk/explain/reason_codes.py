from __future__ import annotations

from typing import List, Tuple
import pandas as pd

CODE_MAP = {
    "RC01": "high weather uncertainty",
    "RC02": "sharp ramp or regime shift expected",
    "RC03": "model disagreement spike",
    "RC04": "calendar anomaly or holiday distortion",
    "RC05": "input feature drift or source instability",
    "RC06": "historical analogues show elevated miss risk",
    "RC07": "wide forecast spread / uncertainty band",
    "RC08": "large prior miss magnitude persists",
    "RC09": "data freshness / latency risk",
    "RC10": "low source health",
}


def build_reason_codes(row: pd.Series) -> Tuple[List[str], List[str]]:
    codes: List[str] = []
    if row.get("weather_uncertainty", 0) >= 0.75:
        codes.append("RC01")
    if row.get("load_ramp", 0) >= 0.75:
        codes.append("RC02")
    if row.get("model_disagreement", 0) >= 0.72:
        codes.append("RC03")
    if row.get("calendar_anomaly", 0) >= 0.60 or row.get("holiday_flag", 0) == 1:
        codes.append("RC04")
    if row.get("input_drift_score", 0) >= 0.65 or row.get("missingness_pct", 0) >= 0.08:
        codes.append("RC05")
    if row.get("analog_miss_rate", 0) >= 0.65:
        codes.append("RC06")
    if row.get("forecast_spread", 0) >= 0.70:
        codes.append("RC07")
    if row.get("prior_abs_error", 0) >= 24:
        codes.append("RC08")
    if row.get("data_latency_min", 0) >= 75:
        codes.append("RC09")
    if row.get("source_health", 1) <= 0.82:
        codes.append("RC10")
    if not codes:
        codes = ["RC06"] if row.get("analog_miss_rate", 0) > 0.45 else ["RC07"]
    messages = [CODE_MAP[c] for c in codes[:4]]
    return codes[:4], messages

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

REGIONS = ["CNI", "UNI", "LNI", "LSI", "USI"]
RISK_TYPES = ["weather", "ramp", "model_disagreement", "calendar", "data_quality"]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_synthetic_raw(output_csv: str | Path, feedback_csv: str | Path, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", "2025-06-30", freq="D")
    rows = []
    region_exposure = {"CNI": 1.20, "UNI": 1.00, "LNI": 1.10, "LSI": 0.95, "USI": 0.90}

    for d in dates:
        seasonal = 0.45 + 0.12 * np.sin(2 * np.pi * d.dayofyear / 365.25)
        winter = 1 if d.month in {6, 7, 8} else 0
        weekend = 1 if d.dayofweek >= 5 else 0
        holidayish = 1 if (d.month == 12 and d.day > 20) or (d.month == 1 and d.day < 10) else 0

        for region in REGIONS:
            region_vol = {"CNI": 0.12, "UNI": 0.09, "LNI": 0.10, "LSI": 0.07, "USI": 0.08}[region]
            for horizon in (24, 48):
                horizon_penalty = 0.10 if horizon == 48 else 0.0
                weather_uncertainty = np.clip(seasonal + region_vol + 0.15 * winter + rng.normal(0, 0.12), 0, 1.4)
                load_ramp = np.clip(0.3 + 0.45 * winter + 0.25 * (1 - weekend) + rng.normal(0, 0.15), 0, 1.5)
                model_disagreement = np.clip(0.25 + 0.65 * weather_uncertainty + 0.2 * horizon_penalty + rng.normal(0, 0.14), 0, 1.8)
                calendar_anomaly = np.clip(0.08 + 0.85 * holidayish + 0.25 * weekend + rng.normal(0, 0.08), 0, 1.4)
                input_drift_score = np.clip(0.12 + 0.35 * holidayish + 0.28 * rng.beta(2, 6), 0, 1.3)
                analog_miss_rate = np.clip(0.18 + 0.55 * model_disagreement + rng.normal(0, 0.12), 0, 1.5)
                missingness_pct = np.clip(0.01 + 0.05 * rng.beta(1.5, 15) + 0.04 * holidayish, 0, 0.35)
                data_latency_min = np.clip(rng.normal(16 + 18 * holidayish, 9), 0, 220)
                prior_abs_error = np.clip(rng.normal(12 + 18 * weather_uncertainty + 9 * horizon_penalty, 6), 0, 120)
                forecast_spread = np.clip(0.18 + 0.75 * weather_uncertainty + 0.3 * horizon_penalty + rng.normal(0, 0.10), 0, 1.8)
                price_spike_proxy = np.clip(0.12 + 0.55 * load_ramp + 0.45 * weather_uncertainty + rng.normal(0, 0.10), 0, 1.8)
                exposure_nzd = max(3000, rng.normal(18000 * region_exposure[region], 4500))
                source_health = np.clip(0.98 - missingness_pct - max(data_latency_min - 60, 0) / 400 + rng.normal(0, 0.03), 0.4, 1.0)

                latent = (
                    1.30 * weather_uncertainty +
                    1.15 * load_ramp +
                    1.35 * model_disagreement +
                    0.80 * calendar_anomaly +
                    1.00 * input_drift_score +
                    0.75 * analog_miss_rate +
                    0.65 * forecast_spread +
                    0.40 * price_spike_proxy +
                    0.70 * (1 - source_health) +
                    0.35 * horizon_penalty +
                    rng.normal(0, 0.35)
                )
                p_high = _sigmoid(latent - 4.2)
                target_high_risk = int(rng.random() < p_high)

                severity = np.clip(
                    6
                    + 11 * weather_uncertainty
                    + 9 * load_ramp
                    + 14 * model_disagreement
                    + 7 * input_drift_score
                    + 5 * forecast_spread
                    + 4 * horizon_penalty
                    + 18 * target_high_risk
                    + rng.normal(0, 5.5),
                    1,
                    100,
                )
                actual_abs_error = np.clip(severity * rng.normal(0.85, 0.12), 0, 120)

                drivers = {
                    "weather": weather_uncertainty + 0.3 * forecast_spread,
                    "ramp": load_ramp + 0.2 * price_spike_proxy,
                    "model_disagreement": model_disagreement + 0.15 * analog_miss_rate,
                    "calendar": calendar_anomaly + 0.1 * holidayish,
                    "data_quality": input_drift_score + missingness_pct + data_latency_min / 180,
                }
                target_risk_type = max(drivers, key=drivers.get)

                rows.append(
                    {
                        "forecast_date": d.strftime("%Y-%m-%d"),
                        "region": region,
                        "horizon_hours": horizon,
                        "weather_uncertainty": round(float(weather_uncertainty), 5),
                        "load_ramp": round(float(load_ramp), 5),
                        "model_disagreement": round(float(model_disagreement), 5),
                        "calendar_anomaly": round(float(calendar_anomaly), 5),
                        "holiday_flag": int(holidayish),
                        "input_drift_score": round(float(input_drift_score), 5),
                        "analog_miss_rate": round(float(analog_miss_rate), 5),
                        "missingness_pct": round(float(missingness_pct), 5),
                        "data_latency_min": round(float(data_latency_min), 5),
                        "prior_abs_error": round(float(prior_abs_error), 5),
                        "forecast_spread": round(float(forecast_spread), 5),
                        "price_spike_proxy": round(float(price_spike_proxy), 5),
                        "exposure_nzd": round(float(exposure_nzd), 2),
                        "source_health": round(float(source_health), 5),
                        "actual_abs_error": round(float(actual_abs_error), 3),
                        "target_high_risk": target_high_risk,
                        "target_severity": round(float(severity), 3),
                        "target_risk_type": target_risk_type,
                        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
                    }
                )

    df = pd.DataFrame(rows).sort_values(["forecast_date", "region", "horizon_hours"]).reset_index(drop=True)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    # seed a tiny initial feedback file to demonstrate HITL integration
    seed_feedback = df.sample(12, random_state=seed)[["forecast_date", "region", "horizon_hours"]].copy()
    seed_feedback["reviewer"] = rng.choice(["analyst_a", "analyst_b"], size=len(seed_feedback))
    seed_feedback["reviewed_label"] = rng.integers(0, 2, size=len(seed_feedback))
    seed_feedback["override_reason"] = rng.choice(
        [
            "post_event_review_confirmed",
            "false_alert_due_to_data_delay",
            "missed_weather_shift",
            "analyst_override_based_on_domain_context",
        ],
        size=len(seed_feedback),
    )
    seed_feedback["review_status"] = rng.choice(["true_positive", "false_positive", "needs_relabel"], size=len(seed_feedback))
    seed_feedback["reviewed_at"] = pd.Timestamp.now(tz="UTC").isoformat()
    feedback_csv = Path(feedback_csv)
    feedback_csv.parent.mkdir(parents=True, exist_ok=True)
    seed_feedback.to_csv(feedback_csv, index=False)
    return df

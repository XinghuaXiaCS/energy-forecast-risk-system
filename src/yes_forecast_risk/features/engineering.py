from __future__ import annotations

from typing import List, Tuple
import pandas as pd
import numpy as np

FEATURE_COLUMNS = [
    "region_code",
    "horizon_hours",
    "weekday",
    "month",
    "weather_uncertainty",
    "load_ramp",
    "model_disagreement",
    "calendar_anomaly",
    "holiday_flag",
    "input_drift_score",
    "analog_miss_rate",
    "missingness_pct",
    "data_latency_min",
    "prior_abs_error",
    "forecast_spread",
    "price_spike_proxy",
    "exposure_nzd",
    "source_health",
    "data_health_score",
]

REGION_CODE = {"CNI": 0, "UNI": 1, "LNI": 2, "LSI": 3, "USI": 4}


def prepare_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["forecast_date"] = pd.to_datetime(out["forecast_date"])
    out["region_code"] = out["region"].map(REGION_CODE).fillna(-1).astype(int)
    out["weekday"] = out.get("weekday", out["forecast_date"].dt.dayofweek)
    out["month"] = out.get("month", out["forecast_date"].dt.month)
    if "data_health_score" not in out.columns:
        out["data_health_score"] = (1 - out["missingness_pct"].clip(0, 1)) * out["source_health"].clip(0, 1)
    out["weekday"] = out["weekday"].astype(int)
    out["month"] = out["month"].astype(int)
    for col in FEATURE_COLUMNS:
        if col not in out.columns:
            out[col] = 0.0
    return out


def split_train_valid_test(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["forecast_date"] = pd.to_datetime(df["forecast_date"])
    df = df.sort_values("forecast_date").reset_index(drop=True)
    train = df[df["forecast_date"] < pd.Timestamp("2024-09-01")].copy()
    valid = df[(df["forecast_date"] >= pd.Timestamp("2024-09-01")) & (df["forecast_date"] < pd.Timestamp("2025-02-01"))].copy()
    test = df[df["forecast_date"] >= pd.Timestamp("2025-02-01")].copy()
    return train, valid, test


def make_model_matrices(df: pd.DataFrame):
    f = prepare_feature_frame(df)
    X = f[FEATURE_COLUMNS].astype(float)
    return X, f

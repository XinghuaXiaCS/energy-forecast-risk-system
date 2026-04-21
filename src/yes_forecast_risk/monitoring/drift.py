from __future__ import annotations

from typing import Dict, Iterable
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


DRIFT_FEATURES = [
    "weather_uncertainty",
    "load_ramp",
    "model_disagreement",
    "input_drift_score",
    "missingness_pct",
    "data_latency_min",
    "forecast_spread",
]


def psi(expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
    expected = pd.Series(expected).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    actual = pd.Series(actual).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if expected.empty or actual.empty:
        return 0.0
    breakpoints = np.quantile(expected, q=np.linspace(0, 1, buckets + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    expected_bins = pd.cut(expected, bins=breakpoints, include_lowest=True)
    actual_bins = pd.cut(actual, bins=breakpoints, include_lowest=True)
    e = expected_bins.value_counts(normalize=True, sort=False).replace(0, 1e-6)
    a = actual_bins.value_counts(normalize=True, sort=False).replace(0, 1e-6)
    return float(((a - e) * np.log(a / e)).sum())


def compute_drift(train_df: pd.DataFrame, score_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    out = {}
    for col in DRIFT_FEATURES:
        out[col] = {
            "psi": psi(train_df[col], score_df[col]),
            "ks_stat": float(ks_2samp(train_df[col], score_df[col]).statistic),
        }
    return out

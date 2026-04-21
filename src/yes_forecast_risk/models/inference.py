from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from yes_forecast_risk.features.engineering import make_model_matrices


def load_bundle(model_dir: str | Path):
    return joblib.load(Path(model_dir) / "model_bundle.joblib")


def score_rows(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    X, feat = make_model_matrices(df)
    clf = bundle["classifier"]
    reg = bundle["regressor"]
    rt_model = bundle["risk_type_model"]
    baseline = bundle["baseline_classifier"]
    inv_map = bundle["risk_type_map"]

    prob = clf.predict_proba(X)[:, 1]
    sev = reg.predict(X)
    rt_pred = rt_model.predict(X)
    baseline_prob = baseline.predict_proba(X)[:, 1]

    candidate_probs = []
    for model in bundle.get("classifier_candidates", {}).values():
        candidate_probs.append(model.predict_proba(X)[:, 1])
    if len(candidate_probs) >= 2:
        # std of probabilities in [0,1] is bounded by 0.5 for two-point masses;
        # multiplying by 2 maps full-disagreement to 0 and full-agreement to 1.
        agreement = 1 - 2 * np.std(np.vstack(candidate_probs), axis=0)
    else:
        agreement = np.full(len(X), 1.0)

    out = feat.copy()
    out["risk_score"] = prob
    out["predicted_severity"] = np.clip(sev, 0, 100)
    out["predicted_risk_type"] = pd.Series(rt_pred).map(inv_map).fillna("weather")
    out["baseline_risk_score"] = baseline_prob
    out["model_agreement"] = np.clip(agreement, 0, 1)
    return out

from __future__ import annotations

import os
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, HTTPException

from yes_forecast_risk.config import load_config
from yes_forecast_risk.schemas import ScoreRequest, ScoreResponse
from yes_forecast_risk.models.inference import load_bundle, score_rows
from yes_forecast_risk.decision.policy import apply_decision_policy

app = FastAPI(title="Forecast Risk Decision Support API", version="0.1.0")

CFG_PATH = os.getenv("YES_RISK_CONFIG", "configs/base.yaml")
cfg = load_config(CFG_PATH)


def _load_bundle_or_none():
    model_dir = Path(cfg["model_dir"])
    if not (model_dir / "model_bundle.joblib").exists():
        return None
    return load_bundle(model_dir)


@app.get("/health")
def health():
    bundle = _load_bundle_or_none()
    return {"status": "ok", "model_loaded": bundle is not None}


@app.post("/score", response_model=ScoreResponse)
def score(payload: ScoreRequest):
    bundle = _load_bundle_or_none()
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model bundle not found. Run the training/demo pipeline first.")
    df = pd.DataFrame([payload.model_dump()])
    scored = score_rows(df, bundle)
    decided = apply_decision_policy(scored, cfg)
    row = decided.iloc[0]
    return ScoreResponse(
        risk_score=float(row["risk_score"]),
        risk_band=str(row["risk_band"]),
        predicted_severity=float(row["predicted_severity"]),
        predicted_risk_type=str(row["predicted_risk_type"]),
        expected_impact_nzd=float(row["expected_impact_nzd"]),
        recommended_action=str(row["recommended_action"]),
        escalation_priority=str(row["escalation_priority"]),
        confidence_level=str(row["confidence_level"]),
        suggested_owner=str(row["suggested_owner"]),
        workflow_step=str(row["workflow_step"]),
        reason_codes=str(row["reason_codes"]).split("|"),
        reason_messages=str(row["reason_messages"]).split("|"),
        decision_mode=str(row["decision_mode"]),
    )

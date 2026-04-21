from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class ScoreRequest(BaseModel):
    forecast_date: str
    region: str = Field(pattern="^(CNI|UNI|LNI|LSI|USI)$")
    horizon_hours: int = Field(ge=1, le=168)
    weather_uncertainty: float = Field(ge=0)
    load_ramp: float = Field(ge=0)
    model_disagreement: float = Field(ge=0)
    calendar_anomaly: float = Field(ge=0)
    holiday_flag: int = Field(ge=0, le=1)
    input_drift_score: float = Field(ge=0)
    analog_miss_rate: float = Field(ge=0)
    missingness_pct: float = Field(ge=0, le=1)
    data_latency_min: float = Field(ge=0)
    prior_abs_error: float = Field(ge=0)
    forecast_spread: float = Field(ge=0)
    price_spike_proxy: float = Field(ge=0)
    exposure_nzd: float = Field(ge=0)
    source_health: float = Field(ge=0, le=1)


class ScoreResponse(BaseModel):
    risk_score: float
    risk_band: str
    predicted_severity: float
    predicted_risk_type: str
    expected_impact_nzd: float
    recommended_action: str
    escalation_priority: str
    confidence_level: str
    suggested_owner: str
    workflow_step: str
    reason_codes: List[str]
    reason_messages: List[str]
    decision_mode: str

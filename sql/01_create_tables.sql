DROP TABLE IF EXISTS raw_forecast_risk;
DROP TABLE IF EXISTS feedback_events;

CREATE TABLE raw_forecast_risk (
    forecast_date TEXT,
    region TEXT,
    horizon_hours INTEGER,
    weather_uncertainty REAL,
    load_ramp REAL,
    model_disagreement REAL,
    calendar_anomaly REAL,
    holiday_flag INTEGER,
    input_drift_score REAL,
    analog_miss_rate REAL,
    missingness_pct REAL,
    data_latency_min REAL,
    prior_abs_error REAL,
    forecast_spread REAL,
    price_spike_proxy REAL,
    exposure_nzd REAL,
    source_health REAL,
    actual_abs_error REAL,
    target_high_risk INTEGER,
    target_severity REAL,
    target_risk_type TEXT,
    created_at TEXT
);

CREATE TABLE feedback_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    forecast_date TEXT,
    region TEXT,
    horizon_hours INTEGER,
    reviewer TEXT,
    reviewed_label INTEGER,
    override_reason TEXT,
    review_status TEXT,
    reviewed_at TEXT
);

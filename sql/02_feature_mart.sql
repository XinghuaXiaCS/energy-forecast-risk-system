DROP TABLE IF EXISTS feature_mart;

CREATE TABLE feature_mart AS
SELECT
    r.forecast_date,
    r.region,
    r.horizon_hours,
    CAST(strftime('%w', r.forecast_date) AS INTEGER) AS weekday,
    CAST(strftime('%m', r.forecast_date) AS INTEGER) AS month,
    r.weather_uncertainty,
    r.load_ramp,
    r.model_disagreement,
    r.calendar_anomaly,
    r.holiday_flag,
    r.input_drift_score,
    r.analog_miss_rate,
    r.missingness_pct,
    r.data_latency_min,
    r.prior_abs_error,
    r.forecast_spread,
    r.price_spike_proxy,
    r.exposure_nzd,
    r.source_health,
    (1.0 - r.missingness_pct) * r.source_health AS data_health_score,
    r.actual_abs_error,
    r.target_high_risk,
    r.target_severity,
    r.target_risk_type,
    COALESCE(f.reviewed_label, r.target_high_risk) AS effective_label,
    f.override_reason,
    f.review_status
FROM raw_forecast_risk r
LEFT JOIN (
    SELECT forecast_date, region, horizon_hours, reviewed_label, override_reason, review_status
    FROM feedback_events
) f
ON r.forecast_date = f.forecast_date
AND r.region = f.region
AND r.horizon_hours = f.horizon_hours;

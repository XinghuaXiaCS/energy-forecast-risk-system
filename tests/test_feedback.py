from pathlib import Path
from yes_forecast_risk.data.synthetic import generate_synthetic_raw
from yes_forecast_risk.data.warehouse import initialise_warehouse, build_feature_mart, read_table
from yes_forecast_risk.workflow.feedback import record_feedback


def test_feedback_event_is_appended(tmp_path: Path):
    raw = tmp_path / "raw.csv"
    feedback = tmp_path / "feedback.csv"
    db = tmp_path / "demo.sqlite"
    generate_synthetic_raw(raw, feedback, seed=1)
    initialise_warehouse(db, raw, feedback, "sql/01_create_tables.sql")
    build_feature_mart(db, "sql/02_feature_mart.sql")
    before = len(read_table(db, "feedback_events"))
    record_feedback(str(db), "2025-01-02", "CNI", 24, "analyst_x", 1, "manual_review", "true_positive")
    after = len(read_table(db, "feedback_events"))
    assert after == before + 1

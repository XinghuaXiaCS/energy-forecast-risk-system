from pathlib import Path
from yes_forecast_risk.data.synthetic import generate_synthetic_raw
from yes_forecast_risk.data.warehouse import initialise_warehouse, build_feature_mart, read_table
from yes_forecast_risk.models.train import train_all_models
from yes_forecast_risk.workflow.batch import run_batch
from yes_forecast_risk.config import load_config


def test_end_to_end_demo_components(tmp_path: Path):
    cfg = load_config("configs/base.yaml")
    cfg["raw_csv_path"] = str(tmp_path / "raw.csv")
    cfg["feedback_csv_path"] = str(tmp_path / "feedback.csv")
    cfg["warehouse_path"] = str(tmp_path / "demo.sqlite")
    cfg["model_dir"] = str(tmp_path / "models")
    cfg["report_dir"] = str(tmp_path / "report")

    generate_synthetic_raw(cfg["raw_csv_path"], cfg["feedback_csv_path"], seed=7)
    initialise_warehouse(cfg["warehouse_path"], cfg["raw_csv_path"], cfg["feedback_csv_path"], "sql/01_create_tables.sql")
    build_feature_mart(cfg["warehouse_path"], "sql/02_feature_mart.sql")
    feature_mart = read_table(cfg["warehouse_path"], "feature_mart")
    metrics = train_all_models(feature_mart, cfg["model_dir"])
    assert metrics["test"]["classification"]["auc"] > 0.60
    out_dir = run_batch(cfg, cfg["report_dir"])
    assert (out_dir / "daily_risk_table.csv").exists()
    assert (out_dir / "monitoring_summary.json").exists()

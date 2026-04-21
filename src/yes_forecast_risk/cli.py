from __future__ import annotations

import argparse
from pathlib import Path
import json

from yes_forecast_risk.config import load_config
from yes_forecast_risk.data.synthetic import generate_synthetic_raw
from yes_forecast_risk.data.warehouse import initialise_warehouse, build_feature_mart, read_table
from yes_forecast_risk.models.train import train_all_models
from yes_forecast_risk.workflow.batch import run_batch
from yes_forecast_risk.workflow.feedback import record_feedback


def cmd_bootstrap(args):
    cfg = load_config(args.config)
    generate_synthetic_raw(cfg["raw_csv_path"], cfg["feedback_csv_path"], seed=cfg.get("seed", 42))
    print(f"Synthetic raw data written to {cfg['raw_csv_path']}")


def cmd_build_warehouse(args):
    cfg = load_config(args.config)
    initialise_warehouse(
        cfg["warehouse_path"],
        cfg["raw_csv_path"],
        cfg["feedback_csv_path"],
        "sql/01_create_tables.sql",
    )
    build_feature_mart(cfg["warehouse_path"], "sql/02_feature_mart.sql")
    print(f"Warehouse initialised at {cfg['warehouse_path']}")


def cmd_train(args):
    cfg = load_config(args.config)
    feature_mart = read_table(cfg["warehouse_path"], "feature_mart")
    metrics = train_all_models(feature_mart, cfg["model_dir"])
    print(json.dumps(metrics, indent=2))


def cmd_score_batch(args):
    cfg = load_config(args.config)
    out = run_batch(cfg, args.output_dir or cfg["report_dir"])
    print(f"Batch outputs written to {out}")


def cmd_feedback(args):
    cfg = load_config(args.config)
    record_feedback(
        cfg["warehouse_path"],
        forecast_date=args.forecast_date,
        region=args.region,
        horizon_hours=args.horizon_hours,
        reviewer=args.reviewer,
        reviewed_label=args.reviewed_label,
        override_reason=args.override_reason,
        review_status=args.review_status,
    )
    build_feature_mart(cfg["warehouse_path"], "sql/02_feature_mart.sql")
    print("Feedback recorded and feature mart rebuilt.")


def cmd_run_demo(args):
    cfg = load_config(args.config)
    generate_synthetic_raw(cfg["raw_csv_path"], cfg["feedback_csv_path"], seed=cfg.get("seed", 42))
    initialise_warehouse(
        cfg["warehouse_path"],
        cfg["raw_csv_path"],
        cfg["feedback_csv_path"],
        "sql/01_create_tables.sql",
    )
    build_feature_mart(cfg["warehouse_path"], "sql/02_feature_mart.sql")
    feature_mart = read_table(cfg["warehouse_path"], "feature_mart")
    train_all_models(feature_mart, cfg["model_dir"])
    out = run_batch(cfg, cfg["report_dir"])
    print(f"Demo finished. Artefacts written to {out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Forecast Risk Decision Support System")
    parser.add_argument("--config", default="configs/base.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("bootstrap-data")
    p.set_defaults(func=cmd_bootstrap)

    p = sub.add_parser("build-warehouse")
    p.set_defaults(func=cmd_build_warehouse)

    p = sub.add_parser("train")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("score-batch")
    p.add_argument("--output-dir", default=None)
    p.set_defaults(func=cmd_score_batch)

    p = sub.add_parser("add-feedback")
    p.add_argument("forecast_date")
    p.add_argument("region")
    p.add_argument("horizon_hours", type=int)
    p.add_argument("reviewer")
    p.add_argument("reviewed_label", type=int)
    p.add_argument("override_reason")
    p.add_argument("review_status")
    p.set_defaults(func=cmd_feedback)

    p = sub.add_parser("run-demo")
    p.set_defaults(func=cmd_run_demo)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

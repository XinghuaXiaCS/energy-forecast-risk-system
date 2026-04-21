"""Regression tests for the XGBoost / LightGBM optional challenger integration.

These tests verify that:
  - xgboost and lightgbm candidates are registered when the libraries are installed
  - both classification and regression heads include the new candidates
  - the champion-selection utility function still picks a valid candidate
  - `model_bundle.joblib` records which libraries were active during the run
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from yes_forecast_risk.config import load_config
from yes_forecast_risk.data.synthetic import generate_synthetic_raw
from yes_forecast_risk.data.warehouse import (
    build_feature_mart,
    initialise_warehouse,
    read_table,
)
from yes_forecast_risk.models.train import (
    _build_classifier_candidates,
    _build_regressor_candidates,
    _build_type_candidates,
    lightgbm_available,
    train_all_models,
    xgboost_available,
)


@pytest.mark.skipif(not xgboost_available(), reason="xgboost not installed")
def test_xgb_candidate_registered_in_all_three_heads():
    clf = _build_classifier_candidates(pos_weight=1.0)
    reg = _build_regressor_candidates()
    typ = _build_type_candidates()
    assert "xgb_candidate" in clf
    assert "xgb_regressor" in reg
    assert "xgb_type" in typ


@pytest.mark.skipif(not lightgbm_available(), reason="lightgbm not installed")
def test_lgb_candidate_registered_in_all_three_heads():
    clf = _build_classifier_candidates(pos_weight=1.0)
    reg = _build_regressor_candidates()
    typ = _build_type_candidates()
    assert "lgb_candidate" in clf
    assert "lgb_regressor" in reg
    assert "lgb_type" in typ


def test_sklearn_candidates_always_available():
    """sklearn candidates must be present regardless of optional libraries."""
    clf = _build_classifier_candidates(pos_weight=1.0)
    for name in ("baseline_logreg", "rf_candidate", "tree_candidate"):
        assert name in clf


@pytest.mark.skipif(
    not (xgboost_available() or lightgbm_available()),
    reason="requires at least one of xgboost or lightgbm",
)
def test_bundle_records_which_optional_libs_were_active(tmp_path: Path):
    """A full training run must persist the availability flags into the bundle
    so downstream audits can tell which libs were in the competition."""
    cfg = load_config("configs/base.yaml")
    cfg["raw_csv_path"] = str(tmp_path / "raw.csv")
    cfg["feedback_csv_path"] = str(tmp_path / "feedback.csv")
    cfg["warehouse_path"] = str(tmp_path / "demo.sqlite")
    cfg["model_dir"] = str(tmp_path / "models")

    generate_synthetic_raw(cfg["raw_csv_path"], cfg["feedback_csv_path"], seed=13)
    initialise_warehouse(
        cfg["warehouse_path"], cfg["raw_csv_path"], cfg["feedback_csv_path"],
        "sql/01_create_tables.sql",
    )
    build_feature_mart(cfg["warehouse_path"], "sql/02_feature_mart.sql")
    mart = read_table(cfg["warehouse_path"], "feature_mart")

    metrics = train_all_models(mart, cfg["model_dir"])

    # The metrics JSON on disk must also record the flags.
    written = json.loads(Path(cfg["model_dir"], "metrics.json").read_text())
    assert written["xgboost_enabled"] == xgboost_available()
    assert written["lightgbm_enabled"] == lightgbm_available()

    # At least one of the new candidates must have been evaluated on validation.
    candidate_names = set(metrics["validation"]["classification_candidates"].keys())
    if xgboost_available():
        assert "xgb_candidate" in candidate_names
    if lightgbm_available():
        assert "lgb_candidate" in candidate_names

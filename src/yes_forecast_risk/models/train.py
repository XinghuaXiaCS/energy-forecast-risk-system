from __future__ import annotations

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

from yes_forecast_risk.features.engineering import FEATURE_COLUMNS, make_model_matrices, split_train_valid_test


# --- Optional-library gating -----------------------------------------------------
# XGBoost and LightGBM are added as *optional* challengers. If either library is
# missing, that candidate set is silently skipped and the pipeline continues with
# the always-available sklearn models.

try:
    from xgboost import XGBClassifier, XGBRegressor
    _XGB_OK = True
except ImportError:  # pragma: no cover
    _XGB_OK = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    _LGB_OK = True
except ImportError:  # pragma: no cover
    _LGB_OK = False


def xgboost_available() -> bool:
    return _XGB_OK


def lightgbm_available() -> bool:
    return _LGB_OK


RISK_TYPE_MAP = {"weather": 0, "ramp": 1, "model_disagreement": 2, "calendar": 3, "data_quality": 4}
INV_RISK_TYPE_MAP = {v: k for k, v in RISK_TYPE_MAP.items()}


def _classification_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "positive_rate": float(y_pred.mean()),
        "precision_at_threshold": float(((y_pred == 1) & (y_true == 1)).sum() / max((y_pred == 1).sum(), 1)),
        "recall_at_threshold": float(((y_pred == 1) & (y_true == 1)).sum() / max((y_true == 1).sum(), 1)),
    }


def _regression_metrics(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def _multiclass_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }


def _utility(metrics):
    """Composite selection score: rewards AUC and PR-AUC, penalises miscalibration."""
    return metrics["auc"] + 0.6 * metrics["pr_auc"] - 0.4 * metrics["brier"]


def _feature_importance_dict(model) -> dict:
    if hasattr(model, "feature_importances_"):
        return dict(zip(FEATURE_COLUMNS, model.feature_importances_.tolist()))
    if hasattr(model, "coef_"):
        coef = model.coef_
        vec = coef[0] if coef.ndim > 1 else coef
        return dict(zip(FEATURE_COLUMNS, [float(v) for v in vec]))
    return {}


# --- Candidate factories ---------------------------------------------------------
# Each factory returns a dict of {name: fresh_estimator}. Factories let the test
# path dependency-inject alternative sets if needed.

def _build_classifier_candidates(pos_weight: float) -> dict:
    cands = {
        "baseline_logreg": LogisticRegression(max_iter=250, solver="liblinear", class_weight="balanced"),
        "rf_candidate": RandomForestClassifier(
            n_estimators=60, random_state=42, min_samples_leaf=8,
            class_weight="balanced_subsample", n_jobs=-1,
        ),
        "tree_candidate": DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=42),
    }
    if _XGB_OK:
        cands["xgb_candidate"] = XGBClassifier(
            n_estimators=80, max_depth=4, learning_rate=0.08, subsample=0.9,
            colsample_bytree=0.9, reg_lambda=1.0, random_state=42,
            scale_pos_weight=pos_weight, eval_metric="logloss",
            tree_method="hist", n_jobs=-1, verbosity=0,
        )
    if _LGB_OK:
        cands["lgb_candidate"] = LGBMClassifier(
            n_estimators=120, max_depth=-1, num_leaves=31, learning_rate=0.06,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            class_weight="balanced", random_state=42, n_jobs=-1, verbosity=-1,
        )
    return cands


def _build_regressor_candidates() -> dict:
    cands = {
        "baseline_ridge": Ridge(alpha=1.0),
        "rf_regressor": RandomForestRegressor(n_estimators=60, random_state=42, min_samples_leaf=6, n_jobs=-1),
        "tree_regressor": DecisionTreeRegressor(max_depth=6, min_samples_leaf=20, random_state=42),
    }
    if _XGB_OK:
        cands["xgb_regressor"] = XGBRegressor(
            n_estimators=80, max_depth=4, learning_rate=0.08, subsample=0.9,
            colsample_bytree=0.9, reg_lambda=1.0, random_state=42,
            tree_method="hist", n_jobs=-1, verbosity=0,
        )
    if _LGB_OK:
        cands["lgb_regressor"] = LGBMRegressor(
            n_estimators=120, num_leaves=31, learning_rate=0.06, subsample=0.9,
            colsample_bytree=0.9, reg_lambda=1.0, random_state=42, n_jobs=-1,
            verbosity=-1,
        )
    return cands


def _build_type_candidates() -> dict:
    cands = {
        "baseline_shallow_rf": RandomForestClassifier(n_estimators=60, max_depth=4, random_state=42, n_jobs=-1),
        "rf_type": RandomForestClassifier(n_estimators=60, random_state=42, min_samples_leaf=5, n_jobs=-1),
        "tree_type": DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=42),
    }
    if _XGB_OK:
        cands["xgb_type"] = XGBClassifier(
            n_estimators=80, max_depth=4, learning_rate=0.08, subsample=0.9,
            colsample_bytree=0.9, reg_lambda=1.0, random_state=42,
            objective="multi:softprob", num_class=len(RISK_TYPE_MAP),
            eval_metric="mlogloss", tree_method="hist", n_jobs=-1, verbosity=0,
        )
    if _LGB_OK:
        cands["lgb_type"] = LGBMClassifier(
            n_estimators=120, num_leaves=31, learning_rate=0.06, subsample=0.9,
            colsample_bytree=0.9, reg_lambda=1.0, random_state=42,
            objective="multiclass", num_class=len(RISK_TYPE_MAP),
            n_jobs=-1, verbosity=-1,
        )
    return cands


def train_all_models(feature_mart: pd.DataFrame, model_dir: str | Path) -> dict:
    """Train baseline / challenger classifiers, severity regressor, and risk-type
    classifier. Each head picks the highest-utility candidate (respecting guard-
    rails), then the promoted classifier is isotonic-calibrated on the validation
    set so downstream `risk_score` values are calibrated probabilities.

    XGBoost and LightGBM are added as *optional* challengers: if either library
    is not importable, that candidate is silently skipped — sklearn candidates
    always remain as a safety net.
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    train_df, valid_df, test_df = split_train_valid_test(feature_mart)
    X_train, train_f = make_model_matrices(train_df)
    X_valid, valid_f = make_model_matrices(valid_df)
    X_test, test_f = make_model_matrices(test_df)

    y_train = train_f["effective_label"].astype(int).values
    y_valid = valid_f["effective_label"].astype(int).values
    y_test = test_f["effective_label"].astype(int).values

    sev_train = train_f["target_severity"].astype(float).values
    sev_valid = valid_f["target_severity"].astype(float).values
    sev_test = test_f["target_severity"].astype(float).values

    rt_train = train_f["target_risk_type"].map(RISK_TYPE_MAP).astype(int).values
    rt_valid = valid_f["target_risk_type"].map(RISK_TYPE_MAP).astype(int).values
    rt_test = test_f["target_risk_type"].map(RISK_TYPE_MAP).astype(int).values

    # --- Classification candidates ---------------------------------------------------
    # Positive/negative balance for XGBoost's scale_pos_weight.
    n_pos = max(int(y_train.sum()), 1)
    n_neg = max(len(y_train) - n_pos, 1)
    pos_weight = float(n_neg / n_pos)

    clf_candidates = _build_classifier_candidates(pos_weight)
    for name, m in clf_candidates.items():
        try:
            m.fit(X_train, y_train)
        except Exception as err:  # pragma: no cover
            warnings.warn(f"Classifier '{name}' failed to fit: {err}", RuntimeWarning)

    clf_scores = {
        name: _classification_metrics(y_valid, m.predict_proba(X_valid)[:, 1])
        for name, m in clf_candidates.items()
    }
    champion_name = max(clf_scores, key=lambda k: _utility(clf_scores[k]))
    champion_raw = clf_candidates[champion_name]

    # Isotonic calibration of the promoted champion using the validation set.
    # sklearn >= 1.6 requires wrapping the fitted estimator in FrozenEstimator; older
    # versions accept `cv="prefit"`. We support both for portability.
    try:
        from sklearn.frozen import FrozenEstimator  # sklearn >= 1.6
        calibrated_champion = CalibratedClassifierCV(
            FrozenEstimator(champion_raw), method="isotonic"
        ).fit(X_valid, y_valid)
    except ImportError:
        calibrated_champion = CalibratedClassifierCV(
            estimator=champion_raw, method="isotonic", cv="prefit"
        ).fit(X_valid, y_valid)

    # --- Severity regressor ----------------------------------------------------------
    reg_candidates = _build_regressor_candidates()
    for name, m in reg_candidates.items():
        try:
            m.fit(X_train, sev_train)
        except Exception as err:  # pragma: no cover
            warnings.warn(f"Regressor '{name}' failed to fit: {err}", RuntimeWarning)

    reg_scores = {name: _regression_metrics(sev_valid, m.predict(X_valid)) for name, m in reg_candidates.items()}
    selected_reg_name = min(reg_scores, key=lambda k: reg_scores[k]["rmse"])
    selected_regressor = reg_candidates[selected_reg_name]

    # --- Risk-type multiclass classifier ---------------------------------------------
    type_candidates = _build_type_candidates()
    for name, m in type_candidates.items():
        try:
            m.fit(X_train, rt_train)
        except Exception as err:  # pragma: no cover
            warnings.warn(f"Risk-type model '{name}' failed to fit: {err}", RuntimeWarning)

    type_scores = {name: _multiclass_metrics(rt_valid, m.predict(X_valid)) for name, m in type_candidates.items()}
    selected_type_name = max(type_scores, key=lambda k: type_scores[k]["f1_macro"])
    selected_type_model = type_candidates[selected_type_name]

    # --- Bundle ----------------------------------------------------------------------
    bundle = {
        "feature_columns": FEATURE_COLUMNS,
        "classifier": calibrated_champion,
        "classifier_name": f"{champion_name}+isotonic",
        "classifier_uncalibrated": champion_raw,
        "classifier_candidates": clf_candidates,
        "regressor": selected_regressor,
        "regressor_name": selected_reg_name,
        "risk_type_model": selected_type_model,
        "risk_type_name": selected_type_name,
        "baseline_classifier": clf_candidates["baseline_logreg"],
        "baseline_regressor": reg_candidates["baseline_ridge"],
        "baseline_type_model": type_candidates["baseline_shallow_rf"],
        "risk_type_map": INV_RISK_TYPE_MAP,
        "xgboost_enabled": _XGB_OK,
        "lightgbm_enabled": _LGB_OK,
    }
    joblib.dump(bundle, model_dir / "model_bundle.joblib")

    # --- Holdout metrics -------------------------------------------------------------
    test_prob = calibrated_champion.predict_proba(X_test)[:, 1]
    test_prob_raw = champion_raw.predict_proba(X_test)[:, 1]
    test_sev = selected_regressor.predict(X_test)
    test_rt = selected_type_model.predict(X_test)

    metrics = {
        "validation": {
            "classification_candidates": clf_scores,
            "severity_candidates": reg_scores,
            "risk_type_candidates": type_scores,
            "selected_classifier": champion_name,
            "selected_regressor": selected_reg_name,
            "selected_risk_type_model": selected_type_name,
        },
        "test": {
            "classification_calibrated": _classification_metrics(y_test, test_prob),
            "classification_uncalibrated": _classification_metrics(y_test, test_prob_raw),
            "severity": _regression_metrics(sev_test, test_sev),
            "risk_type": _multiclass_metrics(rt_test, test_rt),
        },
        "train_rows": int(len(train_f)),
        "valid_rows": int(len(valid_f)),
        "test_rows": int(len(test_f)),
        "xgboost_enabled": _XGB_OK,
        "lightgbm_enabled": _LGB_OK,
    }
    # Keep a legacy-compatible "classification" key so existing readers / tests still work.
    metrics["test"]["classification"] = metrics["test"]["classification_calibrated"]

    with open(model_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Persist feature-importance for all three model heads, not only the classifier.
    importances = {
        "classifier": _feature_importance_dict(champion_raw),
        "regressor": _feature_importance_dict(selected_regressor),
        "risk_type": _feature_importance_dict(selected_type_model),
    }
    with open(model_dir / "feature_importance.json", "w", encoding="utf-8") as f:
        json.dump(importances, f, indent=2)

    return metrics

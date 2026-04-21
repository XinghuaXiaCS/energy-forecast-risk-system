# Model Card

## Product
Forecast Risk Decision Support System for power market workflows.

## Intended use
Prioritize forecast-risk review and escalation for analysts and forecast operations.

## Not intended for
Automated replacement of human forecasters without review.

## Outputs
- high-risk probability
- predicted severity
- predicted risk type
- expected impact
- risk band
- recommended action
- reason codes

## Primary models
- binary high-risk classifier
- severity regressor
- risk-type classifier
- interpretable logistic/ridge baseline
- rule-based fallback policy

## Candidate registry per head
Each model head (classifier / regressor / risk-type) trains the following
candidates on every run and promotes the one that maximises the utility
composite (classifier: `AUC + 0.6·PR-AUC − 0.4·Brier`; regressor: min RMSE;
risk-type: max macro-F1).

- **Always on**: `baseline_logreg` / `baseline_ridge` / `baseline_shallow_rf`,
  `rf_candidate` / `rf_regressor` / `rf_type`,
  `tree_candidate` / `tree_regressor` / `tree_type`
- **Optional (library-gated)**:
  `xgb_candidate` / `xgb_regressor` / `xgb_type` — when `xgboost` is installed
  `lgb_candidate` / `lgb_regressor` / `lgb_type` — when `lightgbm` is installed
- **Calibration**: the promoted classifier is isotonic-calibrated on the
  validation window before being exposed as `risk_score`.

## Key assumptions
- daily forecast-risk patterns can be summarized with engineered signals
- action thresholds are jointly defined with business stakeholders
- human review remains in the loop for elevated-risk items

## Monitoring requirements
- AUC / PR-AUC / Brier
- RMSE / MAE on severity
- PSI / KS on critical features
- alert precision and false-alert burden by risk band

## Key risks
- over-alerting during unusual regimes
- degraded source data inflating false positives
- threshold drift over time
- reviewer label inconsistency

## Mitigations
- calibrated probabilities
- fallback mode
- champion–challenger governance
- review cadence
- reason-code auditability

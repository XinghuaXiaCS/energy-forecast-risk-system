# Forecast Risk Decision Support System

A production-oriented GitHub project that upgrades a **Forecast Risk Early Warning System** into a fuller **operational forecast risk decision support product** for power market workflows.

This repository is designed to reflect the kind of evidence a senior applied data scientist would show for a role that emphasizes:

- advanced analytics and machine learning
- interpretable, repeatable, scalable models
- end-to-end lifecycle ownership
- deployment support and ongoing monitoring
- business-critical decision support
- workflow integration with cross-functional squads

## What this project adds beyond a plain risk classifier

This repo expands a binary forecast risk model into a fuller operating system for forecast-risk triage:

1. **Business action layer**
   - risk score
   - risk band (`Low`, `Medium`, `High`, `Critical`)
   - recommended action
   - escalation priority
   - suggested owner and workflow step
   - confidence level
   - reason codes

2. **Multi-layer risk targets**
   - high-risk probability
   - expected severity
   - predicted risk type
   - 24h and 48h outlook support

3. **Expected impact / value layer**
   - expected impact in NZD
   - priority score = probability × severity × exposure

4. **Explainability layer**
   - business-readable reason codes instead of only SHAP-style plots

5. **Champion–challenger governance**
   - a single promoted **champion** (isotonic-calibrated) used for production scoring
   - **challenger** candidates trained every run and scored in shadow
   - interpretable **baseline** as a safety-net reference
   - **rule-based fallback** for degraded data conditions
   - explicit promotion and rollback criteria (see [`PROMOTION_POLICY.md`](docs/PROMOTION_POLICY.md))

6. **Monitoring and maintenance**
   - data quality checks
   - drift metrics (PSI / KS)
   - model metrics (AUC / PR-AUC / Brier / RMSE)
   - business metrics (alert precision, false-alert burden, escalated hit rate)
   - retraining triggers and review cadence

7. **Workflow integration and feedback loop**
   - batch risk report
   - analyst review queue
   - feedback capture
   - override reason logging
   - post-event review loop for retraining

8. **Data-product structure**
   - ingestion
   - SQL feature mart
   - training pipeline
   - batch scoring
   - alert service
   - API
   - audit/decision log
   - reporting assets

---

## Repository structure

```text
.
├── configs/
├── data/sample/raw/
├── docs/
├── sql/
├── src/yes_forecast_risk/
│   ├── api/
│   ├── data/
│   ├── decision/
│   ├── explain/
│   ├── features/
│   ├── models/
│   ├── monitoring/
│   ├── workflow/
│   └── cli.py
├── tests/
├── Dockerfile
├── Makefile
├── pyproject.toml
└── README.md
```

---

## Quickstart

### 1) Install

```bash
python -m pip install -r requirements.txt
export PYTHONPATH=src
```

### 2) Run the end-to-end demo

```bash
python -m yes_forecast_risk.cli run-demo
```

This will:

- generate synthetic forecast-risk data with NZ-style energy-market patterns
- create a SQLite warehouse
- build the SQL feature mart
- train champion, challenger, and baseline models
- score the latest batch
- apply the decision/policy layer
- compute monitoring metrics
- generate plots and a markdown report

Outputs are written to:

- `artifacts/demo_run/`
- `artifacts/models/latest/`
- `warehouse/yes_risk_demo.sqlite`

### 3) Start the API

```bash
uvicorn yes_forecast_risk.api.main:app --reload
```

Then visit:

- `GET /health`
- `POST /score`

Example request:

```json
{
  "forecast_date": "2025-06-30",
  "region": "CNI",
  "horizon_hours": 24,
  "weather_uncertainty": 0.82,
  "load_ramp": 0.71,
  "model_disagreement": 0.77,
  "calendar_anomaly": 0.10,
  "holiday_flag": 0,
  "input_drift_score": 0.18,
  "analog_miss_rate": 0.64,
  "missingness_pct": 0.01,
  "data_latency_min": 18,
  "prior_abs_error": 22.4,
  "forecast_spread": 0.73,
  "price_spike_proxy": 0.68,
  "exposure_nzd": 27500,
  "source_health": 0.97
}
```

---

## Demo outputs

A successful demo run creates:

- `daily_risk_table.csv`
- `alert_queue.csv`
- `decision_log.csv`
- `monitoring_summary.json`
- `report.md`
- `plots/risk_heatmap.png`
- `plots/confidence_distribution.png`
- `plots/root_cause_breakdown.png`

---

## Example business output fields

| Field | Description |
|---|---|
| `risk_score` | calibrated high-risk probability |
| `risk_band` | Low / Medium / High / Critical |
| `predicted_severity` | expected severity of the miss |
| `expected_impact_nzd` | exposure-adjusted expected impact |
| `recommended_action` | analyst or ops action |
| `escalation_priority` | P4–P1 style urgency |
| `confidence_level` | low / medium / high |
| `reason_codes` | business-readable trigger codes |
| `suggested_owner` | squad owner for the next step |
| `decision_mode` | champion / fallback |

---

## Why the models are structured this way

This project is intentionally built to be **runnable** and **maintainable**.

For the demo implementation, each model head (binary classifier, severity regressor, risk-type multiclass) competes the following candidates:

- **Always-on sklearn candidates**: logistic / ridge baselines, random forest (full and shallow), and a decision tree for production-style scoring and interpretability
- **Optional `xgb_candidate` / `lgb_candidate`**: XGBoost and LightGBM challengers registered when the libraries are installed (both listed in `requirements.txt`). They compete under the same utility score and coverage guardrails as the sklearn candidates; if either library is missing, the pipeline silently continues with sklearn only. The bundle's `xgboost_enabled` / `lightgbm_enabled` flags record which libraries were active for audit.
- **Rule-based fallback** for degraded data conditions (high missingness, latency, PSI drift, or low source health)

This makes the repo easy to run on a plain Python install and easy to extend. In a real Yes Energy environment, the same architecture can host more advanced ensembles, calibrated gradient-boosted models, monotonic GBDTs, or other production-approved forecasting-risk components.

---

## Key design choices

- **Champion–challenger selection** is based on a composite utility score (AUC + 0.6·PR-AUC − 0.4·Brier), not just headline AUC. See [`PROMOTION_POLICY.md`](docs/PROMOTION_POLICY.md) for the full rules.
- **Isotonic calibration** is applied to the promoted champion on the validation window so `risk_score` is a usable probability.
- **Fallback mode** is triggered by severe missingness, latency, drift, unstable sources, or feature PSI above threshold.
- **Reason codes** are generated from business-readable feature patterns.
- **Feedback events** are stored in the warehouse and used to refine effective labels.
- **Workflow integration** is treated as part of the product, not an afterthought.

---

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Operations Runbook](docs/OPERATIONS_RUNBOOK.md)
- [Workflow Integration](docs/WORKFLOW_INTEGRATION.md)
- [Cross-Functional Delivery](docs/CROSS_FUNCTIONAL_DELIVERY.md)
- [Alert Taxonomy](docs/ALERT_TAXONOMY.md)
- [Model Card](docs/MODEL_CARD.md)
- [Promotion Policy](docs/PROMOTION_POLICY.md)
- [JD Alignment](docs/JD_ALIGNMENT.md)

---


A resume or interview version of this project can be framed as:

> Built an operational forecast risk decision support system for power market workflows, extending a high-risk day classifier into a production-facing analytics product with calibrated risk bands, expected impact scoring, business-readable reason codes, champion–challenger governance, feedback-informed retraining, and workflow-integrated alerting through API and batch reporting.

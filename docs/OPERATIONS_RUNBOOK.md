# Operations Runbook

## Purpose

Operate the forecast-risk decision support system safely in production-style workflows.

## Daily schedule

1. 06:30 local: ingest latest upstream signals
2. 06:40 local: build/update feature mart
3. 06:45 local: run batch scoring for 24h and 48h horizons
4. 06:47 local: apply decision policy and fallback checks
5. 06:50 local: publish `daily_risk_table.csv` and `alert_queue.csv`
6. 07:00 local: forecast analysts review `High` and `Critical` queue items
7. Before 12:00 local: close same-day analyst review actions
8. Weekly: review false-alert burden and reason-code quality
9. Monthly: model review and retraining checkpoint

## Fallback conditions

Fallback scoring is enabled when any of the following occurs:

- missingness > configured threshold
- data latency > configured threshold
- source health too low
- input drift score too high
- feature PSI exceeds threshold

## Escalation guidance

- **Low**: passive monitoring
- **Medium**: analyst review before market close
- **High**: escalate to Forecast Operations Lead
- **Critical**: open cross-functional ops review, include platform/data on-call if source issues exist

## Retraining triggers

Retrain or formally review the system if:

- AUC falls below floor
- PR-AUC falls below floor
- PSI exceeds allowed maximum on key features
- calibration deteriorates materially
- high-severity misses cluster in a region or regime

## Review artefacts

- monitoring summary JSON
- daily report markdown
- weekly top false-positive and false-negative samples
- updated model card
- approval checklist signed by DS lead / ops lead

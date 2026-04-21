# Cross-Functional Delivery

## Operating model

This project assumes a squad-style delivery model with shared ownership across roles.

### Forecast Analysts
- help define what counts as a meaningful high-risk event
- review daily alerts
- submit false-positive / false-negative feedback

### Forecast Operations Lead
- owns escalation thresholds and intervention policy
- reviews `High` and `Critical` recommendations
- signs off on operational changes

### Data Scientist
- maintains model logic, calibration, and monitoring
- owns champion–challenger review
- translates feature patterns into explainable reason codes

### Data / Platform Engineer
- supports ingestion, batch scheduling, API deployment, and logging
- ensures data freshness and source reliability

### Governance / Lead DS
- approves model card and retraining policy
- reviews rollback and fallback conditions

## Delivery artefacts

- architecture doc
- model card
- alert taxonomy
- runbook
- review checklist
- monitoring summary

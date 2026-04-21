# Workflow Integration

## Product workflow

This system is designed to sit inside a real forecast-operations workflow, not as a standalone notebook.

### Daily use

- daily batch produces a ranked queue
- analysts review medium/high/critical alerts
- forecast ops lead focuses on escalated items
- platform or data engineering are notified if data-quality reason codes dominate
- post-event review captures whether the alert was correct and whether the action was useful

## Output tables

- `daily_risk_table.csv`: all scored rows with risk score, severity, type, and reason codes
- `alert_queue.csv`: only action-worthy rows, ranked by impact-weighted priority
- `decision_log.csv`: audit-friendly log of recommendation, owner, and decision mode

## Human-in-the-loop feedback

Reviewers can append:

- reviewed label
- override reason
- review status
- reviewer name
- reviewed timestamp

These feedback events are stored in the warehouse and used to update the effective label for subsequent retraining.

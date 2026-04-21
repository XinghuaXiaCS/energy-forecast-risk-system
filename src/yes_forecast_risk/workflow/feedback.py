from __future__ import annotations

from datetime import datetime, timezone
from yes_forecast_risk.data.warehouse import append_feedback_event


def record_feedback(db_path: str, forecast_date: str, region: str, horizon_hours: int, reviewer: str, reviewed_label: int, override_reason: str, review_status: str) -> None:
    append_feedback_event(
        db_path,
        {
            "forecast_date": forecast_date,
            "region": region,
            "horizon_hours": int(horizon_hours),
            "reviewer": reviewer,
            "reviewed_label": int(reviewed_label),
            "override_reason": override_reason,
            "review_status": review_status,
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
        },
    )

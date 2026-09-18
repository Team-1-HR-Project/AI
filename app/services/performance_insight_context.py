"""Performance Insight Context Builder.

Builds sanitized, fail-closed performance data contexts for employees using ONLY
approved PerformanceRecords. Deterministically calculates trends across chronological
periods without LLM involvement or speculative causal reasoning.
"""

from __future__ import annotations

import re
from typing import Any

from sqlalchemy.orm import Session

from app.models import Employee, PerformanceRecord

# Context budget limits
MAX_PERFORMANCE_RECORDS = 10
MAX_FIELD_CHARS = 100

METRIC_FIELDS = [
    "overall_score",
    "task_completion_rate",
    "goal_achievement_rate",
    "attendance_rate",
]


def _clean_str(val: Any, max_chars: int = MAX_FIELD_CHARS) -> str:
    """Sanitizes text fields and truncates them to max_chars."""
    if val is None:
        return ""
    text = str(val).strip()
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text


def _safe_float(val: Any) -> float:
    """Safely casts a numeric field to a 2-decimal rounded float."""
    if val is None:
        return 0.0
    try:
        return round(float(val), 2)
    except (ValueError, TypeError):
        return 0.0


def parse_period_key(period_str: str | None) -> tuple[int, int, str]:
    """Parses period strings into a sortable tuple (year, sub_period, raw).

    Supports formats like '2026-Q3', '2026Q3', '2026-03', '2026', falling back
    gracefully to lexical order for unknown formats.
    """
    if not period_str:
        return (0, 0, "")
    s = period_str.strip().upper()
    # Match YYYY-Q# or YYYYQ#
    m_q = re.match(r"^(\d{4})[-_/ ]?Q([1-4])$", s)
    if m_q:
        return (int(m_q.group(1)), int(m_q.group(2)), s)
    # Match YYYY-MM or YYYY/MM
    m_m = re.match(r"^(\d{4})[-_/ ]?(\d{1,2})$", s)
    if m_m:
        return (int(m_m.group(1)), int(m_m.group(2)), s)
    # Match YYYY
    m_y = re.match(r"^(\d{4})$", s)
    if m_y:
        return (int(m_y.group(1)), 0, s)
    return (0, 0, s)


def calculate_trend(
    current_value: float,
    previous_value: float,
    threshold: float = 0.0,
) -> dict[str, Any]:
    """Calculates deterministic trend between two metric values.

    Returns direction ('improved', 'declined', 'stable'), absolute delta, and
    percent change. Does NOT hypothesize causes.
    """
    curr = _safe_float(current_value)
    prev = _safe_float(previous_value)
    delta = round(curr - prev, 2)

    if delta > threshold:
        direction = "improved"
    elif delta < -threshold:
        direction = "declined"
    else:
        direction = "stable"

    percent_change = None
    if prev != 0.0:
        percent_change = round((delta / prev) * 100, 2)

    return {
        "previous_value": prev,
        "current_value": curr,
        "delta": delta,
        "direction": direction,
        "percent_change": percent_change,
    }


class PerformanceInsightContextBuilder:
    """Service to query approved performance records and build sanitized context."""

    @staticmethod
    def build_context(
        db: Session,
        employee_id: str,
        period: str | None = None,
        limit: int = MAX_PERFORMANCE_RECORDS,
        delta_threshold: float = 0.0,
        min_periods: int = 1,
    ) -> dict[str, Any]:
        """Builds sanitized performance insight context for the given employee.

        Fail-closed rules:
        - Employee not found -> has_sufficient_data = False
        - No approved records -> has_sufficient_data = False
        - Requested period not found -> has_sufficient_data = False
        - Unapproved records (is_approved=False) are strictly excluded
        - Cross-period comparison requires >= 2 approved records
        """
        # 1. Verify employee exists and extract safe identity fields only
        employee = db.query(Employee).filter(Employee.id == employee_id).first()
        if not employee:
            return {
                "employee": None,
                "has_sufficient_data": False,
                "has_trend_data": False,
                "reason": f"Employee '{employee_id}' not found.",
                "target_period": period,
                "comparison_period": None,
                "facts": {
                    "records_count": 0,
                    "periods": [],
                    "metrics_by_period": [],
                },
                "metrics_by_period": [],
                "calculated_trends": {
                    "comparison_available": False,
                    "from_period": None,
                    "to_period": None,
                    "metric_trends": {},
                    "period_over_period": [],
                },
                "trends": {},
            }

        employee_dict = {
            "id": employee.id,
            "first_name": _clean_str(employee.first_name, 50),
            "last_name": _clean_str(employee.last_name, 50),
            "role_title": _clean_str(employee.role_title, 100),
            "department": _clean_str(employee.department, 100),
        }

        # 2. Query ONLY approved PerformanceRecord rows for this employee
        query = db.query(PerformanceRecord).filter(
            PerformanceRecord.employee_id == employee.id,
            PerformanceRecord.is_approved.is_(True),
        )
        all_records = query.all()

        if not all_records:
            return {
                "employee": employee_dict,
                "has_sufficient_data": False,
                "has_trend_data": False,
                "reason": f"No approved performance records found for employee '{employee_id}'.",
                "target_period": period,
                "comparison_period": None,
                "facts": {
                    "records_count": 0,
                    "periods": [],
                    "metrics_by_period": [],
                },
                "metrics_by_period": [],
                "calculated_trends": {
                    "comparison_available": False,
                    "from_period": None,
                    "to_period": None,
                    "metric_trends": {},
                    "period_over_period": [],
                },
                "trends": {},
            }

        # 3. Sort records chronologically (oldest to newest)
        sorted_records = sorted(
            all_records,
            key=lambda r: (
                parse_period_key(r.period),
                r.created_at or 0,
                r.id or 0,
            ),
        )

        # Enforce record limit budget if more records exist
        if limit and len(sorted_records) > limit:
            # Keep the most recent `limit` records
            sorted_records = sorted_records[-limit:]

        # Map to sanitized factual metric dictionaries
        factual_metrics: list[dict[str, Any]] = [
            {
                "id": r.id,
                "period": _clean_str(r.period, 20),
                "overall_score": _safe_float(r.overall_score),
                "task_completion_rate": _safe_float(r.task_completion_rate),
                "goal_achievement_rate": _safe_float(r.goal_achievement_rate),
                "attendance_rate": _safe_float(r.attendance_rate),
            }
            for r in sorted_records
        ]

        # 4. Handle period targeting if a specific period was requested
        target_record: dict[str, Any] | None = None
        previous_record: dict[str, Any] | None = None

        if period:
            # Check if requested period is present
            target_idx = -1
            for idx, item in enumerate(factual_metrics):
                if item["period"] == period:
                    target_idx = idx
                    target_record = item
                    break

            if target_idx == -1:
                return {
                    "employee": employee_dict,
                    "has_sufficient_data": False,
                    "has_trend_data": False,
                    "reason": f"No approved performance record found for period '{period}'.",
                    "target_period": period,
                    "comparison_period": None,
                    "facts": {
                        "records_count": len(factual_metrics),
                        "periods": [m["period"] for m in factual_metrics],
                        "metrics_by_period": factual_metrics,
                    },
                    "metrics_by_period": factual_metrics,
                    "calculated_trends": {
                        "comparison_available": False,
                        "from_period": None,
                        "to_period": None,
                        "metric_trends": {},
                        "period_over_period": [],
                    },
                    "trends": {},
                }

            if target_idx > 0:
                previous_record = factual_metrics[target_idx - 1]
        else:
            # Default to latest period as target, and the one before it as previous
            target_record = factual_metrics[-1]
            if len(factual_metrics) >= 2:
                previous_record = factual_metrics[-2]

        # 5. Calculate deterministic trends
        has_trend_data = previous_record is not None and target_record is not None
        has_sufficient_data = len(factual_metrics) >= min_periods

        metric_trends: dict[str, dict[str, Any]] = {}
        if has_trend_data and target_record and previous_record:
            for metric in METRIC_FIELDS:
                metric_trends[metric] = calculate_trend(
                    current_value=target_record[metric],
                    previous_value=previous_record[metric],
                    threshold=delta_threshold,
                )

        # Build consecutive period-over-period transitions
        period_over_period: list[dict[str, Any]] = []
        if len(factual_metrics) >= 2:
            for i in range(len(factual_metrics) - 1):
                p_from = factual_metrics[i]
                p_to = factual_metrics[i + 1]
                step_trends = {
                    m: calculate_trend(
                        current_value=p_to[m],
                        previous_value=p_from[m],
                        threshold=delta_threshold,
                    )
                    for m in METRIC_FIELDS
                }
                period_over_period.append(
                    {
                        "from_period": p_from["period"],
                        "to_period": p_to["period"],
                        "trends": step_trends,
                    }
                )

        reason: str | None = None
        if not has_sufficient_data:
            reason = (
                f"Insufficient performance records (minimum {min_periods} required, "
                f"found {len(factual_metrics)})."
            )
        elif not has_trend_data:
            reason = (
                "Only one approved performance period available; "
                "cross-period comparative trend requires at least two periods."
            )

        target_period_str = target_record["period"] if target_record else period
        comparison_period_str = previous_record["period"] if previous_record else None

        facts = {
            "records_count": len(factual_metrics),
            "periods": [m["period"] for m in factual_metrics],
            "metrics_by_period": factual_metrics,
        }

        calculated_trends = {
            "comparison_available": has_trend_data,
            "from_period": comparison_period_str,
            "to_period": target_period_str,
            "metric_trends": metric_trends,
            "period_over_period": period_over_period,
        }

        return {
            "employee": employee_dict,
            "has_sufficient_data": has_sufficient_data,
            "has_trend_data": has_trend_data,
            "reason": reason,
            "target_period": target_period_str,
            "comparison_period": comparison_period_str,
            "facts": facts,
            "metrics_by_period": factual_metrics,
            "calculated_trends": calculated_trends,
            "trends": metric_trends,
        }


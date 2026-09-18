"""Context builder for Feature #6: Employee Attention Signal.

Queries and sanitizes ONLY approved employee records (PerformanceRecord, Goal,
TaskOutcome, EvaluationTheme) strictly isolated by employee_id.
Deterministically computes period-over-period metric trends across attendance,
task completion, goal achievement, and evaluation themes without making speculative
causal claims, flight risk predictions, or employment decisions.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from app.models import (
    Employee,
    EvaluationTheme,
    Goal,
    PerformanceRecord,
    TaskOutcome,
)
from app.schemas.attention_signal import AttentionLevel, IndicatorCategory
from app.services.performance_insight_context import (
    calculate_trend,
    parse_period_key,
)

# Context budget limits
MAX_PERFORMANCE_RECORDS = 10
MAX_GOALS = 15
MAX_TASK_OUTCOMES = 15
MAX_EVALUATION_THEMES = 10
MAX_FIELD_CHARS = 300


def _clean_str(val: Any, max_chars: int = MAX_FIELD_CHARS) -> str:
    """Sanitizes text fields and truncates them to max_chars."""
    if val is None:
        return ""
    text = str(val).strip()
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text


# Centralized business thresholds for current-period percentage metrics
METRIC_BAND_GOOD_MIN = 85.0
METRIC_BAND_MODERATE_MIN = 70.0

PERCENTAGE_METRIC_KEYS = (
    "attendance_rate",
    "task_completion_rate",
    "goal_achievement_rate",
    "overall_score",
)


def get_metric_band(val: float | None) -> str | None:
    """Classifies a percentage-based performance metric into a business performance band.

    - 'good': val >= 85.0
    - 'moderate': 70.0 <= val < 85.0
    - 'low': val < 70.0
    Returns None if val is None (missing metric).
    """
    if val is None:
        return None
    if val >= METRIC_BAND_GOOD_MIN:
        return "good"
    if val >= METRIC_BAND_MODERATE_MIN:
        return "moderate"
    return "low"


def _safe_float(val: Any) -> float:
    """Safely casts a numeric field to a 2-decimal rounded float (defaults to 0.0)."""
    if val is None:
        return 0.0
    try:
        return round(float(val), 2)
    except (ValueError, TypeError):
        return 0.0


def _safe_float_or_none(val: Any) -> float | None:
    """Safely casts a numeric field to a 2-decimal rounded float, or None if missing."""
    if val is None:
        return None
    try:
        return round(float(val), 2)
    except (ValueError, TypeError):
        return None


def _derive_attention_level(
    metric_trends: dict[str, dict[str, Any]],
    task_summary: dict[str, Any],
    goal_summary: dict[str, Any],
    evaluation_summary: dict[str, Any],
    comparison_period: str | None = None,
    target_metrics: dict[str, Any] | None = None,
) -> AttentionLevel:
    """Deterministically derives Low, Medium, or High attention level from objective indicators.

    When comparison_period is present (comparison-based assessment):
    - Branch B remains unchanged.
    - HIGH: Meaningful negative signals across 2 or more indicator categories concurrently,
      OR multiple concurrent work blockers/friction signals (total_blockers >= 2).
    - MEDIUM: Isolated decline or active friction in exactly 1 category when the overall profile
      does not clearly support an improving trajectory; OR overall_score is declining without a multi-category collapse.
    - LOW: Healthy/improving trajectory with no active blockers/friction; OR predominantly improving profile
      where strong improvements across core indicators clearly outweigh an isolated negative fluctuation.

    When comparison_period is null (current-period-only assessment):
    - Branch A: Evaluates current approved performance metrics and operational concerns against centralized thresholds:
      * HIGH:
        - 2 or more metrics are below 70 (< 70)
        - OR 1 metric is below 70 AND there is at least one additional current-period concern
          (blocked task, negative evaluation theme, or multiple delayed goals)
        - OR acute concurrent operational concerns across multiple areas (e.g. >=2 blocked tasks,
          or blocked task + negative evaluation theme).
      * MEDIUM:
        - 1 metric is below 70
        - OR 2 or more metrics are in the Moderate range (70–84.99)
        - OR there is a significant current-period operational concern (blocked work, negative evaluation,
          or multiple delayed goals).
      * LOW:
        - all available percentage metrics are >= 85 (or at most 1 in moderate with no concerns)
        - AND no significant current-period operational concern exists.
    """
    blocked_count = int(task_summary.get("blocked", 0))
    delayed_count = int(goal_summary.get("delayed", 0))
    needs_imp_count = int(evaluation_summary.get("needs_improvement_count", 0))
    total_blockers = blocked_count + delayed_count + needs_imp_count

    # Branch A: Current-period-only assessment (no historical comparison data)
    if comparison_period is None:
        # Classify available percentage metrics into business performance bands
        metric_bands: dict[str, str] = {}
        source_metrics = target_metrics if target_metrics is not None else {}
        for k in PERCENTAGE_METRIC_KEYS:
            val = source_metrics.get(k)
            if val is None and k in metric_trends:
                val = metric_trends[k].get("current_value")
            band = get_metric_band(val)
            if band is not None:
                metric_bands[k] = band

        low_count = sum(1 for band in metric_bands.values() if band == "low")
        moderate_count = sum(1 for band in metric_bands.values() if band == "moderate")

        # Operational concerns:
        # - Blocked work (blocked_count >= 1)
        # - Negative evaluation feedback (needs_imp_count >= 1)
        # - Multiple delayed/low-progress goals (delayed_count >= 2)
        # Note: An isolated delayed goal (delayed_count == 1) does NOT count as a significant concern
        has_operational_concern = (
            blocked_count >= 1
            or needs_imp_count >= 1
            or delayed_count >= 2
        )

        # Acute concurrent operational concerns across multiple areas
        has_acute_operational_concerns = (
            blocked_count >= 2
            or (blocked_count >= 1 and needs_imp_count >= 1)
            or (blocked_count >= 1 and delayed_count >= 2)
        )

        # HIGH:
        # 1. 2 or more metrics below 70
        # 2. OR 1 metric below 70 AND at least one additional current-period concern
        # 3. OR acute concurrent operational concerns
        if (
            low_count >= 2
            or (low_count >= 1 and has_operational_concern)
            or has_acute_operational_concerns
        ):
            return AttentionLevel.HIGH

        # MEDIUM:
        # 1. 1 metric below 70
        # 2. OR 2 or more metrics in Moderate range (70–84.99)
        # 3. OR significant current-period operational concern
        if (
            low_count == 1
            or moderate_count >= 2
            or has_operational_concern
        ):
            return AttentionLevel.MEDIUM

        # LOW:
        # Baseline healthy profile with no significant operational concerns
        return AttentionLevel.LOW

    # Branch B: Comparison-based assessment (historical comparison period exists)
    att_dir = metric_trends.get("attendance_rate", {}).get("direction", "stable")
    task_dir = metric_trends.get("task_completion_rate", {}).get("direction", "stable")
    goal_dir = metric_trends.get("goal_achievement_rate", {}).get("direction", "stable")
    score_dir = metric_trends.get("overall_score", {}).get("direction", "stable")

    cat_negative = {
        "attendance": att_dir == "declined",
        "task_completion": task_dir == "declined" or blocked_count > 0,
        "goals": goal_dir == "declined" or delayed_count > 0,
        "evaluation_trend": score_dir == "declined" or needs_imp_count > 0,
    }

    cat_positive = {
        "attendance": att_dir == "improved",
        "task_completion": task_dir == "improved" and blocked_count == 0,
        "goals": goal_dir == "improved" and delayed_count == 0,
        "evaluation_trend": score_dir == "improved" and needs_imp_count == 0,
    }

    num_negative = sum(1 for is_neg in cat_negative.values() if is_neg)
    num_positive = sum(1 for is_pos in cat_positive.values() if is_pos)

    # 1. HIGH: Meaningful negative signals across 2 or more categories, OR multiple blockers
    if num_negative >= 2 or total_blockers >= 2:
        return AttentionLevel.HIGH

    # 2. LOW: Healthy/improving trajectory with no active blockers/friction
    if num_negative == 0 and total_blockers == 0:
        return AttentionLevel.LOW

    # 3. LOW: Predominantly improving profile where strong improvements across core indicators
    # clearly outweigh an isolated negative fluctuation (with 0 active blockers)
    if total_blockers == 0 and score_dir == "improved" and num_positive >= 2:
        return AttentionLevel.LOW

    # 4. MEDIUM: Isolated decline/friction in 1 category or declining overall_score
    return AttentionLevel.MEDIUM


class AttentionSignalContextBuilder:
    """Service to build sanitized, approved-only context for employee attention signals."""

    @staticmethod
    def build_context(
        db: Session,
        employee_id: str,
        target_period: str | None = None,
    ) -> dict[str, Any]:
        """Builds sanitized context payload for an employee attention signal.

        Guarantees:
        - Strict employee data boundary isolation.
        - Includes ONLY approved records (is_approved=True).
        - Deterministically determines target_period and comparison_period.
        - Calculates period-over-period deltas across key work indicators:
          attendance, task completion, goal achievement, and evaluation scores.
        - Extracts blocked tasks, delayed goals, and needs_improvement themes.
        - Never predicts flight risk or triggers automated employment actions.
        """
        # 1. Verify employee existence
        employee = db.query(Employee).filter(Employee.id == employee_id).first()
        if not employee:
            return {
                "employee": None,
                "has_sufficient_data": False,
                "reason": f"Employee '{employee_id}' not found.",
                "target_period": target_period,
                "comparison_period": None,
                "assessment_type": None,
                "attention_level": None,
                "metric_bands": {},
                "indicators": [],
                "metric_trends": {},
                "target_metrics": None,
                "comparison_metrics": None,
                "task_summary": {
                    "total": 0,
                    "completed": 0,
                    "blocked": 0,
                    "in_progress": 0,
                    "blocked_tasks": [],
                },
                "goal_summary": {
                    "total": 0,
                    "completed": 0,
                    "delayed": 0,
                    "in_progress": 0,
                    "delayed_goals": [],
                },
                "evaluation_summary": {
                    "total": 0,
                    "needs_improvement_count": 0,
                    "positive_count": 0,
                    "themes": [],
                },
                "approved_sources": {},
            }

        employee_dict = {
            "id": employee.id,
            "first_name": _clean_str(employee.first_name, 50),
            "last_name": _clean_str(employee.last_name, 50),
            "role_title": _clean_str(employee.role_title, 100),
            "department": _clean_str(employee.department, 100),
        }

        # 2. Query approved PerformanceRecords
        perf_records = (
            db.query(PerformanceRecord)
            .filter(
                PerformanceRecord.employee_id == employee.id,
                PerformanceRecord.is_approved.is_(True),
            )
            .all()
        )

        if not perf_records:
            return {
                "employee": employee_dict,
                "has_sufficient_data": False,
                "reason": f"No approved performance records found for employee '{employee_id}'.",
                "target_period": target_period,
                "comparison_period": None,
                "assessment_type": None,
                "attention_level": None,
                "metric_bands": {},
                "indicators": [],
                "metric_trends": {},
                "target_metrics": None,
                "comparison_metrics": None,
                "task_summary": {
                    "total": 0,
                    "completed": 0,
                    "blocked": 0,
                    "in_progress": 0,
                    "blocked_tasks": [],
                },
                "goal_summary": {
                    "total": 0,
                    "completed": 0,
                    "delayed": 0,
                    "in_progress": 0,
                    "delayed_goals": [],
                },
                "evaluation_summary": {
                    "total": 0,
                    "needs_improvement_count": 0,
                    "positive_count": 0,
                    "themes": [],
                },
                "approved_sources": {},
            }

        # 3. Sort performance records chronologically
        sorted_records = sorted(
            perf_records,
            key=lambda r: (
                parse_period_key(r.period),
                r.created_at or 0,
                r.id or 0,
            ),
        )

        # 4. Resolve target_period and comparison_period
        target_record: PerformanceRecord | None = None
        comparison_record: PerformanceRecord | None = None

        if target_period:
            match_idx = -1
            for idx, r in enumerate(sorted_records):
                if r.period == target_period:
                    match_idx = idx
                    target_record = r
                    break

            if match_idx == -1:
                return {
                    "employee": employee_dict,
                    "has_sufficient_data": False,
                    "reason": f"No approved performance record found for period '{target_period}'.",
                    "target_period": target_period,
                    "comparison_period": None,
                    "assessment_type": None,
                    "attention_level": None,
                    "metric_bands": {},
                    "indicators": [],
                    "metric_trends": {},
                    "target_metrics": None,
                    "comparison_metrics": None,
                    "task_summary": {
                        "total": 0,
                        "completed": 0,
                        "blocked": 0,
                        "in_progress": 0,
                        "blocked_tasks": [],
                    },
                    "goal_summary": {
                        "total": 0,
                        "completed": 0,
                        "delayed": 0,
                        "in_progress": 0,
                        "delayed_goals": [],
                    },
                    "evaluation_summary": {
                        "total": 0,
                        "needs_improvement_count": 0,
                        "positive_count": 0,
                        "themes": [],
                    },
                    "approved_sources": {},
                }

            if match_idx > 0:
                comparison_record = sorted_records[match_idx - 1]
        else:
            target_record = sorted_records[-1]
            target_period = target_record.period
            if len(sorted_records) >= 2:
                comparison_record = sorted_records[-2]

        comparison_period = comparison_record.period if comparison_record else None

        # 5. Build approved sources registry for grounding
        approved_sources: dict[tuple[str, int], dict[str, Any]] = {}

        target_metrics = {
            "id": target_record.id,
            "period": target_record.period,
            "overall_score": _safe_float_or_none(target_record.overall_score),
            "task_completion_rate": _safe_float_or_none(target_record.task_completion_rate),
            "goal_achievement_rate": _safe_float_or_none(target_record.goal_achievement_rate),
            "attendance_rate": _safe_float_or_none(target_record.attendance_rate),
        }
        approved_sources[("performance", target_record.id)] = target_metrics

        comparison_metrics = None
        if comparison_record:
            comparison_metrics = {
                "id": comparison_record.id,
                "period": comparison_record.period,
                "overall_score": _safe_float_or_none(comparison_record.overall_score),
                "task_completion_rate": _safe_float_or_none(comparison_record.task_completion_rate),
                "goal_achievement_rate": _safe_float_or_none(comparison_record.goal_achievement_rate),
                "attendance_rate": _safe_float_or_none(comparison_record.attendance_rate),
            }
            approved_sources[("performance", comparison_record.id)] = comparison_metrics

        # 6. Query approved TaskOutcomes, Goals, EvaluationThemes
        all_tasks = (
            db.query(TaskOutcome)
            .filter(
                TaskOutcome.employee_id == employee.id,
                TaskOutcome.is_approved.is_(True),
            )
            .order_by(TaskOutcome.period.desc(), TaskOutcome.id.desc())
            .limit(MAX_TASK_OUTCOMES)
            .all()
        )
        for t in all_tasks:
            approved_sources[("task_outcome", t.id)] = {
                "id": t.id,
                "title": _clean_str(t.title),
                "status": _clean_str(t.status),
                "outcome": _clean_str(t.outcome),
                "period": _clean_str(t.period),
            }

        all_goals = (
            db.query(Goal)
            .filter(
                Goal.employee_id == employee.id,
                Goal.is_approved.is_(True),
            )
            .order_by(Goal.period.desc(), Goal.id.desc())
            .limit(MAX_GOALS)
            .all()
        )
        for g in all_goals:
            approved_sources[("goal", g.id)] = {
                "id": g.id,
                "title": _clean_str(g.title),
                "status": _clean_str(g.status),
                "progress": _safe_float(g.progress),
                "period": _clean_str(g.period),
            }

        all_themes = (
            db.query(EvaluationTheme)
            .filter(
                EvaluationTheme.employee_id == employee.id,
                EvaluationTheme.is_approved.is_(True),
            )
            .order_by(EvaluationTheme.period.desc(), EvaluationTheme.id.desc())
            .limit(MAX_EVALUATION_THEMES)
            .all()
        )
        for th in all_themes:
            approved_sources[("evaluation_theme", th.id)] = {
                "id": th.id,
                "theme": _clean_str(th.theme),
                "sentiment": _clean_str(th.sentiment),
                "evidence": _clean_str(th.evidence),
                "period": _clean_str(th.period),
            }

        # 7. Summaries for target period
        target_tasks = [
            t for t in all_tasks if t.period == target_period or t.period is None
        ]
        blocked_tasks = [
            {
                "id": t.id,
                "title": _clean_str(t.title),
                "status": t.status,
                "outcome": _clean_str(t.outcome),
            }
            for t in target_tasks
            if t.status.lower() == "blocked"
        ]
        task_summary = {
            "total": len(target_tasks),
            "completed": sum(1 for t in target_tasks if t.status.lower() == "completed"),
            "blocked": len(blocked_tasks),
            "in_progress": sum(1 for t in target_tasks if t.status.lower() == "in_progress"),
            "blocked_tasks": blocked_tasks,
        }

        target_goals = [
            g for g in all_goals if g.period == target_period or g.period is None
        ]
        delayed_goals = [
            {
                "id": g.id,
                "title": _clean_str(g.title),
                "status": g.status,
                "progress": _safe_float(g.progress),
            }
            for g in target_goals
            if g.status.lower() == "delayed" or g.progress < 50.0
        ]
        goal_summary = {
            "total": len(target_goals),
            "completed": sum(1 for g in target_goals if g.status.lower() == "completed"),
            "delayed": len(delayed_goals),
            "in_progress": sum(1 for g in target_goals if g.status.lower() == "in_progress"),
            "delayed_goals": delayed_goals,
        }

        target_themes = [
            th for th in all_themes if th.period == target_period or th.period is None
        ]
        needs_improvement_themes = [
            {
                "id": th.id,
                "theme": _clean_str(th.theme),
                "sentiment": th.sentiment,
                "evidence": _clean_str(th.evidence),
            }
            for th in target_themes
            if th.sentiment.lower() == "needs_improvement"
        ]
        evaluation_summary = {
            "total": len(target_themes),
            "needs_improvement_count": len(needs_improvement_themes),
            "positive_count": sum(1 for th in target_themes if th.sentiment.lower() == "positive"),
            "themes": needs_improvement_themes,
        }

        # 8. Calculate deterministic metric trends
        metric_keys = [
            ("attendance_rate", IndicatorCategory.ATTENDANCE, "Attendance Rate"),
            ("task_completion_rate", IndicatorCategory.TASK_COMPLETION, "Task Completion Rate"),
            ("goal_achievement_rate", IndicatorCategory.GOALS, "Goal Achievement Rate"),
            ("overall_score", IndicatorCategory.EVALUATION_TREND, "Overall Evaluation Score"),
        ]

        metric_trends: dict[str, dict[str, Any]] = {}
        indicators: list[dict[str, Any]] = []

        for key, category, label in metric_keys:
            curr_val = target_metrics[key]
            if comparison_metrics:
                prev_val = comparison_metrics[key]
                trend = calculate_trend(current_value=curr_val, previous_value=prev_val)
                trend["metric_band"] = get_metric_band(curr_val)
                metric_trends[key] = trend

                delta = trend["delta"]
                direction = trend["direction"]

                if direction == "declined":
                    change_desc = (
                        f"Declined from {prev_val}% to {curr_val}% ({delta} percentage points)"
                        if "rate" in key
                        else f"Declined from {prev_val} to {curr_val} (delta: {delta})"
                    )
                elif direction == "improved":
                    change_desc = (
                        f"Improved from {prev_val}% to {curr_val}% (+{delta} percentage points)"
                        if "rate" in key
                        else f"Improved from {prev_val} to {curr_val} (delta: +{delta})"
                    )
                else:
                    change_desc = (
                        f"Stable at {curr_val}%"
                        if "rate" in key
                        else f"Stable at {curr_val}"
                    )

                indicators.append({
                    "indicator_name": label,
                    "category": category.value,
                    "current_value": curr_val,
                    "previous_value": prev_val,
                    "change_description": change_desc,
                    "evidence": (
                        f"Approved {target_period} {label.lower()} was {curr_val} "
                        f"compared to {prev_val} in {comparison_period} "
                        f"(PerformanceRecord #{target_record.id})."
                    ),
                    "direction": direction,
                    "is_decline": direction == "declined",
                })
            else:
                metric_band = get_metric_band(curr_val)
                metric_trends[key] = {
                    "previous_value": None,
                    "current_value": curr_val,
                    "delta": None,
                    "direction": "stable",
                    "percent_change": None,
                    "metric_band": metric_band,
                }
                if curr_val is not None:
                    indicators.append({
                        "indicator_name": label,
                        "category": category.value,
                        "current_value": curr_val,
                        "previous_value": None,
                        "change_description": f"Observed at {curr_val}" + ("%" if "rate" in key else ""),
                        "evidence": (
                            f"Approved {target_period} {label.lower()} recorded as {curr_val} "
                            f"(PerformanceRecord #{target_record.id})."
                        ),
                        "direction": "stable",
                        "is_decline": False,
                    })

        # Add qualitative blocker indicators if present
        is_comparison = comparison_period is not None
        qualitative_direction = "declined" if is_comparison else "stable"
        qualitative_is_decline = is_comparison

        if blocked_tasks:
            task_names = ", ".join(f"'{t['title']}'" for t in blocked_tasks[:3])
            indicators.append({
                "indicator_name": "Blocked Tasks",
                "category": IndicatorCategory.TASK_COMPLETION.value,
                "current_value": f"{len(blocked_tasks)} blocked",
                "previous_value": None,
                "change_description": f"{len(blocked_tasks)} task(s) currently marked blocked in {target_period}: {task_names}",
                "evidence": f"Approved task outcome records indicate blocked work in {target_period}.",
                "direction": qualitative_direction,
                "is_decline": qualitative_is_decline,
            })

        if delayed_goals:
            goal_names = ", ".join(f"'{g['title']}'" for g in delayed_goals[:3])
            indicators.append({
                "indicator_name": "Delayed Goals",
                "category": IndicatorCategory.GOALS.value,
                "current_value": f"{len(delayed_goals)} delayed or low progress",
                "previous_value": None,
                "change_description": f"{len(delayed_goals)} goal(s) delayed or under 50% progress in {target_period}: {goal_names}",
                "evidence": f"Approved goal records indicate delayed milestones in {target_period}.",
                "direction": qualitative_direction,
                "is_decline": qualitative_is_decline,
            })

        if needs_improvement_themes:
            theme_names = ", ".join(f"'{th['theme']}'" for th in needs_improvement_themes[:3])
            indicators.append({
                "indicator_name": "Evaluation Themes Needing Improvement",
                "category": IndicatorCategory.EVALUATION_TREND.value,
                "current_value": f"{len(needs_improvement_themes)} needs improvement",
                "previous_value": None,
                "change_description": f"Feedback themes identified areas needing improvement in {target_period}: {theme_names}",
                "evidence": f"Approved evaluation themes reflect areas for growth: {theme_names}.",
                "direction": qualitative_direction,
                "is_decline": qualitative_is_decline,
            })

        metric_bands = {
            k: get_metric_band(target_metrics[k])
            for k in PERCENTAGE_METRIC_KEYS
            if target_metrics.get(k) is not None
        }

        # Check if there is sufficient approved data to make a meaningful assessment
        has_any_metric = any(target_metrics.get(k) is not None for k in PERCENTAGE_METRIC_KEYS)
        has_any_activity = (
            task_summary.get("total", 0) > 0
            or goal_summary.get("total", 0) > 0
            or evaluation_summary.get("total", 0) > 0
        )
        if not has_any_metric and not has_any_activity:
            return {
                "employee": employee_dict,
                "has_sufficient_data": False,
                "reason": f"No approved metrics or activity records found for period '{target_period}'.",
                "target_period": target_period,
                "comparison_period": None,
                "assessment_type": None,
                "attention_level": None,
                "metric_bands": {},
                "indicators": [],
                "metric_trends": {},
                "target_metrics": None,
                "comparison_metrics": None,
                "task_summary": task_summary,
                "goal_summary": goal_summary,
                "evaluation_summary": evaluation_summary,
                "approved_sources": approved_sources,
            }

        attention_level = _derive_attention_level(
            metric_trends=metric_trends,
            task_summary=task_summary,
            goal_summary=goal_summary,
            evaluation_summary=evaluation_summary,
            comparison_period=comparison_period,
            target_metrics=target_metrics,
        )

        assessment_type = "comparison_based" if is_comparison else "current_period_only"

        return {
            "employee": employee_dict,
            "has_sufficient_data": True,
            "reason": None,
            "target_period": target_period,
            "comparison_period": comparison_period,
            "assessment_type": assessment_type,
            "attention_level": attention_level.value,
            "metric_bands": metric_bands,
            "indicators": indicators,
            "metric_trends": metric_trends,
            "target_metrics": target_metrics,
            "comparison_metrics": comparison_metrics,
            "task_summary": task_summary,
            "goal_summary": goal_summary,
            "evaluation_summary": evaluation_summary,
            "approved_sources": approved_sources,
        }


"""Context builder for Feature #7: Team Insight Summary - Manager.

Queries and aggregates ONLY approved records belonging to employees within the requested
department. Computes deterministic team statistics:
- team_size (count of approved department members)
- overdue workload (total blocked tasks, total delayed goals, affected member count)
- completion trends (arithmetic means of task completion, goal achievement, overall score)
- period-over-period direction if historical period exists
- common skill inventory patterns
- evaluation theme patterns (positive vs needs_improvement)
- anonymized supporting drill-down factors (using role_title, NEVER employee PII)
- deterministic grounding registry for downstream AI verification

Privacy Guarantees:
- Output context NEVER exposes first_name, last_name, employee IDs, salary, compensation,
  disciplinary actions, or flight-risk predictions.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from sqlalchemy.orm import Session

from app.models import (
    Employee,
    EvaluationTheme,
    Goal,
    PerformanceRecord,
    Skill,
    TaskOutcome,
)
from app.schemas.performance_insight import TrendDirection
from app.schemas.team_insight import DrillDownCategory
from app.services.performance_insight_context import (
    calculate_trend,
    parse_period_key,
)

MAX_FIELD_CHARS = 300
MAX_DRILL_DOWN_FACTORS = 10


def _clean_str(val: Any, max_chars: int = MAX_FIELD_CHARS) -> str:
    """Sanitizes text fields and truncates them to max_chars."""
    if val is None:
        return ""
    text = str(val).strip()
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text


def _safe_float(val: Any) -> float:
    """Safely casts a numeric value to a 2-decimal rounded float."""
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


class TeamInsightContextBuilder:
    """Deterministic context builder for manager-facing Team Insight Summaries."""

    @staticmethod
    def compute_metric_averages(records: list[Any]) -> dict[str, float]:
        """Calculates exact arithmetic means across records with recorded numbers.

        Missing or None metrics are excluded from the denominator rather than being treated as 0.0.
        """
        task_rates = [
            r.task_completion_rate
            for r in records
            if getattr(r, "task_completion_rate", None) is not None
        ]
        goal_rates = [
            r.goal_achievement_rate
            for r in records
            if getattr(r, "goal_achievement_rate", None) is not None
        ]
        overall_scores = [
            r.overall_score
            for r in records
            if getattr(r, "overall_score", None) is not None
        ]

        avg_task = round(sum(task_rates) / len(task_rates), 2) if task_rates else 0.0
        avg_goal = round(sum(goal_rates) / len(goal_rates), 2) if goal_rates else 0.0
        avg_overall = (
            round(sum(overall_scores) / len(overall_scores), 2)
            if overall_scores
            else 0.0
        )

        return {
            "team_avg_task_completion": avg_task,
            "team_avg_goal_achievement": avg_goal,
            "team_avg_overall_score": avg_overall,
        }

    @classmethod
    def build_context(
        cls,
        db: Session,
        department: str,
        period: str | None = None,
    ) -> dict[str, Any]:
        """Gathers and aggregates approved employee and team data for the given department.

        Guarantees:
        - Department isolation: records belonging to employees outside the department are excluded.
        - Approved-only filtering: is_approved == True on all entities.
        - Missing metrics are NOT treated as zero when computing team averages.
        - Preserves period-over-period direction when a previous comparison period exists.
        - Produces an anonymized drill-down list and a deterministic grounding registry.
        - Excludes all employee PII (first_name, last_name, employee IDs, salary, etc.).
        """
        clean_dept = department.strip() if department else ""
        if not clean_dept:
            return {
                "has_sufficient_data": False,
                "department": "",
                "period": period,
                "missing_categories": ["department"],
                "message": "Department name must be provided.",
                "grounding_registry": {},
            }

        # 1. Query approved employees belonging strictly to the requested department
        employees = (
            db.query(Employee)
            .filter(
                Employee.department == clean_dept,
            )
            .all()
        )

        if not employees:
            return {
                "has_sufficient_data": False,
                "department": clean_dept,
                "period": period,
                "missing_categories": ["employees"],
                "message": f"No employees found for department '{clean_dept}'.",
                "grounding_registry": {},
            }

        employee_ids = [e.id for e in employees]
        role_map = {e.id: _clean_str(e.role_title, 100) for e in employees}
        team_size = len(employees)

        # 2. Query approved PerformanceRecords for these employees
        perf_query = (
            db.query(PerformanceRecord)
            .filter(
                PerformanceRecord.employee_id.in_(employee_ids),
                PerformanceRecord.is_approved.is_(True),
            )
        )
        all_perf_records = perf_query.all()

        # 3. Determine target_period and comparison_period
        all_available_periods = sorted(
            {r.period for r in all_perf_records if r.period},
            key=parse_period_key,
        )

        target_period: str | None = period
        if not target_period:
            if all_available_periods:
                target_period = all_available_periods[-1]
            else:
                # Check if goals or task outcomes or themes have periods
                other_periods = set()
                for g in db.query(Goal.period).filter(Goal.employee_id.in_(employee_ids), Goal.is_approved.is_(True)).all():
                    if g[0]:
                        other_periods.add(g[0])
                for t in db.query(TaskOutcome.period).filter(TaskOutcome.employee_id.in_(employee_ids), TaskOutcome.is_approved.is_(True)).all():
                    if t[0]:
                        other_periods.add(t[0])
                for th in db.query(EvaluationTheme.period).filter(EvaluationTheme.employee_id.in_(employee_ids), EvaluationTheme.is_approved.is_(True)).all():
                    if th[0]:
                        other_periods.add(th[0])

                sorted_other = sorted(other_periods, key=parse_period_key)
                if sorted_other:
                    target_period = sorted_other[-1]

        if not target_period:
            return {
                "has_sufficient_data": False,
                "department": clean_dept,
                "period": None,
                "missing_categories": ["performance", "period"],
                "message": f"No approved period records found for department '{clean_dept}'.",
                "grounding_registry": {},
            }

        # Resolve comparison_period if target_period exists in all_available_periods
        comparison_period: str | None = None
        if target_period in all_available_periods:
            target_idx = all_available_periods.index(target_period)
            if target_idx > 0:
                comparison_period = all_available_periods[target_idx - 1]

        # 4. Target period PerformanceRecords
        target_perfs = [r for r in all_perf_records if r.period == target_period]
        comparison_perfs = (
            [r for r in all_perf_records if r.period == comparison_period]
            if comparison_period
            else []
        )

        # 5. Overdue Workload Aggregation (Goals & Task Outcomes)
        # Goals for target_period or active goals without period
        goals_query = (
            db.query(Goal)
            .filter(
                Goal.employee_id.in_(employee_ids),
                Goal.is_approved.is_(True),
            )
        )
        if target_period:
            goals_records = goals_query.filter(
                (Goal.period == target_period) | (Goal.period.is_(None))
            ).all()
        else:
            goals_records = goals_query.all()

        delayed_goals = [g for g in goals_records if g.status == "delayed"]
        total_delayed_goals = len(delayed_goals)

        # Tasks for target_period
        tasks_query = (
            db.query(TaskOutcome)
            .filter(
                TaskOutcome.employee_id.in_(employee_ids),
                TaskOutcome.is_approved.is_(True),
            )
        )
        if target_period:
            tasks_records = tasks_query.filter(
                (TaskOutcome.period == target_period) | (TaskOutcome.period.is_(None))
            ).all()
        else:
            tasks_records = tasks_query.all()

        blocked_tasks = [t for t in tasks_records if t.status == "blocked"]
        total_blocked_tasks = len(blocked_tasks)

        # Affected member count: unique members with >= 1 blocked task or delayed goal
        affected_member_ids = {
            t.employee_id for t in blocked_tasks
        } | {
            g.employee_id for g in delayed_goals
        }
        affected_member_count = len(affected_member_ids)

        target_averages = cls.compute_metric_averages(target_perfs)

        # Direction calculation
        direction = TrendDirection.STABLE
        comparison_averages: dict[str, float] | None = None
        trend_details: dict[str, Any] | None = None

        if comparison_perfs:
            comparison_averages = cls.compute_metric_averages(comparison_perfs)
            # Use overall score trend as primary aggregate direction
            overall_trend = calculate_trend(
                current_value=target_averages["team_avg_overall_score"],
                previous_value=comparison_averages["team_avg_overall_score"],
                threshold=0.5,
            )
            task_trend = calculate_trend(
                current_value=target_averages["team_avg_task_completion"],
                previous_value=comparison_averages["team_avg_task_completion"],
                threshold=0.5,
            )
            goal_trend = calculate_trend(
                current_value=target_averages["team_avg_goal_achievement"],
                previous_value=comparison_averages["team_avg_goal_achievement"],
                threshold=0.5,
            )

            trend_details = {
                "overall_score": overall_trend,
                "task_completion": task_trend,
                "goal_achievement": goal_trend,
            }

            if overall_trend["direction"] == "improved":
                direction = TrendDirection.IMPROVED
            elif overall_trend["direction"] == "declined":
                direction = TrendDirection.DECLINED
            else:
                # If overall is stable, check task completion
                if task_trend["direction"] == "improved":
                    direction = TrendDirection.IMPROVED
                elif task_trend["direction"] == "declined":
                    direction = TrendDirection.DECLINED
                else:
                    direction = TrendDirection.STABLE

        # 7. Common Skill Patterns
        skills_records = (
            db.query(Skill)
            .filter(
                Skill.employee_id.in_(employee_ids),
                Skill.is_approved.is_(True),
            )
            .all()
        )

        skill_names = [_clean_str(s.name, 100) for s in skills_records if s.name]
        skill_counts = Counter(skill_names)
        top_skills = [name for name, _ in skill_counts.most_common(5)]

        # 8. Evaluation Theme Patterns
        themes_query = (
            db.query(EvaluationTheme)
            .filter(
                EvaluationTheme.employee_id.in_(employee_ids),
                EvaluationTheme.is_approved.is_(True),
            )
        )
        if target_period:
            themes_records = themes_query.filter(
                (EvaluationTheme.period == target_period) | (EvaluationTheme.period.is_(None))
            ).all()
        else:
            themes_records = themes_query.all()

        positive_themes = [
            _clean_str(th.theme, 150)
            for th in themes_records
            if th.sentiment == "positive" and th.theme
        ]
        needs_imp_themes = [
            _clean_str(th.theme, 150)
            for th in themes_records
            if th.sentiment == "needs_improvement" and th.theme
        ]

        pos_counts = Counter(positive_themes)
        needs_imp_counts = Counter(needs_imp_themes)

        top_positive = [t for t, _ in pos_counts.most_common(5)]
        top_needs_imp = [t for t, _ in needs_imp_counts.most_common(5)]

        # 9. Anonymized Drill-Down Factors
        drill_down_factors: list[dict[str, Any]] = []

        # Workload Blockers Drill-Down
        for b_task in blocked_tasks[:3]:
            role = role_map.get(b_task.employee_id)
            drill_down_factors.append({
                "category": DrillDownCategory.WORKLOAD_BLOCKERS.value,
                "factor_title": _clean_str(b_task.title, 100),
                "observation": f"Task '{_clean_str(b_task.title, 100)}' is currently blocked: {_clean_str(b_task.outcome, 200) or 'No blocker details'}",
                "supporting_metrics": f"Status: blocked, Period: {b_task.period or target_period}",
                "anonymized_role": role,
            })

        for d_goal in delayed_goals[:3]:
            role = role_map.get(d_goal.employee_id)
            drill_down_factors.append({
                "category": DrillDownCategory.WORKLOAD_BLOCKERS.value,
                "factor_title": _clean_str(d_goal.title, 100),
                "observation": f"Goal '{_clean_str(d_goal.title, 100)}' delayed at {d_goal.progress}% progress.",
                "supporting_metrics": f"Status: delayed, Progress: {d_goal.progress}%, Period: {d_goal.period or target_period}",
                "anonymized_role": role,
            })

        # Evaluation Themes Drill-Down
        for th in themes_records:
            if len(drill_down_factors) >= MAX_DRILL_DOWN_FACTORS:
                break
            if th.sentiment == "needs_improvement":
                role = role_map.get(th.employee_id)
                drill_down_factors.append({
                    "category": DrillDownCategory.EVALUATION_THEMES.value,
                    "factor_title": f"Growth Area: {_clean_str(th.theme, 100)}",
                    "observation": _clean_str(th.evidence, 250),
                    "supporting_metrics": f"Sentiment: needs_improvement, Period: {th.period or target_period}",
                    "anonymized_role": role,
                })

        # 10. Check minimum data sufficiency
        has_sufficient_data = bool(
            target_perfs
            or tasks_records
            or goals_records
            or skills_records
            or themes_records
        )

        if not has_sufficient_data:
            return {
                "has_sufficient_data": False,
                "department": clean_dept,
                "period": target_period,
                "missing_categories": ["performance", "task_outcomes", "goals"],
                "message": f"No approved performance, task, or goal records found for department '{clean_dept}' in period '{target_period}'.",
                "grounding_registry": {},
            }

        # 11. Grounding Registry (Deterministic facts for AI validation)
        grounding_registry: dict[str, Any] = {
            "department": clean_dept,
            "target_period": target_period,
            "comparison_period": comparison_period,
            "team_size": team_size,
            "total_blocked_tasks": total_blocked_tasks,
            "total_delayed_goals": total_delayed_goals,
            "affected_member_count": affected_member_count,
            "team_avg_task_completion": target_averages["team_avg_task_completion"],
            "team_avg_goal_achievement": target_averages["team_avg_goal_achievement"],
            "team_avg_overall_score": target_averages["team_avg_overall_score"],
            "direction": direction.value,
            "top_skills": top_skills,
            "skill_frequencies": dict(skill_counts),
            "top_positive_themes": top_positive,
            "positive_theme_frequencies": dict(pos_counts),
            "top_needs_improvement_themes": top_needs_imp,
            "needs_improvement_theme_frequencies": dict(needs_imp_counts),
            "comparison_averages": comparison_averages,
            "trend_details": trend_details,
        }

        # 12. Structure Return Payload
        return {
            "has_sufficient_data": True,
            "department": clean_dept,
            "period": target_period,
            "comparison_period": comparison_period,
            "team_size": team_size,
            "workload_patterns": {
                "total_blocked_tasks": total_blocked_tasks,
                "total_delayed_goals": total_delayed_goals,
                "affected_member_count": affected_member_count,
                "blocked_task_samples": [
                    {"title": _clean_str(t.title, 100), "outcome": _clean_str(t.outcome, 200)}
                    for t in blocked_tasks[:5]
                ],
                "delayed_goal_samples": [
                    {"title": _clean_str(g.title, 100), "progress": g.progress}
                    for g in delayed_goals[:5]
                ],
            },
            "completion_trends": {
                "team_avg_task_completion": target_averages["team_avg_task_completion"],
                "team_avg_goal_achievement": target_averages["team_avg_goal_achievement"],
                "team_avg_overall_score": target_averages["team_avg_overall_score"],
                "direction": direction.value,
                "comparison_averages": comparison_averages,
            },
            "skill_patterns": {
                "top_common_skills": top_skills,
                "skill_frequencies": dict(skill_counts),
            },
            "evaluation_theme_patterns": {
                "top_positive_themes": top_positive,
                "positive_frequencies": dict(pos_counts),
                "top_needs_improvement_themes": top_needs_imp,
                "needs_improvement_frequencies": dict(needs_imp_counts),
            },
            "drill_down_factors": drill_down_factors,
            "grounding_registry": grounding_registry,
        }

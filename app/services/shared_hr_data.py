"""Read-only access helpers for the HR team's Laravel-owned schema.

The AI service keeps its legacy SQLite/test models for local tests, but the
shared Railway database has a different, Laravel-owned schema.  This module
is the small compatibility boundary for that schema.  It never creates,
updates, or deletes HR records.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from sqlalchemy import inspect, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session


@dataclass(frozen=True)
class SharedEmployee:
    """Employee identity exposed to the AI service using the public HR code."""

    id: str
    role_title: str
    department: str
    user_id: int


def is_shared_hr_schema(bind: Any) -> bool:
    """Return true only when the Laravel employee schema is present."""
    try:
        inspector = inspect(bind)
        tables = set(inspector.get_table_names())
        if not {"employees", "users", "departments"}.issubset(tables):
            return False
        employee_columns = {column["name"] for column in inspector.get_columns("employees")}
        return {"id", "employee_id", "user_id", "department_id"}.issubset(employee_columns)
    except (SQLAlchemyError, OSError):
        return False


def get_shared_employee(db: Session, external_id: str) -> SharedEmployee | None:
    """Resolve the public employee code against Laravel users/employees.

    The HR application can have a user before an optional row exists in
    ``employees``.  ``users.employee_id`` is also unique and is the identifier
    shown by the HR UI, so it is a valid fallback for authentication and
    employee-scoped reads.
    """
    row = db.execute(
        text("""
            SELECT COALESCE(e.employee_id, u.employee_id) AS employee_id,
                   COALESCE(e.job_title, u.job_title) AS job_title,
                   d.name AS department,
                   u.id AS user_id
            FROM users AS u
            LEFT JOIN employees AS e ON e.user_id = u.id AND e.deleted_at IS NULL
            LEFT JOIN departments AS d
              ON d.id = COALESCE(e.department_id, u.department_id)
             AND d.deleted_at IS NULL
            WHERE u.employee_id = :employee_id
              AND u.deleted_at IS NULL
            LIMIT 1
        """),
        {"employee_id": external_id.strip()},
    ).mappings().first()
    if row is None:
        return None
    return SharedEmployee(
        id=str(row["employee_id"]),
        role_title=str(row["job_title"] or ""),
        department=str(row["department"] or ""),
        user_id=int(row["user_id"]),
    )


def build_shared_career_context(
    db: Session,
    employee: SharedEmployee,
    period: str | None,
    limits: dict[str, int],
) -> dict[str, Any]:
    """Build a grounded context from existing Laravel HR records only."""
    period_like = None
    if period and "-Q" in period:
        year, quarter = period.split("-Q", 1)
        period_like = f"%Q{quarter} {year}%"

    params: dict[str, Any] = {
        "user_id": employee.user_id,
        "period_like": period_like,
    }
    period_filter = "AND (ep.name = :period OR ep.name LIKE :period_like)" if period else ""
    if period:
        params["period"] = period

    performance_rows = db.execute(
        text(f"""
            SELECT ev.id, ep.name AS period,
                   COALESCE(ev.overall_score, AVG(es.score)) AS overall_score,
                   ev.feedback
            FROM evaluations AS ev
            LEFT JOIN evaluation_periods AS ep ON ep.id = ev.period_id
            LEFT JOIN evaluation_scores AS es ON es.evaluation_id = ev.id
            WHERE ev.user_id = :user_id AND ev.status = 'completed' {period_filter}
            GROUP BY ev.id, ep.name, ev.overall_score, ev.feedback, ev.created_at
            ORDER BY ev.created_at DESC, ev.id DESC
            LIMIT {int(limits['performance'])}
        """),
        params,
    ).mappings().all()

    metric_row = db.execute(
        text("""
            SELECT
                (SELECT AVG(t.progress)
                 FROM task_assignments AS ta
                 INNER JOIN tasks AS t ON t.id = ta.task_id
                 WHERE ta.user_id = :user_id) AS task_completion_rate,
                (SELECT 100.0 * AVG(CASE WHEN g.status = 'completed' THEN 1 ELSE 0 END)
                 FROM goals AS g WHERE g.user_id = :user_id) AS goal_achievement_rate,
                (SELECT 100.0 * AVG(CASE WHEN a.status = 'Present' THEN 1 ELSE 0 END)
                 FROM attendances AS a WHERE a.user_id = :user_id) AS attendance_rate
        """),
        {"user_id": employee.user_id},
    ).mappings().one()

    goal_rows = db.execute(
        text(f"""
            SELECT g.id, g.title, g.description, g.status, g.target_date
            FROM goals AS g
            WHERE g.user_id = :user_id AND g.status IN ('active', 'completed')
            ORDER BY g.updated_at DESC, g.id DESC
            LIMIT {int(limits['goals'])}
        """),
        {"user_id": employee.user_id},
    ).mappings().all()

    task_rows = db.execute(
        text(f"""
            SELECT t.id, t.title, t.status, t.description, t.progress, t.deadline
            FROM task_assignments AS ta
            INNER JOIN tasks AS t ON t.id = ta.task_id
            WHERE ta.user_id = :user_id
            ORDER BY t.updated_at DESC, t.id DESC
            LIMIT {int(limits['task_outcomes'])}
        """),
        {"user_id": employee.user_id},
    ).mappings().all()

    theme_rows = db.execute(
        text(f"""
            SELECT es.id, ec.name AS theme, es.score, ev.feedback,
                   ep.name AS period, ee.description AS evidence
            FROM evaluation_scores AS es
            INNER JOIN evaluations AS ev ON ev.id = es.evaluation_id
            INNER JOIN evaluation_categories AS ec ON ec.id = es.category_id
            LEFT JOIN evaluation_periods AS ep ON ep.id = ev.period_id
            LEFT JOIN evaluation_evidence AS ee ON ee.evaluation_id = ev.id
            WHERE ev.user_id = :user_id AND ev.status = 'completed' {period_filter}
            ORDER BY ev.created_at DESC, es.id DESC
            LIMIT {int(limits['evaluation_themes'])}
        """),
        params,
    ).mappings().all()

    def normalize_period(value: Any) -> str:
        """Normalize Laravel labels such as '[TEST] Q4 2026 Review' to 2026-Q4."""
        raw = str(value or "").strip()
        match = re.search(r"Q([1-4])\s+(\d{4})", raw, re.IGNORECASE)
        return f"{match.group(2)}-Q{match.group(1)}" if match else raw

    normalized_performance = [
        {
            "id": int(row["id"]),
            "source_type": "performance",
            "period": normalize_period(row["period"]),
            "overall_score": float(row["overall_score"]),
            "task_completion_rate": float(metric_row["task_completion_rate"] or 0),
            "goal_achievement_rate": float(metric_row["goal_achievement_rate"] or 0),
            "attendance_rate": float(metric_row["attendance_rate"] or 0),
            "feedback": str(row["feedback"] or ""),
        }
        for row in performance_rows
        if row["overall_score"] is not None
    ]

    normalized_themes = [
        {
            "id": int(row["id"]),
            "source_type": "evaluation_theme",
            "theme": str(row["theme"] or ""),
            "sentiment": "",
            "evidence": str(row["evidence"] or row["feedback"] or ""),
            "period": normalize_period(row["period"]),
            "score": float(row["score"]),
        }
        for row in theme_rows
    ]

    context = {
        "employee": {
            "id": employee.id,
            "role_title": employee.role_title,
            "department": employee.department,
        },
        "performance": normalized_performance,
        "goals": [
            {
                "id": int(row["id"]),
                "source_type": "goal",
                "title": str(row["title"] or ""),
                "description": str(row["description"] or ""),
                "status": str(row["status"] or ""),
                "deadline": str(row["target_date"] or ""),
                "period": "",
            }
            for row in goal_rows
        ],
        # The Laravel schema has no standalone skills table. Evaluation
        # categories are the authoritative competency records, so expose the
        # same scored category rows as factual skill evidence.
        "skills": [
            {
                "id": item["id"],
                "source_type": "skill",
                "name": item["theme"],
                "level": f"score {item['score']}",
                "evidence": item["evidence"],
                "period": item["period"],
            }
            for item in normalized_themes
        ],
        "task_outcomes": [
            {
                "id": int(row["id"]),
                "source_type": "task_outcome",
                "title": str(row["title"] or ""),
                "status": str(row["status"] or ""),
                "outcome": str(row["description"] or ""),
                "completion_date": str(row["deadline"] or ""),
                "period": "",
                "progress": int(row["progress"] or 0),
            }
            for row in task_rows
        ],
        "evaluation_themes": normalized_themes,
    }
    missing = [category for category in ("performance", "goals", "skills", "task_outcomes", "evaluation_themes") if not context[category]]
    return {
        "has_sufficient_data": not missing,
        "missing_categories": missing,
        "context": context,
        "approved_sources": _approved_sources(context),
        "selected_source_ids": {
            category: [record["id"] for record in records]
            for category, records in context.items()
            if category != "employee"
        },
    }


def build_shared_performance_context(
    db: Session,
    employee: SharedEmployee,
    period: str | None = None,
    limit: int = 10,
    min_periods: int = 1,
) -> dict[str, Any]:
    """Build performance metrics from Laravel evaluations, not AI-owned tables.

    Laravel stores the employee relationship on ``evaluations.user_id`` and
    stores the period label in ``evaluation_periods.name``.  ``overall_score``
    is preferred; when it is null, the weighted category scores are normalized
    to 0..100 using ``evaluation_categories.max_score`` and ``weight``.
    """
    rows = db.execute(
        text(f"""
            SELECT ev.id, ep.name AS period, ev.overall_score,
                   COALESCE(
                       ev.overall_score,
                       100.0 * SUM(es.score * ec.weight / NULLIF(ec.max_score, 0))
                       / NULLIF(SUM(ec.weight), 0)
                   ) AS calculated_score
            FROM evaluations AS ev
            INNER JOIN evaluation_periods AS ep ON ep.id = ev.period_id
            LEFT JOIN evaluation_scores AS es ON es.evaluation_id = ev.id
            LEFT JOIN evaluation_categories AS ec ON ec.id = es.category_id
            WHERE ev.user_id = :user_id
              AND LOWER(ev.status) IN ('completed', 'approved', 'finalized')
            GROUP BY ev.id, ep.name, ev.overall_score, ev.created_at
            ORDER BY ep.end_date ASC, ev.created_at ASC, ev.id ASC
            LIMIT {int(limit)}
        """),
        {"user_id": employee.user_id},
    ).mappings().all()

    def normalize_period(value: Any) -> str:
        raw = str(value or "").strip()
        match = re.search(r"Q([1-4])\s+(\d{4})", raw, re.IGNORECASE)
        return f"{match.group(2)}-Q{match.group(1)}" if match else raw

    records = []
    for row in rows:
        normalized = normalize_period(row["period"])
        score = row["overall_score"] if row["overall_score"] is not None else row["calculated_score"]
        if score is None:
            continue
        records.append({
            "id": int(row["id"]),
            "period": normalized,
            "overall_score": round(float(score), 2),
            "task_completion_rate": 0.0,
            "goal_achievement_rate": 0.0,
            "attendance_rate": 0.0,
        })

    if period:
        requested = normalize_period(period)
        target = [record for record in records if record["period"] == requested]
        if not target:
            return {
                "employee": {"id": employee.id, "role_title": employee.role_title, "department": employee.department},
                "has_sufficient_data": False, "has_trend_data": False,
                "reason": f"No approved performance record found for period '{period}'.",
                "target_period": period, "comparison_period": None,
                "facts": {"records_count": len(records), "periods": [r["period"] for r in records], "metrics_by_period": records},
                "metrics_by_period": records, "calculated_trends": {"comparison_available": False, "from_period": None, "to_period": None, "metric_trends": {}, "period_over_period": []}, "trends": {},
            }

    previous = records[-2] if len(records) >= 2 else None
    target = next((r for r in records if r["period"] == normalize_period(period)), records[-1] if records else None) if period else (records[-1] if records else None)
    has_trend = target is not None and previous is not None and (not period or records.index(target) > 0)
    reason = None if len(records) >= min_periods and has_trend else (
        f"Insufficient performance records (minimum {min_periods} required, found {len(records)})." if len(records) < min_periods
        else "Only one approved performance period available; cross-period comparative trend requires at least two periods."
    )
    from app.services.performance_insight_context import METRIC_FIELDS, calculate_trend

    metric_trends: dict[str, dict[str, Any]] = {}
    if has_trend and target and previous:
        for metric in METRIC_FIELDS:
            metric_trends[metric] = calculate_trend(
                current_value=target.get(metric, 0.0),
                previous_value=previous.get(metric, 0.0),
                threshold=0.0,
            )

    period_over_period: list[dict[str, Any]] = []
    if len(records) >= 2:
        for i in range(len(records) - 1):
            p_from = records[i]
            p_to = records[i + 1]
            step_trends = {
                m: calculate_trend(
                    current_value=p_to.get(m, 0.0),
                    previous_value=p_from.get(m, 0.0),
                    threshold=0.0,
                )
                for m in METRIC_FIELDS
            }
            period_over_period.append({
                "from_period": p_from["period"],
                "to_period": p_to["period"],
                "metric_trends": step_trends,
            })

    return {
        "employee": {"id": employee.id, "role_title": employee.role_title, "department": employee.department},
        "has_sufficient_data": len(records) >= min_periods,
        "has_trend_data": has_trend,
        "reason": reason,
        "target_period": target["period"] if target else period,
        "comparison_period": previous["period"] if previous else None,
        "facts": {"records_count": len(records), "periods": [r["period"] for r in records], "metrics_by_period": records},
        "metrics_by_period": records,
        "calculated_trends": {
            "comparison_available": has_trend,
            "from_period": previous["period"] if previous else None,
            "to_period": target["period"] if target else period,
            "metric_trends": metric_trends,
            "period_over_period": period_over_period,
        },
        "trends": metric_trends,
    }


def build_shared_team_context(db: Session, department: str, period: str | None = None) -> dict[str, Any]:
    """Build the manager-scoped team shape from the Laravel tables."""
    users = db.execute(text("""
        SELECT u.id AS user_id, u.employee_id, u.job_title
        FROM users AS u
        LEFT JOIN departments AS d ON d.id = u.department_id
        WHERE u.deleted_at IS NULL AND u.status = 'active'
          AND COALESCE(d.name, '') = :department
    """), {"department": department.strip()}).mappings().all()
    if not users:
        return {"has_sufficient_data": False, "department": department, "period": period,
                "missing_categories": ["employees"], "message": f"No employees found for department '{department}'.",
                "grounding_registry": {}}

    performance = []
    for user in users:
        employee = SharedEmployee(str(user["employee_id"]), str(user["job_title"] or ""), department, int(user["user_id"]))
        performance.extend(build_shared_performance_context(db, employee, period, limit=10, min_periods=1)["metrics_by_period"])
    periods = sorted({p["period"] for p in performance})
    target_period = period or (periods[-1] if periods else None)
    target = [p for p in performance if p["period"] == target_period]
    goals = db.execute(text("""
        SELECT COUNT(*) AS total, SUM(LOWER(status) = 'completed') AS completed
        FROM goals WHERE user_id IN (SELECT id FROM users WHERE department_id = (SELECT id FROM departments WHERE name = :department LIMIT 1))
    """), {"department": department.strip()}).mappings().one()
    tasks = db.execute(text("""
        SELECT COUNT(*) AS total, SUM(LOWER(t.status) IN ('completed', 'closed')) AS completed
        FROM task_assignments AS ta INNER JOIN tasks AS t ON t.id = ta.task_id
        WHERE ta.user_id IN (SELECT id FROM users WHERE department_id = (SELECT id FROM departments WHERE name = :department LIMIT 1))
    """), {"department": department.strip()}).mappings().one()
    if not target and not goals["total"] and not tasks["total"]:
        return {"has_sufficient_data": False, "department": department, "period": target_period,
                "missing_categories": ["performance", "goals", "task_outcomes"],
                "message": f"No approved performance, task, or goal records found for department '{department}'.",
                "grounding_registry": {}}
    avg = lambda key: round(sum(float(r[key]) for r in target) / len(target), 2) if target else 0.0
    comparison = [p for p in performance if p["period"] == periods[-2]] if len(periods) > 1 and target_period == periods[-1] else []
    current_score = avg("overall_score")
    previous_score = round(sum(float(r["overall_score"]) for r in comparison) / len(comparison), 2) if comparison else None
    direction = "improved" if previous_score is not None and current_score > previous_score else "declined" if previous_score is not None and current_score < previous_score else "stable"
    registry = {"department": department, "target_period": target_period, "comparison_period": periods[-2] if len(periods) > 1 else None,
                "team_size": len(users), "team_avg_task_completion": avg("task_completion_rate"),
                "team_avg_goal_achievement": avg("goal_achievement_rate"), "team_avg_overall_score": avg("overall_score"),
                "total_blocked_tasks": 0, "total_delayed_goals": 0, "affected_member_count": 0,
                "direction": direction, "top_skills": [], "skill_frequencies": {}, "top_positive_themes": [], "positive_theme_frequencies": {},
                "top_needs_improvement_themes": [], "needs_improvement_theme_frequencies": {}}
    return {"has_sufficient_data": True, "department": department, "period": target_period,
            "comparison_period": registry["comparison_period"], "team_size": len(users),
            "workload_patterns": {"total_blocked_tasks": 0, "total_delayed_goals": 0, "affected_member_count": 0,
                                   "blocked_task_samples": [], "delayed_goal_samples": []},
            "completion_trends": {**{k: registry[k] for k in ("team_avg_task_completion", "team_avg_goal_achievement", "team_avg_overall_score")}, "direction": direction, "comparison_averages": None},
            "skill_patterns": {"top_common_skills": [], "skill_frequencies": {}},
            "evaluation_theme_patterns": {"top_positive_themes": [], "positive_frequencies": [], "top_needs_improvement_themes": [], "needs_improvement_frequencies": []},
            "drill_down_factors": [], "grounding_registry": registry}


def build_shared_attention_context(
    db: Session, employee: SharedEmployee, period: str | None = None
) -> dict[str, Any]:
    """Build Attention Signal input from evaluations and assigned HR work."""
    data = build_shared_career_context(
        db, employee, period,
        {"performance": 10, "goals": 10, "task_outcomes": 15, "evaluation_themes": 10},
    )
    context = data["context"]
    performance = context["performance"]
    target = performance[-1] if performance else None
    comparison = performance[-2] if len(performance) > 1 else None
    if period:
        target = next((row for row in performance if row["period"] == period), None)
        if target is not None:
            index = performance.index(target)
            comparison = performance[index - 1] if index > 0 else None
    if not target and not context["goals"] and not context["task_outcomes"] and not context["evaluation_themes"]:
        return {"employee": {"id": employee.id, "role_title": employee.role_title, "department": employee.department},
                "has_sufficient_data": False, "reason": f"No approved metrics or activity records found for period '{period}'.",
                "target_period": period, "comparison_period": None, "approved_sources": {}}
    def metrics(row):
        return {key: row.get(key) for key in ("id", "period", "overall_score", "task_completion_rate", "goal_achievement_rate", "attendance_rate")} if row else None
    tasks = context["task_outcomes"]
    goals = context["goals"]
    task_summary = {"total": len(tasks), "completed": sum(str(t["status"]).lower() in ("completed", "closed") for t in tasks),
                    "blocked": sum(str(t["status"]).lower() == "blocked" for t in tasks), "in_progress": sum(str(t["status"]).lower() in ("in progress", "in_progress") for t in tasks), "blocked_tasks": [t for t in tasks if str(t["status"]).lower() == "blocked"]}
    goal_summary = {"total": len(goals), "completed": sum(str(g["status"]).lower() == "completed" for g in goals),
                    "delayed": sum(str(g["status"]).lower() == "delayed" for g in goals), "in_progress": sum(str(g["status"]).lower() in ("active", "in_progress") for g in goals), "delayed_goals": [g for g in goals if str(g["status"]).lower() == "delayed"]}
    target_metrics = metrics(target)
    score = target_metrics.get("overall_score") if target_metrics else None
    attention_level = "High" if score is not None and score < 70 else "Medium" if score is not None and score < 85 else "Low"
    indicators = []
    if target_metrics and score is not None:
        indicators.append({
            "indicator_name": "Overall Evaluation Score", "category": "evaluation_trend",
            "current_value": score, "previous_value": comparison.get("overall_score") if comparison else None,
            "change_description": "Observed evaluation score for the requested period.",
            "evidence": f"Completed evaluation score for {target_metrics['period']} is {score}.",
        })
    if task_summary["blocked"]:
        indicators.append({"indicator_name": "Blocked Tasks", "category": "task_completion", "current_value": task_summary["blocked"], "previous_value": None, "change_description": "Assigned tasks are marked blocked.", "evidence": "Assigned task records contain blocked work."})
    if goal_summary["delayed"]:
        indicators.append({"indicator_name": "Delayed Goals", "category": "goals", "current_value": goal_summary["delayed"], "previous_value": None, "change_description": "Goals are marked delayed.", "evidence": "Goal records contain delayed goals."})
    return {"employee": {"id": employee.id, "role_title": employee.role_title, "department": employee.department},
            "has_sufficient_data": True, "reason": None, "target_period": target["period"] if target else period,
            "comparison_period": comparison["period"] if comparison else None, "assessment_type": "comparison_based" if comparison else "current_period_only",
            "attention_level": attention_level, "metric_bands": {}, "indicators": indicators, "metric_trends": {},
            "target_metrics": target_metrics, "comparison_metrics": metrics(comparison), "task_summary": task_summary,
            "goal_summary": goal_summary, "evaluation_summary": {"total": len(context["evaluation_themes"]), "needs_improvement_count": 0, "positive_count": 0, "themes": context["evaluation_themes"]},
            "approved_sources": data["approved_sources"]}


def _approved_sources(context: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    return {
        (record["source_type"], record["id"]): record
        for category, records in context.items()
        if category != "employee"
        for record in records
    }

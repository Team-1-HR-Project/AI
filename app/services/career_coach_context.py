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

REQUIRED_CATEGORIES = [
    "performance",
    "goals",
    "skills",
    "task_outcomes",
    "evaluation_themes",
]

# Context Budget Limits
MAX_PERFORMANCE_RECORDS = 5
MAX_GOALS = 8
MAX_SKILLS = 12
MAX_TASK_OUTCOMES = 8
MAX_EVALUATION_THEMES = 8
MAX_FIELD_CHARS = 300
MAX_TOTAL_CONTEXT_CHARS = 12000


def _clean_str(val: Any, max_chars: int = MAX_FIELD_CHARS) -> str:
    if val is None:
        return ""
    text = str(val).strip()
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text


class CareerCoachContextBuilder:
    @staticmethod
    def build_context(
        db: Session,
        employee_id: str,
        period: str | None = None,
    ) -> dict[str, Any]:
        """
        Gathers and prepares sanitized employee data for the AI Career Coach.
        - Guarantees strict employee data isolation.
        - Enforces approved-data only (P0-4).
        - Enforces deterministic context budget and record caps (P1-2).
        - Exposes source IDs and canonical types for grounding validation (P0-3).
        - Detects missing data categories without fabricating facts.
        """
        employee = db.query(Employee).filter(Employee.id == employee_id).first()
        if not employee:
            return {
                "has_sufficient_data": False,
                "missing_categories": REQUIRED_CATEGORIES.copy(),
                "error": f"Employee with id '{employee_id}' not found.",
                "context": None,
                "approved_sources": {},
                "selected_source_ids": {},
            }

        # 1. Employee baseline info (safe fields only)
        employee_dict = {
            "id": employee.id,
            "role_title": _clean_str(employee.role_title, 100),
            "department": _clean_str(employee.department, 100),
        }

        # 2. Performance records (strictly isolated to employee.id, approved only)
        perf_query = db.query(PerformanceRecord).filter(
            PerformanceRecord.employee_id == employee.id,
            PerformanceRecord.is_approved.is_(True),
        )
        if period:
            perf_query = perf_query.filter(PerformanceRecord.period == period)
        perf_records = (
            perf_query.order_by(PerformanceRecord.period.desc(), PerformanceRecord.created_at.desc())
            .limit(MAX_PERFORMANCE_RECORDS)
            .all()
        )

        performance_data: list[dict[str, Any]] = [
            {
                "id": p.id,
                "source_type": "performance",
                "period": p.period,
                "overall_score": float(p.overall_score),
                "task_completion_rate": float(p.task_completion_rate),
                "goal_achievement_rate": float(p.goal_achievement_rate),
                "attendance_rate": float(p.attendance_rate),
            }
            for p in perf_records
        ]

        # 3. Goals (strictly isolated to employee.id, approved only)
        goal_query = db.query(Goal).filter(
            Goal.employee_id == employee.id,
            Goal.is_approved.is_(True),
        )
        if period:
            goal_query = goal_query.filter((Goal.period == period) | (Goal.period.is_(None)))
        goals = (
            goal_query.order_by(
                Goal.period == period if period else Goal.created_at.desc(),
                Goal.created_at.desc(),
            )
            .limit(MAX_GOALS)
            .all()
        )

        goals_data: list[dict[str, Any]] = [
            {
                "id": g.id,
                "source_type": "goal",
                "title": _clean_str(g.title),
                "progress": float(g.progress) if g.progress is not None else 0.0,
                "status": _clean_str(g.status, 50),
                "deadline": _clean_str(g.deadline, 50),
                "period": _clean_str(g.period, 20),
            }
            for g in goals
        ]

        # 4. Skills (strictly isolated to employee.id, approved only)
        skills = (
            db.query(Skill)
            .filter(
                Skill.employee_id == employee.id,
                Skill.is_approved.is_(True),
            )
            .order_by(Skill.created_at.desc())
            .limit(MAX_SKILLS)
            .all()
        )

        skills_data: list[dict[str, Any]] = [
            {
                "id": s.id,
                "source_type": "skill",
                "name": _clean_str(s.name, 100),
                "level": _clean_str(s.level, 50),
                "evidence": _clean_str(s.evidence),
            }
            for s in skills
        ]

        # 5. Task Outcomes (strictly isolated to employee.id, approved only)
        task_query = db.query(TaskOutcome).filter(
            TaskOutcome.employee_id == employee.id,
            TaskOutcome.is_approved.is_(True),
        )
        if period:
            task_query = task_query.filter((TaskOutcome.period == period) | (TaskOutcome.period.is_(None)))
        tasks = (
            task_query.order_by(
                TaskOutcome.period == period if period else TaskOutcome.created_at.desc(),
                TaskOutcome.created_at.desc(),
            )
            .limit(MAX_TASK_OUTCOMES)
            .all()
        )

        task_outcomes_data: list[dict[str, Any]] = [
            {
                "id": t.id,
                "source_type": "task_outcome",
                "title": _clean_str(t.title),
                "status": _clean_str(t.status, 50),
                "outcome": _clean_str(t.outcome),
                "completion_date": _clean_str(t.completion_date, 50),
                "period": _clean_str(t.period, 20),
            }
            for t in tasks
        ]

        # 6. Evaluation Themes (strictly isolated to employee.id, approved only)
        theme_query = db.query(EvaluationTheme).filter(
            EvaluationTheme.employee_id == employee.id,
            EvaluationTheme.is_approved.is_(True),
        )
        if period:
            theme_query = theme_query.filter(
                (EvaluationTheme.period == period) | (EvaluationTheme.period.is_(None))
            )
        themes = (
            theme_query.order_by(
                EvaluationTheme.period == period if period else EvaluationTheme.created_at.desc(),
                EvaluationTheme.created_at.desc(),
            )
            .limit(MAX_EVALUATION_THEMES)
            .all()
        )

        evaluation_themes_data: list[dict[str, Any]] = [
            {
                "id": et.id,
                "source_type": "evaluation_theme",
                "theme": _clean_str(et.theme, 150),
                "sentiment": _clean_str(et.sentiment, 50),
                "evidence": _clean_str(et.evidence),
                "period": _clean_str(et.period, 20),
            }
            for et in themes
        ]

        # Assemble approved sources mapping and selected IDs for auditability & grounding
        approved_sources: dict[tuple[str, int], dict[str, Any]] = {}
        selected_source_ids: dict[str, list[int]] = {
            "performance": [],
            "goal": [],
            "skill": [],
            "task_outcome": [],
            "evaluation_theme": [],
        }

        for p_item in performance_data:
            approved_sources[("performance", p_item["id"])] = p_item
            selected_source_ids["performance"].append(p_item["id"])

        for g_item in goals_data:
            approved_sources[("goal", g_item["id"])] = g_item
            selected_source_ids["goal"].append(g_item["id"])

        for s_item in skills_data:
            approved_sources[("skill", s_item["id"])] = s_item
            selected_source_ids["skill"].append(s_item["id"])

        for t_item in task_outcomes_data:
            approved_sources[("task_outcome", t_item["id"])] = t_item
            selected_source_ids["task_outcome"].append(t_item["id"])

        for th_item in evaluation_themes_data:
            approved_sources[("evaluation_theme", th_item["id"])] = th_item
            selected_source_ids["evaluation_theme"].append(th_item["id"])

        # Check for missing categories
        missing_categories: list[str] = []
        if not performance_data:
            missing_categories.append("performance")
        if not goals_data:
            missing_categories.append("goals")
        if not skills_data:
            missing_categories.append("skills")
        if not task_outcomes_data:
            missing_categories.append("task_outcomes")
        if not evaluation_themes_data:
            missing_categories.append("evaluation_themes")

        has_sufficient_data = len(missing_categories) == 0

        context = {
            "employee": employee_dict,
            "performance": performance_data,
            "goals": goals_data,
            "skills": skills_data,
            "task_outcomes": task_outcomes_data,
            "evaluation_themes": evaluation_themes_data,
        }

        return {
            "has_sufficient_data": has_sufficient_data,
            "missing_categories": missing_categories,
            "context": context,
            "approved_sources": approved_sources,
            "selected_source_ids": selected_source_ids,
        }

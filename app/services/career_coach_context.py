import json
from typing import Any

from sqlalchemy import case
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


def _serialize_context(context: dict[str, Any]) -> str:
    """Deterministically serializes context to JSON."""
    return json.dumps(context, indent=2)


def _rebuild_approved_sources(
    context: dict[str, Any],
) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, list[int]]]:
    """Rebuilds approved_sources and selected_source_ids matching the surviving context records."""
    approved_sources: dict[tuple[str, int], dict[str, Any]] = {}
    selected_source_ids: dict[str, list[int]] = {
        "performance": [],
        "goal": [],
        "skill": [],
        "task_outcome": [],
        "evaluation_theme": [],
    }
    for p_item in context.get("performance", []):
        approved_sources[("performance", p_item["id"])] = p_item
        selected_source_ids["performance"].append(p_item["id"])
    for g_item in context.get("goals", []):
        approved_sources[("goal", g_item["id"])] = g_item
        selected_source_ids["goal"].append(g_item["id"])
    for s_item in context.get("skills", []):
        approved_sources[("skill", s_item["id"])] = s_item
        selected_source_ids["skill"].append(s_item["id"])
    for t_item in context.get("task_outcomes", []):
        approved_sources[("task_outcome", t_item["id"])] = t_item
        selected_source_ids["task_outcome"].append(t_item["id"])
    for th_item in context.get("evaluation_themes", []):
        approved_sources[("evaluation_theme", th_item["id"])] = th_item
        selected_source_ids["evaluation_theme"].append(th_item["id"])
    return approved_sources, selected_source_ids


class CareerCoachContextBuilder:
    @staticmethod
    def enforce_total_context_limit(
        context: dict[str, Any],
        period: str | None = None,
        max_chars: int = MAX_TOTAL_CONTEXT_CHARS,
    ) -> tuple[dict[str, Any], dict[tuple[str, int], dict[str, Any]], dict[str, list[int]]]:
        """Enforces MAX_TOTAL_CONTEXT_CHARS after serialization in a deterministic, record-safe manner.

        Priority order:
        1. Current-period records have higher priority than older records.
        2. Older non-current records are pruned from lowest priority categories first (evaluation_themes, task_outcomes, goals, skills, performance).
        3. If still exceeding budget, current-period records beyond the minimum required 1 record per category are pruned.
        4. If still exceeding budget, long text fields in surviving records are progressively truncated.
        5. Updates approved_sources and selected_source_ids so they are 100% consistent with surviving records.
        """
        serialized = _serialize_context(context)
        if len(serialized) <= max_chars:
            approved_sources, selected_ids = _rebuild_approved_sources(context)
            return context, approved_sources, selected_ids

        # Phase 1: Prune older non-matching period records from end of lists (lowest priority first)
        # Order of categories to prune: evaluation_themes, task_outcomes, goals, performance
        prune_order = ["evaluation_themes", "task_outcomes", "goals", "performance"]
        changed = True
        while len(_serialize_context(context)) > max_chars and changed:
            changed = False
            for cat in prune_order:
                records = context.get(cat, [])
                # Find an older record (record period != requested period) to remove from the back
                for i in range(len(records) - 1, -1, -1):
                    rec_period = records[i].get("period")
                    if period and rec_period != period and len(records) > 1:
                        records.pop(i)
                        changed = True
                        break
                if len(_serialize_context(context)) <= max_chars:
                    break

        # Phase 2: If still oversized, prune excess current-period / cumulative records while preserving at least 1 record per category
        if len(_serialize_context(context)) > max_chars:
            excess_prune_order = ["evaluation_themes", "task_outcomes", "skills", "goals", "performance"]
            changed = True
            while len(_serialize_context(context)) > max_chars and changed:
                changed = False
                for cat in excess_prune_order:
                    records = context.get(cat, [])
                    if len(records) > 1:
                        records.pop()  # remove lowest priority (end of sorted list)
                        changed = True
                    if len(_serialize_context(context)) <= max_chars:
                        break

        # Phase 3: If still oversized, progressively truncate long text fields (evidence, outcome, description)
        if len(_serialize_context(context)) > max_chars:
            for cat in ["evaluation_themes", "task_outcomes", "skills", "goals"]:
                for rec in context.get(cat, []):
                    for field in ("evidence", "outcome", "title"):
                        if field in rec and isinstance(rec[field], str) and len(rec[field]) > 100:
                            rec[field] = rec[field][:97] + "..."
                            if len(_serialize_context(context)) <= max_chars:
                                break
                    if len(_serialize_context(context)) <= max_chars:
                        break
                if len(_serialize_context(context)) <= max_chars:
                    break

        approved_sources, selected_ids = _rebuild_approved_sources(context)
        return context, approved_sources, selected_ids

    @staticmethod
    def build_context(
        db: Session,
        employee_id: str,
        period: str | None = None,
    ) -> dict[str, Any]:
        """Gathers and prepares sanitized employee data for the AI Career Coach.

        - Guarantees strict employee data isolation.
        - Enforces approved-data only (P0-4).
        - Enforces deterministic context budget, record caps, and period prioritization (P1-4, P1-5).
        - Enforces total context character limit <= 12000 after serialization.
        - Exposes source IDs and canonical types for grounding validation.
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

        # Helper to construct deterministic period ordering:
        # 1. Exact requested period prioritized first (case condition = 0, else 1)
        # 2. Within each partition: recency (created_at descending, id descending)
        def _get_period_order_criteria(model_cls):
            if period:
                period_priority = case((model_cls.period == period, 0), else_=1)
                return [period_priority, model_cls.created_at.desc(), model_cls.id.desc()]
            return [model_cls.period.desc(), model_cls.created_at.desc(), model_cls.id.desc()]

        # 2. Performance records (strictly isolated to employee.id, approved only)
        perf_query = db.query(PerformanceRecord).filter(
            PerformanceRecord.employee_id == employee.id,
            PerformanceRecord.is_approved.is_(True),
        )
        if period:
            perf_query = perf_query.filter(PerformanceRecord.period == period)
        perf_records = (
            perf_query.order_by(*_get_period_order_criteria(PerformanceRecord))
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
        # P1-5: Exact requested period prioritized first, then older records
        goal_query = db.query(Goal).filter(
            Goal.employee_id == employee.id,
            Goal.is_approved.is_(True),
        )
        if period:
            goal_query = goal_query.filter((Goal.period == period) | (Goal.period.is_(None)))
        goals = (
            goal_query.order_by(*_get_period_order_criteria(Goal))
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

        # 4. Skills (cumulative across employee tenure, approved only)
        skills = (
            db.query(Skill)
            .filter(
                Skill.employee_id == employee.id,
                Skill.is_approved.is_(True),
            )
            .order_by(Skill.created_at.desc(), Skill.id.desc())
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
        # P1-5: Exact requested period prioritized first, then older records
        task_query = db.query(TaskOutcome).filter(
            TaskOutcome.employee_id == employee.id,
            TaskOutcome.is_approved.is_(True),
        )
        if period:
            task_query = task_query.filter((TaskOutcome.period == period) | (TaskOutcome.period.is_(None)))
        tasks = (
            task_query.order_by(*_get_period_order_criteria(TaskOutcome))
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
        # P1-5: Exact requested period prioritized first, then older records
        theme_query = db.query(EvaluationTheme).filter(
            EvaluationTheme.employee_id == employee.id,
            EvaluationTheme.is_approved.is_(True),
        )
        if period:
            theme_query = theme_query.filter(
                (EvaluationTheme.period == period) | (EvaluationTheme.period.is_(None))
            )
        themes = (
            theme_query.order_by(*_get_period_order_criteria(EvaluationTheme))
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

        # Check for missing categories before serialization truncation
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

        # P1-4: Enforce total context budget <= MAX_TOTAL_CONTEXT_CHARS after serialization
        context, approved_sources, selected_source_ids = (
            CareerCoachContextBuilder.enforce_total_context_limit(
                context=context,
                period=period,
                max_chars=MAX_TOTAL_CONTEXT_CHARS,
            )
        )

        return {
            "has_sufficient_data": has_sufficient_data,
            "missing_categories": missing_categories,
            "context": context,
            "approved_sources": approved_sources,
            "selected_source_ids": selected_source_ids,
        }


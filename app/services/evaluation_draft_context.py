"""Context service for the Evaluation Draft Assistant.

Builds a deterministic, sanitized, approved-only context payload for an employee
evaluation cycle, prioritizing the target evaluation period, bounding total context size,
and indexing approved source records for strict grounding validation.
"""

from __future__ import annotations

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

# Context Budget Limits
MAX_PERFORMANCE_RECORDS = 5
MAX_GOALS = 8
MAX_SKILLS = 10
MAX_TASK_OUTCOMES = 8
MAX_EVALUATION_THEMES = 8
MAX_FIELD_CHARS = 300
MAX_TOTAL_CONTEXT_CHARS = 12000

EVALUATION_CATEGORIES = [
    "performance",
    "goals",
    "skills",
    "task_outcomes",
    "evaluation_themes",
]


def _clean_str(val: Any, max_chars: int = MAX_FIELD_CHARS) -> str:
    """Sanitizes text fields, stripping whitespace and bounding length."""
    if val is None:
        return ""
    text = str(val).strip()
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text


def _serialize_context(context: dict[str, Any]) -> str:
    """Deterministically serializes context to JSON, excluding internal indexing dictionaries."""
    clean = {k: v for k, v in context.items() if k not in ("approved_sources", "selected_source_ids")}
    return json.dumps(clean, indent=2, sort_keys=True)


def _rebuild_approved_sources(
    context: dict[str, Any],
) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, list[int]]]:
    """Rebuilds approved_sources and selected_source_ids matching surviving context records."""
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


class EvaluationDraftContextBuilder:
    """Builds sanitized, deterministic, approved-only context for employee evaluation drafts."""

    @staticmethod
    def enforce_total_context_limit(
        context: dict[str, Any],
        period: str | None = None,
        max_chars: int = MAX_TOTAL_CONTEXT_CHARS,
    ) -> tuple[dict[str, Any], dict[tuple[str, int], dict[str, Any]], dict[str, list[int]]]:
        """Deterministically enforces context character budget without corrupting JSON."""
        serialized = _serialize_context(context)
        if len(serialized) <= max_chars:
            approved_sources, selected_ids = _rebuild_approved_sources(context)
            return context, approved_sources, selected_ids

        # Phase 1: Prune non-target period records from lowest priority categories first
        prune_order = ["evaluation_themes", "task_outcomes", "goals", "performance"]
        changed = True
        while len(_serialize_context(context)) > max_chars and changed:
            changed = False
            for cat in prune_order:
                records = context.get(cat, [])
                for i in range(len(records) - 1, -1, -1):
                    rec_period = records[i].get("period")
                    if period and rec_period != period and len(records) > 1:
                        records.pop(i)
                        changed = True
                        break
                if len(_serialize_context(context)) <= max_chars:
                    break

        # Phase 2: If still exceeding budget, prune lowest priority records while keeping at least 1
        if len(_serialize_context(context)) > max_chars:
            excess_prune_order = ["evaluation_themes", "task_outcomes", "skills", "goals", "performance"]
            changed = True
            while len(_serialize_context(context)) > max_chars and changed:
                changed = False
                for cat in excess_prune_order:
                    records = context.get(cat, [])
                    if len(records) > 1:
                        records.pop()
                        changed = True
                    if len(_serialize_context(context)) <= max_chars:
                        break

        # Phase 3: Progressively truncate long text fields if still oversized
        if len(_serialize_context(context)) > max_chars:
            for cat in ["evaluation_themes", "task_outcomes", "skills", "goals"]:
                for rec in context.get(cat, []):
                    for field in ("evidence", "outcome", "title"):
                        if field in rec and isinstance(rec[field], str) and len(rec[field]) > 80:
                            rec[field] = rec[field][:77] + "..."
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
        """Builds sanitized context payload for an employee's evaluation draft.

        - Enforces strict employee boundary isolation.
        - Includes only approved records (is_approved == True).
        - Prioritizes target period records, then newer records deterministically.
        - Provides approved_sources index for grounding verification.
        """
        employee = db.query(Employee).filter(Employee.id == employee_id).first()
        if not employee:
            return {
                "employee": None,
                "has_sufficient_data": False,
                "missing_categories": EVALUATION_CATEGORIES.copy(),
                "period": period,
                "approved_sources": {},
                "selected_source_ids": {},
            }

        # 1. Performance Records (Approved only)
        perf_query = db.query(PerformanceRecord).filter(
            PerformanceRecord.employee_id == employee_id,
            PerformanceRecord.is_approved.is_(True),
        )
        if period:
            perf_query = perf_query.order_by(
                case((PerformanceRecord.period == period, 0), else_=1),
                PerformanceRecord.period.desc(),
                PerformanceRecord.id.desc(),
            )
        else:
            perf_query = perf_query.order_by(
                PerformanceRecord.period.desc(),
                PerformanceRecord.id.desc(),
            )
        performance_records = perf_query.limit(MAX_PERFORMANCE_RECORDS).all()

        # 2. Goals (Approved only)
        goals_query = db.query(Goal).filter(
            Goal.employee_id == employee_id,
            Goal.is_approved.is_(True),
        )
        if period:
            goals_query = goals_query.order_by(
                case((Goal.period == period, 0), else_=1),
                Goal.period.desc(),
                Goal.id.desc(),
            )
        else:
            goals_query = goals_query.order_by(
                Goal.period.desc(),
                Goal.id.desc(),
            )
        goals = goals_query.limit(MAX_GOALS).all()

        # 3. Skills (Approved only)
        skills = (
            db.query(Skill)
            .filter(
                Skill.employee_id == employee_id,
                Skill.is_approved.is_(True),
            )
            .order_by(Skill.level.desc(), Skill.id.asc())
            .limit(MAX_SKILLS)
            .all()
        )

        # 4. Task Outcomes (Approved only)
        task_query = db.query(TaskOutcome).filter(
            TaskOutcome.employee_id == employee_id,
            TaskOutcome.is_approved.is_(True),
        )
        if period:
            task_query = task_query.order_by(
                case((TaskOutcome.period == period, 0), else_=1),
                TaskOutcome.period.desc(),
                TaskOutcome.id.desc(),
            )
        else:
            task_query = task_query.order_by(
                TaskOutcome.period.desc(),
                TaskOutcome.id.desc(),
            )
        task_outcomes = task_query.limit(MAX_TASK_OUTCOMES).all()

        # 5. Evaluation Themes (Approved only)
        theme_query = db.query(EvaluationTheme).filter(
            EvaluationTheme.employee_id == employee_id,
            EvaluationTheme.is_approved.is_(True),
        )
        if period:
            theme_query = theme_query.order_by(
                case((EvaluationTheme.period == period, 0), else_=1),
                EvaluationTheme.period.desc(),
                EvaluationTheme.id.desc(),
            )
        else:
            theme_query = theme_query.order_by(
                EvaluationTheme.period.desc(),
                EvaluationTheme.id.desc(),
            )
        evaluation_themes = theme_query.limit(MAX_EVALUATION_THEMES).all()

        # Map to sanitized dicts
        perf_list = [
            {
                "id": p.id,
                "period": p.period,
                "overall_score": round(float(p.overall_score), 2),
                "task_completion_rate": round(float(p.task_completion_rate), 2),
                "goal_achievement_rate": round(float(p.goal_achievement_rate), 2),
                "attendance_rate": round(float(p.attendance_rate), 2),
            }
            for p in performance_records
        ]
        goals_list = [
            {
                "id": g.id,
                "title": _clean_str(g.title),
                "progress": round(float(g.progress), 2),
                "status": _clean_str(g.status),
                "deadline": _clean_str(g.deadline),
                "period": _clean_str(g.period),
            }
            for g in goals
        ]
        skills_list = [
            {
                "id": s.id,
                "name": _clean_str(s.name),
                "level": _clean_str(s.level),
                "evidence": _clean_str(s.evidence),
            }
            for s in skills
        ]
        tasks_list = [
            {
                "id": t.id,
                "title": _clean_str(t.title),
                "status": _clean_str(t.status),
                "outcome": _clean_str(t.outcome),
                "completion_date": _clean_str(t.completion_date),
                "period": _clean_str(t.period),
            }
            for t in task_outcomes
        ]
        themes_list = [
            {
                "id": th.id,
                "theme": _clean_str(th.theme),
                "sentiment": _clean_str(th.sentiment),
                "evidence": _clean_str(th.evidence),
                "period": _clean_str(th.period),
            }
            for th in evaluation_themes
        ]

        # Calculate missing categories
        missing_categories: list[str] = []
        if not perf_list:
            missing_categories.append("performance")
        if not goals_list:
            missing_categories.append("goals")
        if not skills_list:
            missing_categories.append("skills")
        if not tasks_list:
            missing_categories.append("task_outcomes")
        if not themes_list:
            missing_categories.append("evaluation_themes")

        # Evaluation requires concrete evidence: at least 1 performance record AND at least 1 record from goals/tasks/themes
        has_sufficient_data = bool(
            perf_list and (goals_list or tasks_list or themes_list)
        )

        # Determine effective target period
        effective_period = period
        if not effective_period and perf_list:
            effective_period = perf_list[0]["period"]

        context: dict[str, Any] = {
            "employee": {
                "id": employee.id,
                "first_name": _clean_str(employee.first_name),
                "last_name": _clean_str(employee.last_name),
                "role_title": _clean_str(employee.role_title),
                "department": _clean_str(employee.department),
            },
            "period": effective_period,
            "has_sufficient_data": has_sufficient_data,
            "missing_categories": missing_categories,
            "performance": perf_list,
            "goals": goals_list,
            "skills": skills_list,
            "task_outcomes": tasks_list,
            "evaluation_themes": themes_list,
        }

        # Enforce budget limit and compute approved_sources index
        context, approved_sources, selected_source_ids = (
            EvaluationDraftContextBuilder.enforce_total_context_limit(
                context, period=effective_period, max_chars=MAX_TOTAL_CONTEXT_CHARS
            )
        )

        context["approved_sources"] = approved_sources
        context["selected_source_ids"] = selected_source_ids

        return context

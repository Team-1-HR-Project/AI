"""Context service for AI #4: Skill-Gap & Development Recommendations.

Loads and sanitizes ONLY approved records belonging to the requested employee.
Builds deterministic, structured context for the AI skill-gap analyzer, enforces
strict character budgets, checks data sufficiency before LLM invocation, and
creates an approved_sources registry for downstream grounding validation.
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

# Canonical source types matching existing system conventions
CANONICAL_SOURCE_TYPES = (
    "performance",
    "goal",
    "skill",
    "task_outcome",
    "evaluation_theme",
)

# Context Budget Limits
MAX_SKILLS = 15
MAX_PERFORMANCE_RECORDS = 5
MAX_GOALS = 8
MAX_TASK_OUTCOMES = 8
MAX_EVALUATION_THEMES = 8
MAX_FIELD_CHARS = 300
MAX_TOTAL_CONTEXT_CHARS = 12000


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
    clean = {
        k: v
        for k, v in context.items()
        if k not in ("approved_sources", "selected_source_ids")
    }
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

    for s_item in context.get("skills", []):
        approved_sources[("skill", s_item["id"])] = s_item
        selected_source_ids["skill"].append(s_item["id"])
    for p_item in context.get("performance", []):
        approved_sources[("performance", p_item["id"])] = p_item
        selected_source_ids["performance"].append(p_item["id"])
    for g_item in context.get("goals", []):
        approved_sources[("goal", g_item["id"])] = g_item
        selected_source_ids["goal"].append(g_item["id"])
    for t_item in context.get("task_outcomes", []):
        approved_sources[("task_outcome", t_item["id"])] = t_item
        selected_source_ids["task_outcome"].append(t_item["id"])
    for th_item in context.get("evaluation_themes", []):
        approved_sources[("evaluation_theme", th_item["id"])] = th_item
        selected_source_ids["evaluation_theme"].append(th_item["id"])

    return approved_sources, selected_source_ids


class SkillGapContextBuilder:
    """Builds sanitized, deterministic, approved-only context for Skill-Gap & Development Recommendations."""

    @staticmethod
    def enforce_total_context_limit(
        context: dict[str, Any],
        period: str | None = None,
        max_chars: int = MAX_TOTAL_CONTEXT_CHARS,
    ) -> tuple[dict[str, Any], dict[tuple[str, int], dict[str, Any]], dict[str, list[int]]]:
        """Deterministically enforces context character budget without corrupting JSON.

        Pruning priority (lowest priority pruned first):
        1. Older non-matching period records from task_outcomes, goals, evaluation_themes, performance.
        2. Non-matching period records down to minimum retention.
        3. Text field truncation on surviving records if still exceeding budget.
        4. Always preserve approved skills as core baseline for skill-gap analysis.
        """
        serialized = _serialize_context(context)
        if len(serialized) <= max_chars:
            approved_sources, selected_ids = _rebuild_approved_sources(context)
            return context, approved_sources, selected_ids

        # Phase 1: Prune older non-period records from supporting categories (lowest priority first)
        categories_to_prune = ["task_outcomes", "goals", "evaluation_themes", "performance"]
        for cat in categories_to_prune:
            items = context.get(cat, [])
            if period:
                non_period = [item for item in items if item.get("period") != period]
                period_items = [item for item in items if item.get("period") == period]
                while non_period and len(_serialize_context(context)) > max_chars:
                    non_period.pop()
                    context[cat] = period_items + non_period
            else:
                while len(items) > 1 and len(_serialize_context(context)) > max_chars:
                    items.pop()
                    context[cat] = items

            if len(_serialize_context(context)) <= max_chars:
                break

        # Phase 2: If still exceeding, progressively truncate long text fields
        if len(_serialize_context(context)) > max_chars:
            for cat in ["evaluation_themes", "task_outcomes", "goals", "skills"]:
                for item in context.get(cat, []):
                    for field in ["evidence", "outcome", "title", "rationale"]:
                        if field in item and isinstance(item[field], str) and len(item[field]) > 100:
                            item[field] = item[field][:100] + "..."
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
        target_role: str | None = None,
        target_skills: list[str] | None = None,
    ) -> dict[str, Any]:
        """Gathers and sanitizes approved employee data for Skill-Gap analysis.

        - Strictly validates employee existence.
        - Enforces employee data isolation (requested employee_id only).
        - Loads ONLY approved records (is_approved == True).
        - Enforces deterministic context budget limits.
        - Indexes approved sources by (source_type, source_id).
        - Evaluates data sufficiency before any AI invocation.
        """
        employee = db.query(Employee).filter(Employee.id == employee_id).first()
        if not employee:
            return {
                "has_sufficient_data": False,
                "missing_categories": ["employee", "skills"],
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

        # Helper for deterministic period ordering
        def _get_period_order_criteria(model_cls):
            if period:
                period_priority = case((model_cls.period == period, 0), else_=1)
                return [period_priority, model_cls.created_at.desc(), model_cls.id.desc()]
            return [model_cls.period.desc(), model_cls.created_at.desc(), model_cls.id.desc()]

        # 2. Skills (cumulative inventory, approved only)
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

        # 3. Performance records (approved only)
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

        # 4. Goals (approved only)
        goal_query = db.query(Goal).filter(
            Goal.employee_id == employee.id,
            Goal.is_approved.is_(True),
        )
        if period:
            goal_query = goal_query.filter(
                (Goal.period == period) | (Goal.period.is_(None))
            )
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

        # 5. Task Outcomes (approved only)
        task_query = db.query(TaskOutcome).filter(
            TaskOutcome.employee_id == employee.id,
            TaskOutcome.is_approved.is_(True),
        )
        if period:
            task_query = task_query.filter(
                (TaskOutcome.period == period) | (TaskOutcome.period.is_(None))
            )
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

        # 6. Evaluation Themes (approved only)
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

        # 7. Optional Target Context (sanitized)
        target_context: dict[str, Any] = {}
        if target_role:
            target_context["target_role"] = _clean_str(target_role, 100)
        if target_skills:
            target_context["target_skills"] = [
                _clean_str(ts, 50) for ts in target_skills[:10] if ts
            ]

        # 8. Data Sufficiency Assessment
        # Rule: A valid skill-gap analysis strictly requires:
        #   (a) At least one approved Skill
        #   (b) At least one supporting category of approved records (performance, goals, tasks, or evaluation themes)
        missing_categories: list[str] = []
        if not skills_data:
            missing_categories.append("skills")

        has_supporting_data = bool(
            performance_data
            or goals_data
            or task_outcomes_data
            or evaluation_themes_data
        )
        if not has_supporting_data:
            missing_categories.append("supporting_records")

        has_sufficient_data = (len(missing_categories) == 0)

        context: dict[str, Any] = {
            "employee": employee_dict,
            "skills": skills_data,
            "performance": performance_data,
            "goals": goals_data,
            "task_outcomes": task_outcomes_data,
            "evaluation_themes": evaluation_themes_data,
        }
        if target_context:
            context["target"] = target_context

        # Enforce budget limit <= MAX_TOTAL_CONTEXT_CHARS after serialization
        context, approved_sources, selected_source_ids = (
            SkillGapContextBuilder.enforce_total_context_limit(
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


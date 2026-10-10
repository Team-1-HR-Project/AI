"""Explicit response DTO mappers between internal AI responses and public HR responses.

Preserves all critical evidence, citations, metadata, targets, and disclaimers
while aligning with the public HR contract expected by Flutter / Web clients.
"""

from __future__ import annotations

import json
from typing import Any


def map_career_coach_response(ai_data: dict[str, Any]) -> dict[str, Any]:
    """Maps internal Career Coach response to public HR response data."""
    status = ai_data.get("status", "success")
    if status == "insufficient_data":
        return {
            "status": "insufficient_data",
            "employee_id": ai_data.get("employee_id"),
            "missing_categories": ai_data.get("missing_categories", []),
            "message": ai_data.get("message", "Not enough approved employee data to generate a reliable career coaching plan."),
            "created_at": ai_data.get("created_at"),
        }

    dev_areas = ai_data.get("development_areas") or []
    focus = ai_data.get("development_focus")
    if not focus and dev_areas:
        first_title = dev_areas[0].get("title") if isinstance(dev_areas[0], dict) else str(dev_areas[0])
        focus = f"Prioritize: {first_title}"

    return {
        "status": "success",
        "employee_id": ai_data.get("employee_id"),
        "development_focus": focus or "Career Development Focus",
        "strengths": ai_data.get("strengths", []),
        "development_areas": dev_areas,
        "development_plan": ai_data.get("development_plan", []),
        "follow_up": ai_data.get("follow_up"),
        "created_at": ai_data.get("created_at"),
    }


def map_performance_insight_response(ai_data: dict[str, Any]) -> dict[str, Any]:
    """Maps internal Performance Insight response to public HR response data."""
    status = ai_data.get("status", "success")
    if status == "insufficient_data":
        return {
            "status": "insufficient_data",
            "employee_id": ai_data.get("employee_id"),
            "reason": ai_data.get("reason") or ai_data.get("message", "Insufficient approved performance data for this period."),
            "created_at": ai_data.get("created_at"),
        }

    interp = ai_data.get("interpretation") or ai_data.get("ai_interpretation") or {}
    actions = ai_data.get("recommended_actions") or ai_data.get("suggested_review_actions") or []

    return {
        "status": "success",
        "employee_id": ai_data.get("employee_id"),
        "verified_facts": ai_data.get("verified_facts", {}),
        "calculated_trends": ai_data.get("calculated_trends", {}),
        "interpretation": interp,
        "ai_interpretation": interp,
        "recommended_actions": actions,
        "suggested_review_actions": actions,
        "created_at": ai_data.get("created_at"),
    }


def map_evaluation_draft_response(ai_data: dict[str, Any]) -> dict[str, Any]:
    """Maps internal Evaluation Draft response to public HR response data."""
    status = ai_data.get("status", "success")
    if status == "insufficient_data":
        return {
            "status": "insufficient_data",
            "employee_id": ai_data.get("employee_id"),
            "message": ai_data.get("message", "Not enough approved data to generate an evaluation draft."),
            "created_at": ai_data.get("created_at"),
        }

    return {
        "status": "success",
        "employee_id": ai_data.get("employee_id"),
        "period": ai_data.get("period"),
        "evaluation_narrative": ai_data.get("evaluation_narrative"),
        "strengths": ai_data.get("strengths", []),
        "improvement_areas": ai_data.get("improvement_areas", []),
        "entered_scores": ai_data.get("entered_scores"),
        "disclaimer": ai_data.get("disclaimer"),
        "human_review_required": ai_data.get("human_review_required", True),
        "created_at": ai_data.get("created_at"),
    }


def map_skill_gap_response(ai_data: dict[str, Any]) -> dict[str, Any]:
    """Maps internal Skill Gap response to public HR response data."""
    status = ai_data.get("status", "success")
    if status == "insufficient_data":
        return {
            "status": "insufficient_data",
            "employee_id": ai_data.get("employee_id"),
            "message": ai_data.get("message", "Not enough approved employee skill or performance records."),
            "created_at": ai_data.get("created_at"),
        }

    return {
        "status": "success",
        "employee_id": ai_data.get("employee_id"),
        "target_role": ai_data.get("target_role"),
        "skill_gaps": ai_data.get("skill_gaps", []),
        "recommendations": ai_data.get("recommendations", []),
        "disclaimer": ai_data.get("disclaimer"),
        "created_at": ai_data.get("created_at"),
    }


def map_attention_signal_response(ai_data: dict[str, Any]) -> dict[str, Any]:
    """Maps internal Attention Signal response to public HR response data."""
    status = ai_data.get("status", "success")
    if status == "insufficient_data":
        return {
            "status": "insufficient_data",
            "employee_id": ai_data.get("employee_id"),
            "message": ai_data.get("message", "Not enough approved data across recent periods."),
            "created_at": ai_data.get("created_at"),
        }

    return {
        "status": "success",
        "employee_id": ai_data.get("employee_id"),
        "target_period": ai_data.get("target_period"),
        "comparison_period": ai_data.get("comparison_period"),
        "attention_level": ai_data.get("attention_level"),
        "explanation": ai_data.get("explanation"),
        "contributing_indicators": ai_data.get("contributing_indicators", []),
        "recommended_follow_up": ai_data.get("recommended_follow_up", []),
        "advisory_only": ai_data.get("advisory_only", True),
        "human_review_required": ai_data.get("human_review_required", True),
        "created_at": ai_data.get("created_at"),
    }


def map_team_insight_response(ai_data: dict[str, Any]) -> dict[str, Any]:
    """Maps internal Team Insight response to public HR response data."""
    status = ai_data.get("status", "success")
    if status == "insufficient_data":
        return {
            "status": "insufficient_data",
            "department": ai_data.get("department"),
            "message": ai_data.get("message", "Not enough approved data across team members."),
            "created_at": ai_data.get("created_at"),
        }

    findings = ai_data.get("team_findings") or {}
    workload = findings.get("overdue_workload") or {}
    patterns = findings.get("skill_gap_patterns") or {}
    coverage = ai_data.get("data_coverage") or {}

    summary = ai_data.get("summary") or workload.get("summary") or "Team performance and workload insight summary."
    top_skills = ai_data.get("top_common_skills") or patterns.get("top_skills_represented") or []
    top_gaps = ai_data.get("top_common_gaps") or patterns.get("top_missing_skills") or []
    actions = ai_data.get("recommended_management_actions") or []
    team_size = ai_data.get("team_size") or coverage.get("total_team_members") or 0

    return {
        "status": "success",
        "department": ai_data.get("department"),
        "period": ai_data.get("period"),
        "team_size": team_size,
        "summary": summary,
        "top_common_skills": top_skills,
        "top_common_gaps": top_gaps,
        "recommended_management_actions": actions,
        "team_findings": findings,
        "data_coverage": coverage,
        "created_at": ai_data.get("created_at"),
    }


def map_policy_assistant_response(ai_data: dict[str, Any]) -> dict[str, Any]:
    """Maps internal Policy Assistant response to public HR response data."""
    status = ai_data.get("status", "success")
    if status == "unsupported":
        return {
            "status": "unsupported",
            "employee_id": ai_data.get("employee_id"),
            "message": ai_data.get("message", "The question is outside the approved policy scope."),
            "created_at": ai_data.get("created_at"),
        }

    return {
        "status": "success",
        "session_id": ai_data.get("session_id"),
        "employee_id": ai_data.get("employee_id"),
        "answer": ai_data.get("answer"),
        "policy_references": ai_data.get("policy_references", []),
        "employee_facts_used": ai_data.get("employee_facts_used", []),
        "created_at": ai_data.get("created_at"),
    }


def map_snapshot_response(ai_data: dict[str, Any]) -> dict[str, Any]:
    """Maps internal AI snapshot response to public HR snapshot DTO."""
    content = ai_data.get("content")
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            content = ai_data.get("content")

    return {
        "id": ai_data.get("id"),
        "generation_id": ai_data.get("generation_id"),
        "version": ai_data.get("version", 1),
        "feature": ai_data.get("feature"),
        "scope_employee_id": ai_data.get("scope_employee_id"),
        "scope_department": ai_data.get("scope_department"),
        "period": ai_data.get("period"),
        "content": content,
        "actor_employee_id": ai_data.get("actor_employee_id"),
        "actor_role": ai_data.get("actor_role"),
        "previous_snapshot_id": ai_data.get("previous_snapshot_id"),
        "regenerated_at": ai_data.get("regenerated_at"),
        "regeneration_reason": ai_data.get("regeneration_reason"),
        "source_changed": ai_data.get("source_changed"),
        "created_at": ai_data.get("created_at"),
    }


def map_history_response(ai_data: dict[str, Any]) -> dict[str, Any]:
    """Maps internal AI history response to public HR history DTO with pagination."""
    raw_snapshots = ai_data.get("snapshots", [])
    mapped_snapshots = [
        map_snapshot_response(s) if isinstance(s, dict) else s
        for s in raw_snapshots
    ]
    return {
        "feature": str(ai_data.get("feature", "")),
        "scope_employee_id": ai_data.get("scope_employee_id"),
        "scope_department": ai_data.get("scope_department"),
        "period": ai_data.get("period"),
        "total_versions": ai_data.get("total_versions", len(mapped_snapshots)),
        "page": ai_data.get("page", 1),
        "page_size": ai_data.get("page_size", 50),
        "has_more": ai_data.get("has_more", False),
        "snapshots": mapped_snapshots,
    }


def map_feedback_response(ai_data: dict[str, Any]) -> dict[str, Any]:
    """Maps internal AI feedback response to public HR feedback DTO."""
    return {
        "id": ai_data.get("id"),
        "snapshot_id": ai_data.get("snapshot_id"),
        "actor_employee_id": ai_data.get("actor_employee_id"),
        "actor_role": ai_data.get("actor_role"),
        "is_helpful": ai_data.get("is_helpful"),
        "feedback_text": ai_data.get("feedback_text"),
        "created_at": ai_data.get("created_at"),
    }

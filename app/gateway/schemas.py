"""Public HR API schemas for the HR Gateway endpoints.

Defines the exact public contracts expected by Flutter / Web clients,
wrapping data in the standard HR API envelope:
{
  "success": true,
  "message": "...",
  "data": { ... }
}
and translating errors into:
{
  "success": false,
  "message": "...",
  "errors": [...]
}
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

T = TypeVar("T")


# =============================================================================
# Envelopes
# =============================================================================


class HRApiSuccessResponse(BaseModel, Generic[T]):
    """Standard HR API successful response envelope."""

    success: bool = Field(default=True, json_schema_extra={"example": True})
    message: str = Field(..., json_schema_extra={"example": "Operation completed successfully."})
    data: T


class HRApiErrorResponse(BaseModel):
    """Standard HR API error response envelope."""

    success: bool = Field(default=False, json_schema_extra={"example": False})
    message: str = Field(..., json_schema_extra={"example": "An error occurred."})
    errors: list[Any] = Field(default_factory=list, json_schema_extra={"example": []})


# =============================================================================
# Request DTOs
# =============================================================================


class HRCareerCoachRequest(BaseModel):
    """Input request model for HR Career Coach gateway."""

    employee_id: str | None = Field(
        default=None,
        description="Target employee identifier (e.g. 'EMP-001'). If omitted, defaults to authenticated user.",
        json_schema_extra={"example": "EMP-2026-001"},
    )
    period: str | None = Field(
        default=None,
        description="Optional evaluation/quarter period filter (e.g. '2026-Q3')",
        json_schema_extra={"example": "2026-Q3"},
    )

    model_config = ConfigDict(extra="forbid")


class HRPerformanceInsightRequest(BaseModel):
    """Input request model for HR Performance Insight gateway."""

    employee_id: str | None = Field(
        default=None,
        description="Target employee identifier (e.g. 'EMP-001'). If omitted, defaults to authenticated user.",
        json_schema_extra={"example": "EMP-2026-001"},
    )
    period: str = Field(
        ...,
        min_length=1,
        description="Target performance review period (e.g. '2026-Q3')",
        json_schema_extra={"example": "2026-Q3"},
    )

    model_config = ConfigDict(extra="forbid")


class HREvaluationDraftRequest(BaseModel):
    """Input request model for HR Evaluation Draft gateway (Manager / Admin)."""

    employee_id: str | None = Field(
        default=None,
        description="Target employee identifier (e.g. 'EMP-001')",
        json_schema_extra={"example": "EMP-2026-001"},
    )
    period: str = Field(
        ...,
        min_length=1,
        description="Target evaluation period (e.g. '2026-Q3')",
        json_schema_extra={"example": "2026-Q3"},
    )
    evaluation_scores: dict[str, float] = Field(
        ...,
        description="Manager-entered evaluation scores by category (0.0 to 100.0)",
        json_schema_extra={"example": {"technical_skills": 85.0, "communication": 90.0, "leadership": 80.0, "teamwork": 88.0}},
    )
    manager_notes: str | None = Field(
        default=None,
        max_length=3000,
        description="Optional manager observations, notes, or qualitative context",
        json_schema_extra={"example": "Consistent delivery and strong team player."},
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("evaluation_scores")
    @classmethod
    def validate_scores(cls, v: dict[str, float]) -> dict[str, float]:
        for k, score in v.items():
            if not (0.0 <= score <= 100.0):
                raise ValueError(f"Score for category '{k}' must be between 0.0 and 100.0, got {score}")
        return v


class HRSkillGapRequest(BaseModel):
    """Input request model for HR Skill Gap gateway."""

    employee_id: str | None = Field(
        default=None,
        description="Target employee identifier (e.g. 'EMP-001'). If omitted, defaults to authenticated user.",
        json_schema_extra={"example": "EMP-2026-001"},
    )
    period: str | None = Field(
        default=None,
        description="Optional period filter (e.g. '2026-Q3')",
        json_schema_extra={"example": "2026-Q3"},
    )
    target_role: str | None = Field(
        default=None,
        description="Optional target role for career progression gap analysis",
        json_schema_extra={"example": "Senior Backend Engineer"},
    )
    target_skills: list[str] | None = Field(
        default=None,
        max_length=10,
        description="Optional list of specific skills to assess (max 10)",
        json_schema_extra={"example": ["System Design", "Kubernetes"]},
    )

    model_config = ConfigDict(extra="forbid")


class HRAttentionSignalRequest(BaseModel):
    """Input request model for HR Employee Attention Signal gateway (Manager / Admin)."""

    employee_id: str | None = Field(
        default=None,
        description="Target employee identifier (e.g. 'EMP-001')",
        json_schema_extra={"example": "EMP-2026-001"},
    )
    target_period: str = Field(
        ...,
        min_length=1,
        description="Target period to evaluate for work patterns and changes (e.g. '2026-Q3')",
        json_schema_extra={"example": "2026-Q3"},
    )

    model_config = ConfigDict(extra="forbid")


class HRTeamInsightRequest(BaseModel):
    """Input request model for HR Team Insight gateway (Manager / Admin)."""

    department: str | None = Field(
        default=None,
        description="Department name (e.g. 'Engineering'). Either department or department_id must be provided.",
        json_schema_extra={"example": "Engineering"},
    )
    department_id: int | str | None = Field(
        default=None,
        description="Department database ID (e.g. 1). Mapped by gateway to validated department name.",
        json_schema_extra={"example": 1},
    )
    period: str | None = Field(
        default=None,
        description="Optional evaluation period (e.g. '2026-Q3')",
        json_schema_extra={"example": "2026-Q3"},
    )

    model_config = ConfigDict(extra="forbid")


class HRPolicyQuestionRequest(BaseModel):
    """Input request model for HR Policy Assistant gateway."""

    employee_id: str | None = Field(
        default=None,
        description="Target employee identifier. If omitted, defaults to authenticated user.",
        json_schema_extra={"example": "EMP-2026-001"},
    )
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Question regarding company HR policies",
        json_schema_extra={"example": "What is the remote work policy?"},
    )
    session_id: str | None = Field(
        default=None,
        description="Optional existing chat session ID to continue conversation",
        json_schema_extra={"example": "session-abc-123"},
    )

    model_config = ConfigDict(extra="forbid")


class HRFeedbackCreateRequest(BaseModel):
    """Input request model for submitting feedback on an AI snapshot."""

    is_helpful: bool = Field(..., description="Whether the insight was helpful or accurate")
    feedback_text: str | None = Field(
        default=None,
        max_length=2000,
        description="Optional qualitative comments or improvement suggestions",
        json_schema_extra={"example": "The recommendations were very relevant."},
    )

    model_config = ConfigDict(extra="forbid")


class HRRegenerateRequest(BaseModel):
    """Input request model for regenerating an AI insight snapshot."""

    reason: str | None = Field(
        default=None,
        max_length=1000,
        description="Optional reason for regenerating the insight",
        json_schema_extra={"example": "New performance data was approved."},
    )

    model_config = ConfigDict(extra="forbid")


# =============================================================================
# Response Data DTOs (Preserving full evidence, metadata, and contracts)
# =============================================================================


class HRCareerCoachData(BaseModel):
    """Public HR response data for Career Coach."""

    status: str = Field(..., json_schema_extra={"example": "success"})
    employee_id: str = Field(..., json_schema_extra={"example": "EMP-2026-001"})
    development_focus: str | None = Field(default=None, json_schema_extra={"example": "Prioritize: System Architecture"})
    strengths: list[Any] = Field(default_factory=list, description="List of strengths with grounded evidence")
    development_areas: list[Any] = Field(default_factory=list, description="Growth areas with priority and evidence")
    development_plan: list[Any] = Field(default_factory=list, description="Actionable plan items with measurable targets")
    follow_up: dict[str, Any] | None = Field(default=None, description="Follow-up checkpoint and review focus")
    missing_categories: list[str] | None = Field(default=None, description="Missing categories if insufficient data")
    message: str | None = Field(default=None, description="Explanation message")
    created_at: str | datetime | None = None

    model_config = ConfigDict(extra="allow")


class HRPerformanceInsightData(BaseModel):
    """Public HR response data for Performance Insight."""

    status: str = Field(..., json_schema_extra={"example": "success"})
    employee_id: str = Field(..., json_schema_extra={"example": "EMP-2026-001"})
    verified_facts: dict[str, Any] | None = Field(default=None, description="Verified facts and metrics")
    calculated_trends: dict[str, Any] | None = Field(default=None, description="Calculated performance trends")
    interpretation: dict[str, Any] | None = Field(default=None, description="AI interpretation of metrics")
    ai_interpretation: dict[str, Any] | None = Field(default=None, description="Alias for AI interpretation")
    recommended_actions: list[Any] | None = Field(default=None, description="Recommended review actions")
    suggested_review_actions: list[Any] | None = Field(default=None, description="Alias for suggested review actions")
    reason: str | None = Field(default=None, description="Reason if data is insufficient")
    created_at: str | datetime | None = None

    model_config = ConfigDict(extra="allow")


class HREvaluationDraftData(BaseModel):
    """Public HR response data for Evaluation Draft."""

    status: str = Field(..., json_schema_extra={"example": "success"})
    employee_id: str = Field(..., json_schema_extra={"example": "EMP-2026-001"})
    period: str | None = Field(default=None, json_schema_extra={"example": "2026-Q3"})
    evaluation_narrative: str | None = Field(default=None, description="Draft evaluation narrative")
    strengths: list[Any] = Field(default_factory=list, description="Observed strengths with evidence")
    improvement_areas: list[Any] = Field(default_factory=list, description="Growth areas with evidence and priority")
    entered_scores: dict[str, float] | None = Field(default=None, description="Manager-entered evaluation scores")
    disclaimer: str | None = Field(default=None, description="AI safety disclaimer")
    human_review_required: bool = Field(default=True, json_schema_extra={"example": True})
    message: str | None = Field(default=None, description="Message if insufficient data")
    created_at: str | datetime | None = None

    model_config = ConfigDict(extra="allow")


class HRSkillGapData(BaseModel):
    """Public HR response data for Skill Gap."""

    status: str = Field(..., json_schema_extra={"example": "success"})
    employee_id: str = Field(..., json_schema_extra={"example": "EMP-2026-001"})
    target_role: str | None = Field(default=None, json_schema_extra={"example": "Senior Backend Engineer"})
    skill_gaps: list[Any] = Field(default_factory=list, description="Identified skill gaps with severity and evidence")
    recommendations: list[Any] = Field(default_factory=list, description="Learning recommendations with measurable targets")
    disclaimer: str | None = Field(default=None, description="Advisory disclaimer")
    message: str | None = Field(default=None, description="Message if insufficient data")
    created_at: str | datetime | None = None

    model_config = ConfigDict(extra="allow")


class HRAttentionSignalData(BaseModel):
    """Public HR response data for Attention Signal."""

    status: str = Field(..., json_schema_extra={"example": "success"})
    employee_id: str = Field(..., json_schema_extra={"example": "EMP-2026-001"})
    target_period: str | None = Field(default=None, json_schema_extra={"example": "2026-Q3"})
    comparison_period: str | None = Field(default=None, json_schema_extra={"example": "2026-Q2"})
    attention_level: str | None = Field(default=None, json_schema_extra={"example": "Medium"})
    explanation: str | None = Field(default=None, description="Objective pattern explanation")
    contributing_indicators: list[Any] = Field(default_factory=list, description="Work indicators with evidence")
    recommended_follow_up: list[Any] = Field(default_factory=list, description="Supportive manager follow-up actions")
    advisory_only: bool = Field(default=True, json_schema_extra={"example": True})
    human_review_required: bool = Field(default=True, json_schema_extra={"example": True})
    message: str | None = Field(default=None, description="Message if insufficient data")
    created_at: str | datetime | None = None

    model_config = ConfigDict(extra="allow")


class HRTeamInsightData(BaseModel):
    """Public HR response data for Team Insight."""

    status: str = Field(..., json_schema_extra={"example": "success"})
    department: str = Field(..., json_schema_extra={"example": "Engineering"})
    period: str | None = Field(default=None, json_schema_extra={"example": "2026-Q3"})
    team_size: int | None = Field(default=None, json_schema_extra={"example": 8})
    summary: str | None = Field(default=None, description="Narrative summary of team findings")
    top_common_skills: list[Any] = Field(default_factory=list, description="Common skills across the department")
    top_common_gaps: list[Any] = Field(default_factory=list, description="Common skill gaps across the department")
    recommended_management_actions: list[Any] = Field(default_factory=list, description="Recommended management actions")
    team_findings: dict[str, Any] | None = Field(default=None, description="Detailed component findings")
    data_coverage: dict[str, Any] | None = Field(default=None, description="Data coverage and privacy guarantees")
    message: str | None = Field(default=None, description="Message if insufficient data")
    created_at: str | datetime | None = None

    model_config = ConfigDict(extra="allow")


class HRPolicyAssistantData(BaseModel):
    """Public HR response data for Policy Assistant."""

    status: str = Field(..., json_schema_extra={"example": "success"})
    session_id: str | None = Field(default=None, json_schema_extra={"example": "session-abc-123"})
    employee_id: str = Field(..., json_schema_extra={"example": "EMP-2026-001"})
    answer: str | None = Field(default=None, description="Grounded answer to the policy question")
    policy_references: list[Any] = Field(default_factory=list, description="Referenced policies with ID, code, title, and version")
    employee_facts_used: list[str] = Field(default_factory=list, description="Employee facts used to contextualize the answer")
    message: str | None = Field(default=None, description="Message if unsupported or explanation")
    created_at: str | datetime | None = None

    model_config = ConfigDict(extra="allow")


class HRInsightSnapshotData(BaseModel):
    """Public HR response data for a single AI insight snapshot."""

    id: str = Field(..., json_schema_extra={"example": "snap-12345"})
    generation_id: str = Field(..., json_schema_extra={"example": "gen-12345"})
    version: int = Field(..., json_schema_extra={"example": 1})
    feature: str = Field(..., json_schema_extra={"example": "career_coach"})
    scope_employee_id: str | None = None
    scope_department: str | None = None
    period: str | None = None
    content: Any = Field(..., description="Snapshot content (parsed JSON or raw)")
    actor_employee_id: str | None = None
    actor_role: str | None = None
    previous_snapshot_id: str | None = None
    regenerated_at: str | datetime | None = None
    regeneration_reason: str | None = None
    source_changed: bool | None = None
    created_at: str | datetime

    model_config = ConfigDict(extra="allow")


class HRInsightHistoryData(BaseModel):
    """Public HR response data for AI insight history with pagination."""

    feature: str = Field(..., json_schema_extra={"example": "career_coach"})
    scope_employee_id: str | None = None
    scope_department: str | None = None
    period: str | None = None
    total_versions: int = Field(..., json_schema_extra={"example": 3})
    page: int = Field(default=1, json_schema_extra={"example": 1})
    page_size: int = Field(default=50, json_schema_extra={"example": 50})
    has_more: bool = Field(default=False, json_schema_extra={"example": False})
    snapshots: list[HRInsightSnapshotData] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow")


class HRFeedbackData(BaseModel):
    """Public HR response data for feedback entries."""

    id: str = Field(..., json_schema_extra={"example": "fb-12345"})
    snapshot_id: str = Field(..., json_schema_extra={"example": "snap-12345"})
    actor_employee_id: str = Field(..., json_schema_extra={"example": "EMP-001"})
    actor_role: str = Field(..., json_schema_extra={"example": "manager"})
    is_helpful: bool = Field(..., json_schema_extra={"example": True})
    feedback_text: str | None = None
    created_at: str | datetime

    model_config = ConfigDict(extra="allow")

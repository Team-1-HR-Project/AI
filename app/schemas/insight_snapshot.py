"""Pydantic schemas for AI Insight Snapshots, History, Regeneration, and Feedback."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class AIInsightFeature(str, Enum):
    """Published feature identifiers; prevents arbitrary snapshot partitions."""

    CAREER_COACH = "career_coach"
    PERFORMANCE_INSIGHT = "performance_insight"
    POLICY_ASSISTANT = "policy_assistant"
    EVALUATION_DRAFT = "evaluation_draft"
    SKILL_GAP = "skill_gap"
    ATTENTION_SIGNAL = "attention_signal"
    TEAM_INSIGHT = "team_insight"


class AIFeedbackCreateRequest(BaseModel):
    """Request model for submitting helpfulness and qualitative feedback on an AI insight."""

    is_helpful: bool = Field(
        ...,
        description="Whether the generated AI insight was helpful (True) or not helpful (False)",
    )
    feedback_text: str | None = Field(
        default=None,
        max_length=1000,
        description="Optional detailed feedback or suggested corrections",
    )

    model_config = ConfigDict(extra="forbid")


class AIFeedbackResponse(BaseModel):
    """Response model for a recorded AI feedback entry."""

    id: str
    snapshot_id: str
    actor_employee_id: str
    actor_role: str
    is_helpful: bool
    feedback_text: str | None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class AIInsightSnapshotResponse(BaseModel):
    """Response model for a durable AI insight snapshot version."""

    id: str
    generation_id: str
    version: int
    feature: str
    source_version: str | None = None
    source_hash: str | None = None
    context_hash: str | None = None
    prompt_hash: str | None = None
    provider: str | None = None
    model: str | None = None
    scope_employee_id: str | None = None
    scope_department: str | None = None
    period: str | None = None
    content: Any  # Parsed JSON structure or raw dict
    request_payload: Any | None = None
    actor_employee_id: str | None = None
    actor_role: str | None = None
    previous_snapshot_id: str | None = None
    regenerated_at: datetime | None = None
    regeneration_reason: str | None = None
    source_changed: bool | None = None
    created_at: datetime
    feedback_count: int = 0

    model_config = ConfigDict(from_attributes=True)


class AIInsightHistoryResponse(BaseModel):
    """Response model for version history of an AI insight."""

    feature: AIInsightFeature
    scope_employee_id: str | None = None
    scope_department: str | None = None
    period: str | None = None
    total_versions: int
    page: int = 1
    page_size: int = 50
    has_more: bool = False
    snapshots: list[AIInsightSnapshotResponse]

    model_config = ConfigDict(from_attributes=True)

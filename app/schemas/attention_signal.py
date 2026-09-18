"""Pydantic schemas for Feature #6: Employee Attention Signal.

Defines strict request models, attention levels, contributing indicators,
supportive human follow-up actions, safe insufficient-data responses,
and a discriminated union keyed on the "status" field.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag


def utc_now() -> datetime:
    """Returns the current timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


class AttentionLevel(str, Enum):
    """Attention level assigned to recent employee work indicators."""

    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class IndicatorCategory(str, Enum):
    """Recognized categories of employee work indicators."""

    ATTENDANCE = "attendance"
    TASK_COMPLETION = "task_completion"
    GOALS = "goals"
    EVALUATION_TREND = "evaluation_trend"


class ContributingIndicator(BaseModel):
    """Represents a specific work indicator contributing to the attention signal."""

    indicator_name: str = Field(
        ...,
        min_length=1,
        max_length=150,
        description="Name of the indicator, e.g. 'Attendance Rate', 'Task Completion Rate', 'Goal Progress'",
    )
    category: IndicatorCategory = Field(
        ...,
        description="Category of the indicator: attendance, task_completion, goals, or evaluation_trend",
    )
    current_value: str | float | int = Field(
        ...,
        description="Observed current value in the target period",
    )
    previous_value: str | float | int | None = Field(
        default=None,
        description="Observed value in the comparison period, if available",
    )
    change_description: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Objective description of observed change or trend (e.g. 'Declined from 95% to 82%')",
    )
    evidence: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Grounded evidence from approved employee records supporting this observation",
    )

    model_config = ConfigDict(extra="forbid")


class RecommendedFollowUpAction(BaseModel):
    """Represents an advisory, supportive human follow-up action for managers."""

    action_type: str = Field(
        ...,
        min_length=1,
        max_length=150,
        description="Type of human follow-up, e.g. '1-on-1 Check-in', 'Workload Review', 'Impediment Removal'",
    )
    description: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        description="Actionable, supportive guidance for human follow-up",
    )
    rationale: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        description="Reasoning linking the recommended action to observed indicators",
    )

    model_config = ConfigDict(extra="forbid")


class AttentionSignalRequest(BaseModel):
    """Input request model for generating an employee attention signal."""

    employee_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Target employee identifier (e.g. 'EMP-001')",
    )
    target_period: str | None = Field(
        default=None,
        min_length=1,
        max_length=50,
        description="Optional target evaluation period (e.g. '2026-Q3'). Defaults to latest approved period.",
    )

    model_config = ConfigDict(extra="forbid")


class AttentionSignalSuccessResponse(BaseModel):
    """Full API success response for an employee attention signal."""

    status: Literal["success"] = "success"
    employee_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Identifier of the evaluated employee",
    )
    target_period: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Primary period evaluated",
    )
    comparison_period: str | None = Field(
        default=None,
        max_length=50,
        description="Prior period used for comparison, if available",
    )
    attention_level: AttentionLevel = Field(
        ...,
        description="Low, Medium, or High attention level",
    )
    explanation: str = Field(
        ...,
        min_length=5,
        max_length=2000,
        description="Objective explanation of why the employee was flagged with this attention level",
    )
    contributing_indicators: list[ContributingIndicator] = Field(
        default_factory=list,
        description="List of work indicators contributing to the attention signal",
    )
    recommended_follow_up: list[RecommendedFollowUpAction] = Field(
        default_factory=list,
        description="Recommended human follow-up actions (advisory only)",
    )
    advisory_only: bool = Field(
        default=True,
        description="Explicit flag confirming this signal is strictly advisory and non-binding",
    )
    human_review_required: bool = Field(
        default=True,
        description="Explicit flag indicating human review is required before taking any action",
    )
    created_at: datetime = Field(
        default_factory=utc_now,
        description="ISO-8601 UTC timestamp of response generation",
    )

    model_config = ConfigDict(extra="forbid")


class AttentionSignalInsufficientDataResponse(BaseModel):
    """Safe fallback response when approved employee records are insufficient."""

    status: Literal["insufficient_data"] = "insufficient_data"
    employee_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Identifier of the employee lacking approved data",
    )
    reason: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        description="Human-readable explanation of why the attention signal cannot be generated",
    )
    target_period: str | None = Field(
        default=None,
        max_length=50,
        description="Target period evaluated, if specified",
    )
    comparison_period: str | None = Field(
        default=None,
        max_length=50,
        description="Comparison period evaluated, if any",
    )
    created_at: datetime = Field(
        default_factory=utc_now,
        description="ISO-8601 UTC timestamp of response generation",
    )

    model_config = ConfigDict(extra="forbid")


AttentionSignalResponse = Annotated[
    Annotated[AttentionSignalSuccessResponse, Tag("success")]
    | Annotated[AttentionSignalInsufficientDataResponse, Tag("insufficient_data")],
    Discriminator("status"),
]


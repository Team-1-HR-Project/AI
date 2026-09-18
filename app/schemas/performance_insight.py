"""Pydantic schemas for the Performance Insight Generator service.

Defines strict request models, factual metric structures, deterministic trend models,
AI interpretation structures with safe contributing indicators, actionable review recommendations,
fallback insufficient-data models, and a discriminated union keyed on the "status" field.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag

from app.schemas.career_coach import PriorityLevel


def utc_now() -> datetime:
    """Returns the current timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


class TrendDirection(str, Enum):
    """Deterministic direction of performance metric change."""

    IMPROVED = "improved"
    DECLINED = "declined"
    STABLE = "stable"


# 1. Request Schema
class PerformanceInsightRequest(BaseModel):
    """Input request model for generating performance insights."""

    employee_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Target employee identifier (e.g. 'EMP-001')",
    )
    period: str | None = Field(
        default=None,
        min_length=1,
        max_length=50,
        description="Optional target evaluation period (e.g. '2026-Q3'). If omitted, latest approved period is used.",
    )

    model_config = ConfigDict(extra="forbid")


# 2. Verified Facts / Actual Metrics
class PerformancePeriodMetrics(BaseModel):
    """Authoritative, verified factual metrics from an approved performance record."""

    period: str = Field(
        ...,
        min_length=1,
        max_length=20,
        description="Evaluation period identifier (e.g. '2026-Q3')",
    )
    overall_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Overall performance score (0 to 100)",
    )
    task_completion_rate: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Percentage of assigned tasks completed (0 to 100)",
    )
    goal_achievement_rate: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Percentage of goals achieved (0 to 100)",
    )
    attendance_rate: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Attendance percentage rate (0 to 100)",
    )

    model_config = ConfigDict(extra="forbid")


class VerifiedFacts(BaseModel):
    """Authoritative factual data retrieved directly from approved application records."""

    target_period: str = Field(
        ...,
        min_length=1,
        max_length=20,
        description="Target evaluation period being evaluated",
    )
    comparison_period: str | None = Field(
        default=None,
        max_length=20,
        description="Prior baseline period used for comparison (if available)",
    )
    metrics_by_period: list[PerformancePeriodMetrics] = Field(
        ...,
        min_length=1,
        description="Chronological verified performance records for this employee",
    )

    model_config = ConfigDict(extra="forbid")


# 3. Calculated Trends (Deterministic, Mathematical)
class MetricTrend(BaseModel):
    """Deterministic, mathematically computed trend for a specific metric across periods."""

    metric_name: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Name of the evaluated performance metric",
    )
    previous_value: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Baseline metric value in comparison period",
    )
    current_value: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Current metric value in target period",
    )
    delta: float = Field(
        ...,
        description="Absolute change (current_value - previous_value)",
    )
    direction: TrendDirection = Field(
        ...,
        description="Deterministic direction: improved, declined, or stable",
    )
    percent_change: float | None = Field(
        default=None,
        description="Percentage change relative to previous_value (None if previous_value is 0)",
    )

    model_config = ConfigDict(extra="forbid")


class CalculatedTrends(BaseModel):
    """Deterministic mathematical trends computed across periods without LLM involvement."""

    from_period: str = Field(
        ...,
        min_length=1,
        max_length=20,
        description="Baseline period compared against",
    )
    to_period: str = Field(
        ...,
        min_length=1,
        max_length=20,
        description="Target period evaluated",
    )
    metrics: dict[str, MetricTrend] = Field(
        ...,
        description="Deterministic trends keyed by metric name",
    )
    improved_metrics: list[str] = Field(
        default_factory=list,
        description="List of metric names that deterministically improved",
    )
    declined_metrics: list[str] = Field(
        default_factory=list,
        description="List of metric names that deterministically declined",
    )
    stable_metrics: list[str] = Field(
        default_factory=list,
        description="List of metric names that remained stable",
    )

    model_config = ConfigDict(extra="forbid")


# 4. AI Interpretation (Qualitative Analysis with Safe Indicator Semantics)
class ContributingIndicator(BaseModel):
    """Observed operational indicator or related context to review.

    SAFETY RULE: This represents an observed correlation or operational factor
    to review, NOT a proven or verified root cause.
    """

    indicator_name: str = Field(
        ...,
        min_length=2,
        max_length=150,
        description="Name of the observed indicator or context factor",
    )
    category: str = Field(
        ...,
        min_length=2,
        max_length=50,
        description="Category of indicator (e.g., 'task_complexity', 'goal_scope', 'attendance', 'workload')",
    )
    observation: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="Observed correlation or context factor to review (never framed as a verified cause)",
    )

    model_config = ConfigDict(extra="forbid")


class PerformanceImprovementItem(BaseModel):
    """Details an identified area of performance improvement."""

    metric: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="Metric or operational area that improved",
    )
    summary: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="Analytical summary of the observed improvement",
    )
    contributing_indicators: list[ContributingIndicator] = Field(
        default_factory=list,
        description="Candidate indicators or operational factors to review (not verified causes)",
    )

    model_config = ConfigDict(extra="forbid")


class PerformanceDeclineItem(BaseModel):
    """Details an identified area of performance decline."""

    metric: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="Metric or operational area that declined",
    )
    summary: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="Analytical summary of the observed decline",
    )
    contributing_indicators: list[ContributingIndicator] = Field(
        default_factory=list,
        description="Candidate indicators or operational factors to review (not verified causes)",
    )

    model_config = ConfigDict(extra="forbid")


class AIInterpretation(BaseModel):
    """AI-synthesized qualitative analysis.

    Preserves explicit boundaries: interpretations are qualitative hypotheses
    and observations to review, distinct from verified facts and computed trends.
    """

    summary: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="Qualitative analytical summary of performance changes across periods",
    )
    improvements: list[PerformanceImprovementItem] = Field(
        default_factory=list,
        description="Areas where performance improved across evaluated periods",
    )
    declines: list[PerformanceDeclineItem] = Field(
        default_factory=list,
        description="Areas where performance declined across evaluated periods",
    )

    model_config = ConfigDict(extra="forbid")


# 5. Suggested Review / Action
class ReviewActionItem(BaseModel):
    """Actionable recommendation for what management/employee should review next."""

    priority: PriorityLevel = Field(
        ...,
        description="Priority of the review action (high, medium, low)",
    )
    focus_area: str = Field(
        ...,
        min_length=2,
        max_length=150,
        description="Operational area or metric focus of the review",
    )
    recommended_action: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="Concrete review step, check-in, or follow-up action to take",
    )
    rationale: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="Operational rationale for conducting this review",
    )

    model_config = ConfigDict(extra="forbid")


# 6. Public Success Response
class PerformanceInsightSuccessResponse(BaseModel):
    """Authoritative successful response for Performance Insight generation."""

    status: Literal["success"] = Field(
        "success",
        description="Status indicator for successful performance insight",
    )
    employee_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Target employee identifier",
    )
    verified_facts: VerifiedFacts = Field(
        ...,
        description="Authoritative verified factual metrics from approved records",
    )
    calculated_trends: CalculatedTrends = Field(
        ...,
        description="Deterministic mathematical trends computed across periods",
    )
    ai_interpretation: AIInterpretation = Field(
        ...,
        description="AI qualitative analysis (what improved, what declined, contributing indicators)",
    )
    suggested_review_actions: list[ReviewActionItem] = Field(
        ...,
        min_length=1,
        max_length=10,
        description="Prioritized recommendations for what should be reviewed next",
    )
    created_at: datetime = Field(
        default_factory=utc_now,
        description="Timestamp of generation (authoritatively set by application)",
    )

    model_config = ConfigDict(extra="forbid")


# 7. Insufficient Data Fallback Response
class PerformanceInsightInsufficientDataResponse(BaseModel):
    """Structured fallback response when insufficient approved data exists."""

    status: Literal["insufficient_data"] = Field(
        "insufficient_data",
        description="Status indicator for insufficient data fallback",
    )
    employee_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Target employee identifier",
    )
    reason: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="Detailed explanation of why performance insights could not be generated",
    )
    periods_found: list[str] = Field(
        default_factory=list,
        description="List of approved periods found for the employee (if any)",
    )
    message: str = Field(
        default="Insufficient approved performance data to generate comparative performance insights.",
        max_length=500,
        description="User-friendly explanation message",
    )
    created_at: datetime = Field(
        default_factory=utc_now,
        description="Timestamp of generation (authoritatively set by application)",
    )

    model_config = ConfigDict(extra="forbid")


# 8. Discriminated Union for API response handling keyed by 'status'
PerformanceInsightResponse = Annotated[
    Annotated[PerformanceInsightSuccessResponse, Tag("success")]
    | Annotated[PerformanceInsightInsufficientDataResponse, Tag("insufficient_data")],
    Discriminator("status"),
]


# 9. LLM Generation Schema (Internal output structure from AI prompt before server assembly)
class PerformanceInsightAIGeneration(BaseModel):
    """Raw structured output schema expected from LLM generation."""

    summary: str = Field(..., min_length=10, max_length=2000)
    improvements: list[PerformanceImprovementItem] = Field(default_factory=list)
    declines: list[PerformanceDeclineItem] = Field(default_factory=list)
    suggested_review_actions: list[ReviewActionItem] = Field(
        ...,
        min_length=1,
        max_length=10,
    )

    model_config = ConfigDict(extra="ignore")


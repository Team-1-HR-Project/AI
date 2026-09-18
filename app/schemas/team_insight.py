"""Pydantic schemas for Feature #7: Team Insight Summary - Manager.

Defines strict request models, aggregated team findings, overdue workload metrics,
completion trends, skill-gap patterns, evaluation-theme summaries, drill-down factors,
recommended management actions, and safe insufficient-data fallback responses.

Privacy Guarantees:
The schemas strictly exclude individual employee names, first/last names, employee IDs,
salary, compensation, disciplinary statuses, or flight-risk predictions from team-level output.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag

from app.schemas.career_coach import PriorityLevel
from app.schemas.performance_insight import TrendDirection


def utc_now() -> datetime:
    """Returns the current timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


class DrillDownCategory(str, Enum):
    """Categorization for supporting drill-down observations."""

    WORKLOAD_BLOCKERS = "workload_blockers"
    TASK_COMPLETION = "task_completion"
    SKILL_DEVELOPMENT = "skill_development"
    EVALUATION_THEMES = "evaluation_themes"


# =============================================================================
# 1. Request Schema
# =============================================================================


class TeamInsightRequest(BaseModel):
    """Input request model for generating a manager-facing Team Insight Summary."""

    department: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Target department or team name (e.g. 'Engineering', 'Sales')",
    )
    period: str | None = Field(
        default=None,
        min_length=1,
        max_length=50,
        description="Optional evaluation period (e.g. '2026-Q3'). If omitted, latest approved period is used.",
    )

    model_config = ConfigDict(extra="forbid")


# =============================================================================
# 2. Component Finding Schemas
# =============================================================================


class OverdueWorkloadSummary(BaseModel):
    """Aggregated team-level overdue and blocked workload patterns."""

    summary: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        description="Objective narrative summary of overdue work, blocked tasks, and delayed milestones",
    )
    total_blocked_tasks: int = Field(
        ...,
        ge=0,
        description="Total number of approved tasks currently blocked across the team",
    )
    total_delayed_goals: int = Field(
        ...,
        ge=0,
        description="Total number of approved goals currently delayed or under pacing across the team",
    )
    affected_member_count: int = Field(
        ...,
        ge=0,
        description="Number of team members currently experiencing blocked work or delayed goals",
    )

    model_config = ConfigDict(extra="forbid")


class CompletionTrendsSummary(BaseModel):
    """Aggregated team-level task completion and performance metrics."""

    summary: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        description="Objective narrative of team task completion trajectory and delivery trends",
    )
    team_avg_task_completion: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Average task completion rate across approved team records (0 to 100)",
    )
    team_avg_goal_achievement: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Average goal achievement rate across approved team records (0 to 100)",
    )
    team_avg_overall_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Average overall performance score across approved team records (0 to 100)",
    )
    direction: TrendDirection = Field(
        ...,
        description="Period-over-period aggregate team trajectory: improved, declined, or stable",
    )

    model_config = ConfigDict(extra="forbid")


class SkillGapPatternsSummary(BaseModel):
    """Aggregated team-level skill-gap patterns and common learning needs."""

    summary: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        description="Summary of common technical or professional competencies identified for team growth",
    )
    top_common_gaps: list[str] = Field(
        default_factory=list,
        description="List of most prevalent skill gaps or development priorities across team members",
    )

    model_config = ConfigDict(extra="forbid")


class EvaluationThemePatternsSummary(BaseModel):
    """Aggregated patterns across approved employee evaluation themes."""

    summary: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        description="Synthesis of recurring strengths and growth areas observed in evaluation themes",
    )
    top_positive_themes: list[str] = Field(
        default_factory=list,
        description="Most frequent positive evaluation themes across the team",
    )
    top_needs_improvement_themes: list[str] = Field(
        default_factory=list,
        description="Most frequent needs_improvement evaluation themes across the team",
    )

    model_config = ConfigDict(extra="forbid")


class TeamFindings(BaseModel):
    """Consolidated macro team findings across all operational dimensions."""

    executive_summary: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="Executive, manager-facing synthesis of team performance, workload, and operational health",
    )
    overdue_workload: OverdueWorkloadSummary = Field(
        ...,
        description="Overdue workload and operational blocker metrics",
    )
    completion_trends: CompletionTrendsSummary = Field(
        ...,
        description="Task completion and metric trends across the team",
    )
    skill_gap_patterns: SkillGapPatternsSummary = Field(
        ...,
        description="Common skill gap and development patterns",
    )
    evaluation_theme_patterns: EvaluationThemePatternsSummary = Field(
        ...,
        description="Evaluation theme feedback patterns and recurring themes",
    )

    model_config = ConfigDict(extra="forbid")


# =============================================================================
# 3. Drill-Down Factors & Management Action Schemas
# =============================================================================


class DrillDownFactor(BaseModel):
    """Anonymized supporting factor or observation backing the team-level findings.

    Privacy Guarantee:
    Does NOT include individual employee names, first/last names, or employee IDs.
    Uses anonymized role titles (e.g. 'Backend Engineer') for context.
    """

    category: DrillDownCategory = Field(
        ...,
        description="Category of the drill-down factor (workload_blockers, task_completion, skill_development, evaluation_themes)",
    )
    factor_title: str = Field(
        ...,
        min_length=1,
        max_length=150,
        description="Concise title for the observed factor (e.g. 'External IAM Dependency Block')",
    )
    observation: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        description="Objective observation grounded in approved team records",
    )
    supporting_metrics: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Factual, grounded metrics or record references supporting this observation",
    )
    anonymized_role: str | None = Field(
        default=None,
        max_length=100,
        description="Anonymized role title associated with the observation (e.g. 'Software Engineer'), never employee PII",
    )

    model_config = ConfigDict(extra="forbid")


class RecommendedManagementAction(BaseModel):
    """Supportive, advisory follow-up action for managers to address team findings."""

    action_title: str = Field(
        ...,
        min_length=1,
        max_length=150,
        description="Concise title of the recommended managerial action",
    )
    description: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        description="Actionable, constructive guidance to resolve friction, unblock work, or sponsor training",
    )
    priority: PriorityLevel = Field(
        ...,
        description="Priority level for management follow-up: high, medium, or low",
    )

    model_config = ConfigDict(extra="forbid")


# =============================================================================
# 4. Success & Fallback Responses
# =============================================================================


class TeamInsightSuccessResponse(BaseModel):
    """Successful manager-facing Team Insight Summary."""

    status: Literal["success"] = "success"
    department: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Department or team evaluated",
    )
    period: str | None = Field(
        default=None,
        max_length=50,
        description="Evaluation period analyzed (e.g. '2026-Q3')",
    )
    team_size: int = Field(
        ...,
        ge=0,
        description="Total number of employees evaluated in this team or department",
    )
    team_findings: TeamFindings = Field(
        ...,
        description="Consolidated macro team findings across workload, trends, skills, and themes",
    )
    drill_down_factors: list[DrillDownFactor] = Field(
        default_factory=list,
        description="Supporting anonymized drill-down factors grounding the team-level findings",
    )
    recommended_management_actions: list[RecommendedManagementAction] = Field(
        default_factory=list,
        description="Constructive, advisory actions for managers to address team bottlenecks or development gaps",
    )
    advisory_disclaimer: str = Field(
        default=(
            "This team insight summary is an AI-assisted advisory analysis synthesized from approved records. "
            "It does not constitute formal employee evaluations, compensation decisions, or disciplinary actions."
        ),
        description="Mandatory disclaimer affirming advisory nature",
    )
    created_at: datetime = Field(
        default_factory=utc_now,
        description="ISO-8601 UTC timestamp of response generation",
    )

    model_config = ConfigDict(extra="forbid")


class TeamInsightInsufficientDataResponse(BaseModel):
    """Safe fallback response when approved team records are insufficient."""

    status: Literal["insufficient_data"] = "insufficient_data"
    department: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Department or team requested",
    )
    period: str | None = Field(
        default=None,
        max_length=50,
        description="Evaluation period evaluated, if specified",
    )
    missing_categories: list[str] = Field(
        default_factory=list,
        description="Categories lacking sufficient approved records (e.g. ['employees', 'performance'])",
    )
    message: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        description="Human-readable explanation of why the team insight summary cannot be generated",
    )
    created_at: datetime = Field(
        default_factory=utc_now,
        description="ISO-8601 UTC timestamp of response generation",
    )

    model_config = ConfigDict(extra="forbid")


# =============================================================================
# 5. Discriminated Union Response Model
# =============================================================================

TeamInsightResponse = Annotated[
    Annotated[TeamInsightSuccessResponse, Tag("success")]
    | Annotated[TeamInsightInsufficientDataResponse, Tag("insufficient_data")],
    Discriminator("status"),
]


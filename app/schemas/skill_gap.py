"""Pydantic schemas for AI #4: Skill-Gap & Development Recommendations.

Defines strict request models, grounded skill-gap items with evidence references,
advisory learning recommendations, safe insufficient data fallback responses,
and discriminated union response structures.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag

from app.schemas.career_coach import EvidenceItem, PriorityLevel


def utc_now() -> datetime:
    """Returns the current timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


class SkillLevel(str, Enum):
    """Proficiency levels recognized in the system."""

    BEGINNER = "Beginner"
    INTERMEDIATE = "Intermediate"
    ADVANCED = "Advanced"
    EXPERT = "Expert"


class GapSeverity(str, Enum):
    """Severity or urgency of the identified skill gap."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class LearningType(str, Enum):
    """Recognized developmental and educational modalities."""

    TRAINING_COURSE = "training_course"
    CERTIFICATION = "certification"
    MENTORSHIP = "mentorship"
    PEER_SHADOWING = "peer_shadowing"
    HANDS_ON_PROJECT = "hands_on_project"
    SELF_PACED_STUDY = "self_paced_study"
    WORKSHOP = "workshop"


# 1. Request Schema
class SkillGapRequest(BaseModel):
    """Input request model for generating skill-gap and development recommendations."""

    employee_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Target employee identifier (e.g. 'EMP-001')",
    )
    period: str | None = Field(
        default=None,
        max_length=50,
        description="Optional reporting/review period filter (e.g. '2026-Q3')",
    )
    target_role: str | None = Field(
        default=None,
        max_length=100,
        description="Optional desired target role or career aspiration (e.g. 'Senior Backend Engineer')",
    )
    target_skills: list[str] | None = Field(
        default=None,
        max_length=10,
        description="Optional specific target skills to evaluate (max 10 skills, e.g. ['Kubernetes', 'System Design'])",
    )

    model_config = ConfigDict(extra="forbid")


# 2. Skill Gap Item
class SkillGapItem(BaseModel):
    """Represents an identified skill gap grounded in approved employee records."""

    skill_name: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="Name of the skill with an identified proficiency gap",
    )
    current_level: str = Field(
        ...,
        min_length=2,
        max_length=50,
        description="Observed current level based on approved context (e.g. 'Beginner', 'Intermediate', 'Not Observed')",
    )
    desired_level: str = Field(
        ...,
        min_length=2,
        max_length=50,
        description="Target proficiency required for the role or objective (e.g. 'Advanced', 'Expert')",
    )
    gap_severity: GapSeverity = Field(
        ...,
        description="Urgency/impact of the gap: 'critical', 'high', 'medium', or 'low'",
    )
    rationale: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        description="Factual rationale explaining why this gap exists based on approved records",
    )
    evidence: list[EvidenceItem] = Field(
        ...,
        min_length=1,
        max_length=5,
        description="Deterministic source-referenced evidence items demonstrating the gap",
    )

    model_config = ConfigDict(extra="forbid")


# 3. Development Recommendation Item
class SkillRecommendationItem(BaseModel):
    """Actionable, advisory learning and development recommendation."""

    title: str = Field(
        ...,
        min_length=3,
        max_length=150,
        description="Short title of the recommended learning activity",
    )
    learning_type: LearningType = Field(
        ...,
        description="Format of the learning activity (e.g. 'training_course', 'mentorship', 'hands_on_project')",
    )
    focus_skill: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="The primary skill targeted by this development recommendation",
    )
    description: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        description="Detailed description of the recommended training, practice, or resource",
    )
    expected_outcome: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="Measurable capability or outcome the employee should achieve upon completion",
    )
    measurable_target: str = Field(
        ...,
        min_length=5,
        max_length=300,
        description="Concrete, measurable follow-up milestone, deliverable, or metric to track completion",
    )
    timeline: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="Suggested timeframe for completion (e.g. '4-6 weeks', 'Q4 2026')",
    )
    priority: PriorityLevel = Field(
        ...,
        description="Development priority: 'high', 'medium', or 'low'",
    )

    model_config = ConfigDict(extra="forbid")


# 4. LLM Output Structure (Internal Parsing)
class SkillGapModelOutput(BaseModel):
    """Raw structured output generated by the LLM before backend enrichment."""

    status: Literal["success"] = "success"
    skill_gaps: list[SkillGapItem] = Field(
        default_factory=list,
        max_length=10,
        description="Identified skill gaps grounded in context",
    )
    recommendations: list[SkillRecommendationItem] = Field(
        default_factory=list,
        max_length=10,
        description="Prioritized learning and training recommendations",
    )

    model_config = ConfigDict(extra="forbid")


# 5. Success Response Schema
class SkillGapSuccessResponse(BaseModel):
    """Full API success response returned to the client."""

    status: Literal["success"] = "success"
    employee_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Identifier of the evaluated employee",
    )
    period: str | None = Field(
        default=None,
        description="Evaluated period if specified",
    )
    target_role: str | None = Field(
        default=None,
        description="Target role evaluated against, if requested",
    )
    skill_gaps: list[SkillGapItem] = Field(
        ...,
        description="List of identified skill gaps backed by approved evidence",
    )
    recommendations: list[SkillRecommendationItem] = Field(
        ...,
        description="Grounded development recommendations tailored to the identified gaps",
    )
    disclaimer: str = Field(
        default=(
            "This skill-gap analysis and development plan is AI-generated and advisory. "
            "It does not constitute a formal performance appraisal or employment decision."
        ),
        description="Mandatory advisory disclaimer",
    )
    created_at: datetime = Field(
        default_factory=utc_now,
        description="ISO-8601 UTC timestamp of generation",
    )

    model_config = ConfigDict(extra="forbid")


# 6. Insufficient Data Response Schema
class SkillGapInsufficientDataResponse(BaseModel):
    """Safe fallback response when approved employee data is insufficient."""

    status: Literal["insufficient_data"] = "insufficient_data"
    employee_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Identifier of the employee lacking approved data",
    )
    missing_categories: list[str] = Field(
        default_factory=list,
        description="Data categories lacking approved records required for skill-gap analysis",
    )
    message: str = Field(
        default="Not enough approved employee data to generate a reliable skill-gap analysis.",
        description="Human-readable explanation of why the analysis cannot be reliably performed",
    )
    created_at: datetime = Field(
        default_factory=utc_now,
        description="ISO-8601 UTC timestamp of response generation",
    )

    model_config = ConfigDict(extra="forbid")


# 7. Discriminated Union Response
SkillGapResponse = Annotated[
    Annotated[SkillGapSuccessResponse, Tag("success")]
    | Annotated[SkillGapInsufficientDataResponse, Tag("insufficient_data")],
    Discriminator("status"),
]


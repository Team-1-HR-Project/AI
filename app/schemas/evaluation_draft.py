"""Pydantic schemas for the Evaluation Draft Assistant service.

Defines strict request models, evaluation items with grounded evidence references,
editable narrative structures, human review flags, insufficient data responses,
and discriminated union response structures.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag, field_validator

from app.schemas.career_coach import (
    EvidenceItem,
    PriorityLevel,
)


def utc_now() -> datetime:
    """Returns the current timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


# 1. Request Schema
class EvaluationDraftRequest(BaseModel):
    """Input request model for generating an evaluation draft."""

    employee_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Target employee identifier (e.g. 'EMP-001')",
    )
    period: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Evaluation period being reviewed (e.g. '2026-Q3')",
    )
    evaluation_scores: dict[str, float] | None = Field(
        default=None,
        description="Optional manager-entered evaluation scores (e.g. {'overall_score': 88.0, 'leadership': 85.0})",
    )
    manager_notes: str | None = Field(
        default=None,
        max_length=3000,
        description="Optional manager notes, observations, or preliminary feedback to integrate into the draft",
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("evaluation_scores")
    @classmethod
    def validate_scores(cls, v: dict[str, float] | None) -> dict[str, float] | None:
        if v is not None:
            for key, score in v.items():
                if not (0.0 <= score <= 100.0):
                    raise ValueError(f"Score for '{key}' must be between 0.0 and 100.0, got {score}")
        return v


# 2. Evaluation Strength Item
class EvaluationStrengthItem(BaseModel):
    """Key strength observed during the evaluation period."""

    title: str = Field(
        ...,
        min_length=2,
        max_length=150,
        description="Concise title of the observed strength",
    )
    description: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        description="Detailed description of the strength and observed impact",
    )
    evidence: list[EvidenceItem] = Field(
        ...,
        min_length=1,
        description="Grounded evidence items citing approved source records",
    )

    model_config = ConfigDict(extra="forbid")


# 3. Evaluation Improvement Item
class EvaluationImprovementItem(BaseModel):
    """Specific growth area or performance opportunity."""

    title: str = Field(
        ...,
        min_length=2,
        max_length=150,
        description="Concise title of the improvement opportunity",
    )
    description: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        description="Constructive explanation of the growth gap and recommended focus",
    )
    evidence: list[EvidenceItem] = Field(
        ...,
        min_length=1,
        description="Grounded evidence items citing approved source records",
    )
    priority: PriorityLevel = Field(
        default=PriorityLevel.MEDIUM,
        description="Priority level for improvement ('high', 'medium', 'low')",
    )

    model_config = ConfigDict(extra="forbid")


# 4. Raw LLM Model Output Schema
class EvaluationDraftModelOutput(BaseModel):
    """Intermediate model to parse and validate raw LLM JSON response."""

    evaluation_narrative: str = Field(
        ...,
        min_length=10,
        max_length=4000,
        description="Cohesive, professional draft evaluation narrative",
    )
    strengths: list[EvaluationStrengthItem] = Field(
        ...,
        min_length=1,
        description="Identified employee strengths grounded in approved context",
    )
    improvement_areas: list[EvaluationImprovementItem] = Field(
        ...,
        min_length=1,
        description="Constructive development opportunities grounded in approved context",
    )

    model_config = ConfigDict(extra="forbid")


# 5. Success Response Schema
class EvaluationDraftSuccessResponse(BaseModel):
    """Complete successful response containing the evaluation draft."""

    status: Literal["success"] = "success"
    employee_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Identifier of the evaluated employee",
    )
    period: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Evaluation period reviewed",
    )
    evaluation_narrative: str = Field(
        ...,
        description="Comprehensive, professional draft evaluation narrative suitable for manager editing",
    )
    strengths: list[EvaluationStrengthItem] = Field(
        ...,
        description="Key strengths observed during the period",
    )
    improvement_areas: list[EvaluationImprovementItem] = Field(
        ...,
        description="Targeted growth opportunities for the employee",
    )
    entered_scores: dict[str, float] | None = Field(
        default=None,
        description="Scores provided in the request or noted in the draft",
    )
    human_review_required: bool = Field(
        default=True,
        description="Explicit indicator that human manager review, editing, and approval is required",
    )
    review_disclaimer: str = Field(
        default=(
            "This evaluation is an AI-generated draft intended solely to assist manager review. "
            "A human manager must review, edit, and approve this evaluation before any official use or persistence."
        ),
        description="Mandatory disclaimer affirming draft status and human review requirement",
    )
    created_at: datetime = Field(
        default_factory=utc_now,
        description="ISO-8601 UTC timestamp of draft generation",
    )

    model_config = ConfigDict(extra="forbid")


# 6. Insufficient Data Response Schema
class EvaluationDraftInsufficientDataResponse(BaseModel):
    """Safe fallback response when approved employee data is insufficient."""

    status: Literal["insufficient_data"] = "insufficient_data"
    employee_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Identifier of the employee lacking data",
    )
    period: str | None = Field(
        default=None,
        description="Requested evaluation period",
    )
    missing_categories: list[str] = Field(
        default_factory=list,
        description="Categories lacking approved records required for an evaluation draft",
    )
    message: str = Field(
        default="Not enough approved employee data to generate a reliable evaluation draft.",
        description="Human-readable explanation of why a draft cannot be reliably generated",
    )
    human_review_required: bool = Field(
        default=False,
        description="Indicates no human review needed because no draft was produced",
    )
    created_at: datetime = Field(
        default_factory=utc_now,
        description="ISO-8601 UTC timestamp of the response",
    )

    model_config = ConfigDict(extra="forbid")


# 7. Discriminated Union Response
EvaluationDraftResponse = Annotated[
    Annotated[EvaluationDraftSuccessResponse, Tag("success")] | Annotated[EvaluationDraftInsufficientDataResponse, Tag("insufficient_data")],
    Discriminator("status"),
]


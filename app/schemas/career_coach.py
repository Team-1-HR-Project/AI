from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag, field_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class PriorityLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


CANONICAL_SOURCE_TYPES = {
    "performance": "performance",
    "performance_record": "performance",
    "performance_records": "performance",
    "goal": "goal",
    "goals": "goal",
    "skill": "skill",
    "skills": "skill",
    "task": "task_outcome",
    "tasks": "task_outcome",
    "task_outcome": "task_outcome",
    "task_outcomes": "task_outcome",
    "evaluation_theme": "evaluation_theme",
    "evaluation_themes": "evaluation_theme",
    "theme": "evaluation_theme",
    "themes": "evaluation_theme",
}


# Request Schema
class CareerCoachRequest(BaseModel):
    """Input request model for generating a career development plan."""

    employee_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Target employee identifier (e.g. 'EMP-001')",
    )
    period: str | None = Field(
        default=None,
        max_length=50,
        description="Optional performance/reporting period filter (e.g. '2026-Q3')",
    )

    model_config = ConfigDict(extra="forbid")


# 0. Deterministic Grounded Evidence Reference
class EvidenceItem(BaseModel):
    source_type: str = Field(
        ...,
        min_length=2,
        max_length=50,
        description="Type of source record (e.g. 'performance', 'goal', 'skill', 'task_outcome', 'evaluation_theme')",
    )
    source_id: int = Field(
        ...,
        description="ID of the approved source record from employee context",
    )
    claim: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Concrete factual claim supported by the source record",
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("source_type", mode="before")
    @classmethod
    def normalize_source_type(cls, v: Any) -> str:
        if isinstance(v, str):
            cleaned = v.strip().lower()
            return CANONICAL_SOURCE_TYPES.get(cleaned, cleaned)
        return str(v)


# 1. Strength Item
class StrengthItem(BaseModel):
    title: str = Field(
        ...,
        min_length=2,
        max_length=150,
        description="Short title of the employee strength",
    )
    description: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        description="Clear description of the strength observed",
    )
    evidence: list[EvidenceItem] = Field(
        ...,
        min_length=1,
        max_length=5,
        description="Deterministic source-referenced evidence items supporting the strength",
    )

    model_config = ConfigDict(extra="forbid")


# 2. Development Area Item
class DevelopmentAreaItem(BaseModel):
    title: str = Field(
        ...,
        min_length=2,
        max_length=150,
        description="Focus area requiring development",
    )
    description: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        description="Specific gap or growth opportunity",
    )
    evidence: list[EvidenceItem] = Field(
        ...,
        min_length=1,
        max_length=5,
        description="Deterministic source-referenced evidence items demonstrating the gap",
    )
    priority: PriorityLevel = Field(
        ...,
        description="Priority level: high, medium, or low",
    )

    model_config = ConfigDict(extra="forbid")


# 3. Development Plan Item
class DevelopmentPlanAction(BaseModel):
    action: str = Field(
        ...,
        min_length=3,
        max_length=300,
        description="Specific practical action for the employee",
    )
    reason: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        description="Why this action is prescribed based on data",
    )
    measurable_target: str = Field(
        ...,
        min_length=3,
        max_length=300,
        description="Concrete milestone or metric to track completion",
    )
    suggested_timeline: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="Expected completion timeframe (e.g., '30 days')",
    )

    model_config = ConfigDict(extra="forbid")


# 4. Follow-up
class FollowUp(BaseModel):
    checkpoint: str = Field(
        ...,
        min_length=3,
        max_length=150,
        description="Target review date or milestone interval",
    )
    review_focus: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="Key criteria to evaluate during the check-in",
    )

    model_config = ConfigDict(extra="forbid")


# 5. Schema for LLM Model Generation (Strictly omits employee_id and created_at)
class CareerCoachModelOutput(BaseModel):
    status: Literal["success"] = Field("success", description="Status indicator")
    development_focus: str | None = Field(
        default=None,
        min_length=3,
        max_length=500,
        description="Overall development focus for the next review period",
    )
    strengths: list[StrengthItem] = Field(
        ...,
        min_length=1,
        max_length=5,
        description="Identified strengths",
    )
    development_areas: list[DevelopmentAreaItem] = Field(
        ...,
        min_length=1,
        max_length=5,
        description="Identified development areas",
    )
    development_plan: list[DevelopmentPlanAction] = Field(
        ...,
        min_length=1,
        max_length=5,
        description="Actionable improvement plan",
    )
    follow_up: FollowUp = Field(
        ...,
        description="Follow-up milestone and review points",
    )

    # Ignore extra fields from LLM (such as attempted employee_id or created_at tampering)
    model_config = ConfigDict(extra="ignore")


# 6. Public Successful Career Coach Response (Includes authoritative employee_id and created_at)
class CareerCoachSuccessResponse(BaseModel):
    status: Literal["success"] = Field("success", description="Status indicator")
    employee_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Target employee identifier (authoritatively assigned by service)",
    )
    development_focus: str | None = Field(
        default=None,
        min_length=3,
        max_length=500,
        description="Overall development focus for the next review period",
    )
    strengths: list[StrengthItem] = Field(
        ...,
        min_length=1,
        max_length=5,
        description="Identified strengths",
    )
    development_areas: list[DevelopmentAreaItem] = Field(
        ...,
        min_length=1,
        max_length=5,
        description="Identified development areas",
    )
    development_plan: list[DevelopmentPlanAction] = Field(
        ...,
        min_length=1,
        max_length=5,
        description="Actionable improvement plan",
    )
    follow_up: FollowUp = Field(
        ...,
        description="Follow-up milestone and review points",
    )
    created_at: datetime = Field(
        default_factory=utc_now,
        description="Timestamp of generation (authoritatively generated by application)",
    )

    model_config = ConfigDict(extra="forbid")


# 7. Structured Fallback Response for Insufficient Data
class CareerCoachInsufficientDataResponse(BaseModel):
    status: Literal["insufficient_data"] = Field("insufficient_data", description="Status indicator")
    employee_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Target employee identifier",
    )
    missing_categories: list[str] = Field(
        ...,
        min_length=1,
        max_length=10,
        description="Data categories that were missing or incomplete",
    )
    message: str = Field(
        default="Not enough approved employee data to generate a reliable career coaching plan.",
        max_length=500,
        description="User-friendly explanation of why guidance could not be generated",
    )
    created_at: datetime = Field(
        default_factory=utc_now,
        description="Timestamp of generation (authoritatively generated by application)",
    )

    model_config = ConfigDict(extra="forbid")


# 8. Discriminated Union for API response handling keyed by 'status'
CareerCoachResponse = Annotated[
    Annotated[CareerCoachSuccessResponse, Tag("success")] | Annotated[CareerCoachInsufficientDataResponse, Tag("insufficient_data")],
    Discriminator("status"),
]

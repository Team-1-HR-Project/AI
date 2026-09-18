"""Pydantic schemas for the AI HR Policy Assistant service.

Defines strict request models, source reference models, response models,
and a discriminated union keyed by the "status" field.
"""

import uuid
from datetime import datetime, timezone
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag


def utc_now() -> datetime:
    """Returns the current timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


# 1. Request Schema
class PolicyQuestionRequest(BaseModel):
    """Input request model for asking a policy question."""

    employee_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Target employee identifier (e.g. 'EMP-001')",
    )
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Employee question regarding company HR policies",
    )
    session_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=100,
        description="Optional unique identifier of an existing chat session",
    )

    model_config = ConfigDict(extra="forbid")


# 2. Source Policy Reference Schema
class PolicyReference(BaseModel):
    """Metadata reference to an approved company policy cited in the answer."""

    policy_id: int = Field(
        ...,
        ge=1,
        description="Unique database ID of the referenced company policy",
    )
    policy_code: str = Field(
        ...,
        min_length=2,
        max_length=50,
        description="Unique code of the company policy (e.g., 'POL-LEAVE-001')",
    )
    title: str = Field(
        ...,
        min_length=2,
        max_length=255,
        description="Title of the referenced policy",
    )
    version: str = Field(
        ...,
        min_length=1,
        max_length=20,
        description="Policy version string (e.g., '1.0')",
    )

    model_config = ConfigDict(extra="forbid")


# 3. Success Output Schema
class PolicyAnswerResponse(BaseModel):
    """Successful response containing a grounded answer and policy references."""

    status: Literal["success"] = Field(
        "success",
        description="Status indicator for successful policy answer",
    )
    session_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        min_length=1,
        max_length=100,
        description="Unique identifier of the chat session",
    )
    employee_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Target employee identifier",
    )
    answer: str = Field(
        ...,
        min_length=5,
        max_length=4000,
        description="Clear, direct, and grounded answer to the policy question",
    )
    policy_references: list[PolicyReference] = Field(
        ...,
        min_length=1,
        max_length=10,
        description="List of approved company policies cited in the answer",
    )
    employee_facts_used: list[str] = Field(
        default_factory=list,
        max_length=20,
        description="Permitted employee-specific HR facts referenced in the answer",
    )
    created_at: datetime = Field(
        default_factory=utc_now,
        description="Timestamp of answer generation in UTC",
    )

    model_config = ConfigDict(extra="forbid")


# 4. Fallback / Unsupported Output Schema
class PolicyFallbackResponse(BaseModel):
    """Fallback response when a question cannot be answered from approved policies."""

    status: Literal["unsupported"] = Field(
        "unsupported",
        description="Status indicator for unsupported or out-of-scope question",
    )
    session_id: str | None = Field(
        default=None,
        max_length=100,
        description="Unique identifier of the chat session, if established",
    )
    employee_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Target employee identifier",
    )
    message: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        description="User-facing explanation of why the question cannot be answered",
    )
    created_at: datetime = Field(
        default_factory=utc_now,
        description="Timestamp of response generation in UTC",
    )

    model_config = ConfigDict(extra="forbid")


# 5. Discriminated Union keyed by the "status" field
PolicyAssistantResponse = Annotated[
    Annotated[PolicyAnswerResponse, Tag("success")]
    | Annotated[PolicyFallbackResponse, Tag("unsupported")],
    Discriminator("status"),
]


# 6. Models for LLM generation (omits employee_id and created_at; ignores extra fields)
class PolicyAIModelSuccessOutput(BaseModel):
    status: Literal["success"] = Field("success", description="Status indicator")
    answer: str = Field(..., min_length=5, max_length=4000)
    policy_references: list[PolicyReference] = Field(..., min_length=1, max_length=10)
    employee_facts_used: list[str] = Field(default_factory=list, max_length=20)

    model_config = ConfigDict(extra="ignore")


class PolicyAIModelFallbackOutput(BaseModel):
    status: Literal["unsupported"] = Field("unsupported", description="Status indicator")
    message: str = Field(..., min_length=5, max_length=1000)

    model_config = ConfigDict(extra="ignore")


PolicyAIModelOutput = Annotated[
    Annotated[PolicyAIModelSuccessOutput, Tag("success")]
    | Annotated[PolicyAIModelFallbackOutput, Tag("unsupported")],
    Discriminator("status"),
]


"""Pydantic schemas for health, liveness, and readiness probes."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class LivenessResponse(BaseModel):
    """Response model for process liveness probe."""

    model_config = ConfigDict(extra="forbid")

    status: str = "alive"


class ReadinessResponse(BaseModel):
    """Response model for dependency readiness probe."""

    model_config = ConfigDict(extra="forbid")

    status: str
    reason: str | None = None


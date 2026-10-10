"""Health, liveness, and readiness probe endpoints."""

from __future__ import annotations

import logging
from collections.abc import Generator
from typing import Annotated

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.groq_provider import validate_groq_environment
from app.db.session import get_db
from app.schemas.health import LivenessResponse, ReadinessResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Health"])


def check_ai_provider_config() -> tuple[bool, str | None]:
    """Validates required AI provider configuration without exposing secrets or keys."""
    return validate_groq_environment()


def get_health_db(request: Request) -> Generator[Session | None, None, None]:
    """Safely retrieves a database session for health checks without crashing on dependency failure."""
    try:
        get_db_fn = request.app.dependency_overrides.get(get_db, get_db)
        gen = get_db_fn()
        if hasattr(gen, "__next__"):
            db = next(gen)
            try:
                yield db
            finally:
                try:
                    next(gen)
                except StopIteration:
                    pass
        else:
            yield gen
    except (SQLAlchemyError, OSError, RuntimeError) as exc:
        logger.warning("Readiness probe: failed to acquire DB session: %s", exc)
        yield None


@router.get("/health/live", response_model=LivenessResponse)
@router.get("/live", response_model=LivenessResponse, include_in_schema=False)
def liveness_probe() -> LivenessResponse:
    """Process liveness probe indicating the application process is running and alive.

    Does NOT require database connectivity or AI provider configuration.
    """
    return LivenessResponse(status="alive")


@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    responses={
        200: {"model": ReadinessResponse, "description": "Service is ready to accept traffic"},
        503: {"model": ReadinessResponse, "description": "Service dependencies are not ready"},
    },
)
@router.get("/ready", response_model=ReadinessResponse, include_in_schema=False)
def readiness_probe(
    request: Request,
    db: Annotated[Session | None, Depends(get_health_db)],
) -> JSONResponse:
    """Dependency readiness probe for AI service.

    Verifies:
    1. Critical database schema initialization did not fail at application startup.
    2. Active database connectivity via SELECT 1.
    3. Required AI provider configuration (GROQ_API_KEY, GROQ_MODEL).

    Returns:
    - HTTP 200: {"status": "ready"}
    - HTTP 503: {"status": "not_ready", "reason": "..."}
    Does NOT expose connection strings, credentials, API keys, or internal stack traces.
    """
    # 1. Startup DB initialization check
    if getattr(request.app.state, "db_initialization_error", None) is not None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "not_ready", "reason": "database_initialization_failed"},
        )

    # 2. Live database connectivity check
    if db is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "not_ready", "reason": "database_unavailable"},
        )

    try:
        db.execute(text("SELECT 1"))
    except (SQLAlchemyError, OSError, RuntimeError) as exc:
        logger.warning("Readiness probe: database check failed: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "not_ready", "reason": "database_unavailable"},
        )

    # 3. AI provider configuration check
    is_provider_ok, provider_reason = check_ai_provider_config()
    if not is_provider_ok:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "not_ready", "reason": provider_reason or "ai_provider_unconfigured"},
        )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"status": "ready", "reason": None},
    )

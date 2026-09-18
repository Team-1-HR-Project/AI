"""API router for Feature #7: Team Insight Summary - Manager."""

import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.team_insight import (
    TeamInsightInsufficientDataResponse,
    TeamInsightRequest,
    TeamInsightResponse,
    utc_now,
)
from app.services.team_insight_ai import (
    TeamInsightAIService,
    TeamInsightAIServiceError,
)
from app.services.team_insight_context import TeamInsightContextBuilder

logger = logging.getLogger(__name__)

router = APIRouter()


def get_team_insight_ai_service() -> TeamInsightAIService:
    """Dependency provider for TeamInsightAIService instance."""
    return TeamInsightAIService()


@router.post(
    "/team-insight",
    response_model=TeamInsightResponse,
    tags=["Team Insight"],
    summary="Generate Manager Team Insight Summary",
)
def generate_team_insight(
    request: TeamInsightRequest,
    db: Annotated[Session, Depends(get_db)] = None,
    ai_service: Annotated[TeamInsightAIService, Depends(get_team_insight_ai_service)] = None,
) -> TeamInsightResponse:
    """Generates an executive-level Team Insight Summary for managers.

    Guarantees:
    - Department-based team identification.
    - Consumes ONLY approved employee and team records.
    - Pre-LLM fail-closed data sufficiency verification.
    - Returns either:
        - TeamInsightSuccessResponse (status="success", team_size, team_findings, drill_down_factors, actions)
        - TeamInsightInsufficientDataResponse (status="insufficient_data", message, missing_categories)
    - Prohibits employee PII exposure (no employee names, IDs, emails).
    - Prohibits flight risk / resignation predictions and automatic employment/disciplinary decisions.
    - Strictly preserves deterministic backend numbers and trend directions.
    """
    # 1. Build and verify deterministic context
    context = TeamInsightContextBuilder.build_context(
        db=db,
        department=request.department,
        period=request.period,
    )

    if not context.get("has_sufficient_data"):
        return TeamInsightInsufficientDataResponse(
            status="insufficient_data",
            department=context.get("department") or request.department,
            period=context.get("period") or request.period,
            missing_categories=context.get("missing_categories", []),
            message=context.get("message") or "Insufficient approved data to evaluate team insight summary.",
            created_at=utc_now(),
        )

    # 2. Synthesize manager-facing insights via AI Service
    try:
        return ai_service.generate_team_insight(
            db=db,
            department=request.department,
            period=request.period,
            context=context,
        )
    except TeamInsightAIServiceError:
        error_id = str(uuid.uuid4())
        logger.exception(
            "Team Insight AI service error [Reference ID: %s]",
            error_id,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"AI service temporarily unavailable. Reference ID: {error_id}",
        ) from None


"""API router for the Performance Insight Generator."""

import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.performance_insight import (
    PerformanceInsightInsufficientDataResponse,
    PerformanceInsightRequest,
    PerformanceInsightResponse,
    utc_now,
)
from app.services.performance_insight_ai import (
    PerformanceInsightAIService,
    PerformanceInsightAIServiceError,
)
from app.services.performance_insight_context import PerformanceInsightContextBuilder

logger = logging.getLogger(__name__)

router = APIRouter()


def get_performance_insight_ai_service() -> PerformanceInsightAIService:
    """Dependency provider for PerformanceInsightAIService."""
    return PerformanceInsightAIService()


@router.post(
    "/performance-insight",
    response_model=PerformanceInsightResponse,
    tags=["Performance Insight"],
    summary="Generate Performance Insights",
)
def generate_performance_insight(
    request: PerformanceInsightRequest,
    db: Annotated[Session, Depends(get_db)],
    ai_service: Annotated[PerformanceInsightAIService, Depends(get_performance_insight_ai_service)],
) -> PerformanceInsightResponse:
    """Generates structured comparative performance insights for an employee.

    Flow:
    1. Validates PerformanceInsightRequest.
    2. Builds sanitized context using PerformanceInsightContextBuilder.
    3. If data is insufficient (or employee not found or insufficient periods),
       returns safe PerformanceInsightInsufficientDataResponse without calling the LLM.
    4. Calls PerformanceInsightAIService using ONLY the sanitized context.
    5. Returns the validated PerformanceInsightResponse.
    """
    # 1. Build sanitized context
    context = PerformanceInsightContextBuilder.build_context(
        db=db,
        employee_id=request.employee_id,
        period=request.period,
    )

    # 2. Check sufficient data
    if not context.get("has_sufficient_data") or not context.get("has_trend_data"):
        return PerformanceInsightInsufficientDataResponse(
            status="insufficient_data",
            employee_id=request.employee_id,
            reason=context.get("reason")
            or "Insufficient approved performance data to generate comparative performance insights.",
            periods_found=context.get("facts", {}).get("periods", []),
            message="Insufficient approved performance data to generate comparative performance insights.",
            created_at=utc_now(),
        )

    # 3. Call AI Service with sanitized context
    try:
        return ai_service.generate_insight_from_context(
            context=context,
            employee_id=request.employee_id,
        )
    except PerformanceInsightAIServiceError:
        error_id = str(uuid.uuid4())
        logger.exception(
            "Performance Insight AI service error [Reference ID: %s]",
            error_id,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"AI service temporarily unavailable. Reference ID: {error_id}",
        ) from None


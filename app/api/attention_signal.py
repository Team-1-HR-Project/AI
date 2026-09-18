"""API router for Feature #6: Employee Attention Signal."""

import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.attention_signal import AttentionSignalRequest, AttentionSignalResponse
from app.services.attention_signal_ai import (
    AttentionSignalAIService,
    AttentionSignalAIServiceError,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def get_attention_signal_ai_service() -> AttentionSignalAIService:
    """Dependency provider for AttentionSignalAIService instance."""
    return AttentionSignalAIService()


@router.post(
    "/attention-signal",
    response_model=AttentionSignalResponse,
    tags=["Attention Signal"],
    summary="Generate Employee Attention Signal",
)
def generate_attention_signal(
    request: AttentionSignalRequest,
    db: Annotated[Session, Depends(get_db)] = None,
    ai_service: Annotated[AttentionSignalAIService, Depends(get_attention_signal_ai_service)] = None,
) -> AttentionSignalResponse:
    """Generates structured, evidence-grounded employee attention signals and recommended human follow-up.

    Guarantees:
    - Analyzes ONLY approved employee records (PerformanceRecord, Goal, TaskOutcome, EvaluationTheme).
    - Strict employee data boundary isolation.
    - Pre-LLM fail-closed data sufficiency verification.
    - Returns either:
        - AttentionSignalSuccessResponse (status="success", attention_level, explanation, indicators, follow_up)
        - AttentionSignalInsufficientDataResponse (status="insufficient_data" with reason)
    - Prohibits flight risk / resignation predictions and automatic employment/disciplinary decisions.
    """
    try:
        return ai_service.generate_attention_signal(
            db=db,
            employee_id=request.employee_id,
            target_period=request.target_period,
        )
    except AttentionSignalAIServiceError:
        error_id = str(uuid.uuid4())
        logger.exception(
            "Attention Signal AI service error [Reference ID: %s]",
            error_id,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"AI service temporarily unavailable. Reference ID: {error_id}",
        ) from None


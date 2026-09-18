"""API router for the Evaluation Draft Assistant."""

import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.evaluation_draft import (
    EvaluationDraftInsufficientDataResponse,
    EvaluationDraftRequest,
    EvaluationDraftResponse,
    utc_now,
)
from app.services.evaluation_draft_ai import EvaluationDraftAIService
from app.services.evaluation_draft_context import EvaluationDraftContextBuilder

logger = logging.getLogger(__name__)

router = APIRouter()


def get_evaluation_draft_ai_service() -> EvaluationDraftAIService:
    """Dependency provider for EvaluationDraftAIService."""
    return EvaluationDraftAIService()


@router.post(
    "/evaluation-draft",
    response_model=EvaluationDraftResponse,
    tags=["Evaluation Draft Assistant"],
    summary="Generate AI Evaluation Draft",
)
def generate_evaluation_draft(
    request: EvaluationDraftRequest,
    db: Annotated[Session, Depends(get_db)],
    ai_service: Annotated[EvaluationDraftAIService, Depends(get_evaluation_draft_ai_service)],
) -> EvaluationDraftResponse:
    """Generates an evidence-grounded performance evaluation draft for human manager review.

    Guarantees:
    - Uses ONLY approved employee records.
    - Zero persistence: Never saves or automatically submits the draft.
    - Short-circuits with an insufficient-data response if evidence is inadequate.
    - Demands human manager review and approval before any future use.
    """
    # 1. Build sanitized approved context
    context = EvaluationDraftContextBuilder.build_context(
        db=db,
        employee_id=request.employee_id,
        period=request.period,
    )

    # 2. Check sufficient data
    if not context.get("has_sufficient_data"):
        return EvaluationDraftInsufficientDataResponse(
            status="insufficient_data",
            employee_id=request.employee_id,
            period=request.period,
            missing_categories=context.get("missing_categories", []),
            message="Not enough approved employee data to generate a reliable evaluation draft.",
            human_review_required=False,
            created_at=utc_now(),
        )

    # 3. Call AI Service with sanitized approved context
    try:
        return ai_service.generate_draft(
            context=context,
            period=request.period,
            entered_scores=request.evaluation_scores,
            manager_notes=request.manager_notes,
        )
    except (RuntimeError, TimeoutError):
        error_id = str(uuid.uuid4())
        logger.exception("Evaluation Draft AI service error [Reference ID: %s]", error_id)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"AI service temporarily unavailable. Reference ID: {error_id}",
        ) from None


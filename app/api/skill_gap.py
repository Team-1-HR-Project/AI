"""API router for AI #4: Skill-Gap & Development Recommendations."""

import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.skill_gap import SkillGapRequest, SkillGapResponse
from app.services.skill_gap_ai import SkillGapAIService, SkillGapAIServiceError

logger = logging.getLogger(__name__)

router = APIRouter()


def get_skill_gap_ai_service() -> SkillGapAIService:
    """Dependency provider for SkillGapAIService instance."""
    return SkillGapAIService()


@router.post(
    "/skill-gap",
    response_model=SkillGapResponse,
    tags=["Skill Gap"],
    summary="Generate Skill-Gap & Development Recommendations",
)
def generate_skill_gap(
    request: SkillGapRequest,
    db: Annotated[Session, Depends(get_db)] = None,
    ai_service: Annotated[SkillGapAIService, Depends(get_skill_gap_ai_service)] = None,
) -> SkillGapResponse:
    """Generates structured, evidence-grounded skill-gap analyses and development recommendations.

    Guarantees:
    - Analyzes ONLY approved records for the requested employee.
    - Strict employee and session isolation.
    - Pre-LLM fail-closed data sufficiency verification.
    - Returns either:
        - SkillGapSuccessResponse (status="success", skill_gaps, recommendations, advisory disclaimer)
        - SkillGapInsufficientDataResponse (status="insufficient_data" with missing categories)
    - Prohibits binding employment, compensation, promotion, or disciplinary decisions.
    """
    try:
        return ai_service.generate_skill_gap_analysis(
            db=db,
            employee_id=request.employee_id,
            period=request.period,
            target_role=request.target_role,
            target_skills=request.target_skills,
        )
    except SkillGapAIServiceError:
        error_id = str(uuid.uuid4())
        logger.exception(
            "Skill Gap AI service error [Reference ID: %s]",
            error_id,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"AI service temporarily unavailable. Reference ID: {error_id}",
        ) from None


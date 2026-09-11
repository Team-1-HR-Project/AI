from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.career_coach import CareerCoachRequest, CareerCoachResponse
from app.services.career_coach_ai import CareerCoachAIService, CareerCoachAIServiceError

router = APIRouter()


# Dependency to provide the AI service instance
def get_career_coach_ai_service() -> CareerCoachAIService:
    return CareerCoachAIService()


@router.post(
    "/career-coach",
    response_model=CareerCoachResponse,
    tags=["Career Coach"],
    summary="Generate AI Career Coach Development Plan",
)
def generate_career_coach(
    request: CareerCoachRequest,
    db: Annotated[Session, Depends(get_db)] = None,
    ai_service: Annotated[CareerCoachAIService, Depends(get_career_coach_ai_service)] = None,
) -> CareerCoachResponse:
    """
    Generate structured, evidence-based career development guidance for an employee.
    Returns either:
    - CareerCoachSuccessResponse (strengths, development areas, action plan, follow-up)
    - CareerCoachInsufficientDataResponse (if required employee data is missing)
    """
    try:
        return ai_service.generate_career_plan(
            db=db,
            employee_id=request.employee_id,
            period=request.period,
        )
    except CareerCoachAIServiceError as err:
        # Cleanly return a controlled HTTP error without leaking secrets or stack traces
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(err),
        ) from None


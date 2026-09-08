from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.services.career_coach_ai import CareerCoachAIService, CareerCoachAIServiceError
from app.schemas.career_coach import CareerCoachResponse

router = APIRouter()

# Dependency to provide the AI service instance
def get_career_coach_ai_service() -> CareerCoachAIService:
    return CareerCoachAIService()

@router.post(
    "/career-coach/{employee_id}",
    response_model=CareerCoachResponse,
    tags=["Career Coach"],
    summary="Generate AI Career Coach Development Plan",
)
def generate_career_coach(
    employee_id: str,
    period: Optional[str] = Query(None, description="Performance/Reporting period (e.g. '2026-Q3')"),
    db: Session = Depends(get_db),
    ai_service: CareerCoachAIService = Depends(get_career_coach_ai_service),
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
            employee_id=employee_id,
            period=period
        )
    except CareerCoachAIServiceError as err:
        # Cleanly return a controlled HTTP error without leaking secrets or stack traces
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(err)
        ) from None

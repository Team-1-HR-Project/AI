import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.security import (
    CallerContext,
    authorize_employee_scope,
    get_caller_context,
)
from app.db.session import get_db
from app.schemas.career_coach import CareerCoachRequest, CareerCoachResponse
from app.services.audit_service import AIAuditOutcome, AIAuditService
from app.services.career_coach_ai import CareerCoachAIService, CareerCoachAIServiceError
from app.services.career_coach_context import CareerCoachContextBuilder
from app.services.snapshot_service import persist_insight_snapshot

logger = logging.getLogger(__name__)

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
    caller: Annotated[CallerContext, Depends(get_caller_context)],
    db: Annotated[Session, Depends(get_db)] = None,
    ai_service: Annotated[CareerCoachAIService, Depends(get_career_coach_ai_service)] = None,
) -> CareerCoachResponse:
    """Generate structured, evidence-based career development guidance for an employee.

    Returns either:
    - CareerCoachSuccessResponse (strengths, development areas, action plan, follow-up)
    - CareerCoachInsufficientDataResponse (if required employee data is missing)
    """
    audit = AIAuditService.create_context(
        db=db,
        feature="career_coach",
        endpoint="/api/career-coach",
        caller=caller,
        ai_service=ai_service,
        scope_employee_id=request.employee_id,
        scope_department=caller.department if caller else None,
    )
    try:
        target_employee_id = authorize_employee_scope(
            caller=caller,
            target_employee_id=request.employee_id,
            db=db,
            is_manager_only=False,
        )
        audit.set_scope(employee_id=target_employee_id)
        context = CareerCoachContextBuilder.build_context(
            db=db, employee_id=target_employee_id, period=request.period
        )
        response = ai_service.generate_career_plan(
            db=db,
            employee_id=target_employee_id,
            period=request.period,
        )
        audit.record_response(response)
        try:
            persist_insight_snapshot(
                db=db,
                feature="career_coach",
                content=response,
                context=context,
                request_payload={"employee_id": target_employee_id, "period": request.period},
                scope_employee_id=target_employee_id,
                scope_department=caller.department if caller else None,
                period=request.period,
                caller=caller,
                ai_service=ai_service,
            )
        except (SQLAlchemyError, OSError, RuntimeError, ValueError) as exc:
            logger.warning("Failed to record career coach snapshot: %s", exc)
        return response
    except HTTPException as exc:
        if exc.status_code == 403:
            audit.record_outcome(AIAuditOutcome.UNAUTHORIZED)
        raise
    except CareerCoachAIServiceError:
        error_id = str(uuid.uuid4())
        logger.exception(
            "Career Coach AI service error [Reference ID: %s]",
            error_id,
        )
        audit.record_outcome(AIAuditOutcome.PROVIDER_ERROR, reference_id=error_id)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"AI service temporarily unavailable. Reference ID: {error_id}",
        ) from None
    finally:
        audit.flush()




"""API router for AI #4: Skill-Gap & Development Recommendations."""

import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.security import (
    CallerContext,
    authorize_employee_scope,
    get_caller_context,
)
from app.db.session import get_db
from app.schemas.skill_gap import SkillGapRequest, SkillGapResponse
from app.services.audit_service import AIAuditOutcome, AIAuditService
from app.services.skill_gap_ai import SkillGapAIService, SkillGapAIServiceError
from app.services.skill_gap_context import SkillGapContextBuilder
from app.services.snapshot_service import persist_insight_snapshot

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
    caller: Annotated[CallerContext, Depends(get_caller_context)],
    db: Annotated[Session, Depends(get_db)] = None,
    ai_service: Annotated[SkillGapAIService, Depends(get_skill_gap_ai_service)] = None,
) -> SkillGapResponse:
    """Generates structured, evidence-grounded skill-gap analyses and development recommendations.

    Guarantees:
    - Enforces caller authorization scope.
    - Analyzes ONLY approved records for the requested employee.
    - Strict employee and session isolation.
    - Pre-LLM fail-closed data sufficiency verification.
    - Returns either:
        - SkillGapSuccessResponse (status="success", skill_gaps, recommendations, advisory disclaimer)
        - SkillGapInsufficientDataResponse (status="insufficient_data" with missing categories)
    - Prohibits binding employment, compensation, promotion, or disciplinary decisions.
    """
    audit = AIAuditService.create_context(
        db=db,
        feature="skill_gap",
        endpoint="/api/skill-gap",
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
        context_result = SkillGapContextBuilder.build_context(
            db=db, employee_id=target_employee_id, period=request.period,
            target_role=request.target_role, target_skills=request.target_skills,
        )
        response = ai_service.generate_skill_gap_analysis(
            db=db,
            employee_id=target_employee_id,
            period=request.period,
            target_role=request.target_role,
            target_skills=request.target_skills,
        )
        audit.record_response(response)
        persist_insight_snapshot(
            db=db, feature="skill_gap", content=response,
            context=context_result,
            request_payload={
                "employee_id": target_employee_id, "period": request.period,
                "target_role": request.target_role, "target_skills": request.target_skills,
            },
            scope_employee_id=target_employee_id, scope_department=caller.department,
            period=request.period, caller=caller, ai_service=ai_service,
        )
        return response
    except HTTPException as exc:
        if exc.status_code == 403:
            audit.record_outcome(AIAuditOutcome.UNAUTHORIZED)
        raise
    except SkillGapAIServiceError:
        error_id = str(uuid.uuid4())
        logger.exception(
            "Skill Gap AI service error [Reference ID: %s]",
            error_id,
        )
        audit.record_outcome(AIAuditOutcome.PROVIDER_ERROR, reference_id=error_id)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"AI service temporarily unavailable. Reference ID: {error_id}",
        ) from None
    finally:
        audit.flush()

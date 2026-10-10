"""API router for Feature #6: Employee Attention Signal."""

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
from app.schemas.attention_signal import AttentionSignalRequest, AttentionSignalResponse
from app.services.attention_signal_ai import (
    AttentionSignalAIService,
    AttentionSignalAIServiceError,
)
from app.services.attention_signal_context import AttentionSignalContextBuilder
from app.services.audit_service import AIAuditOutcome, AIAuditService
from app.services.snapshot_service import persist_insight_snapshot

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
    caller: Annotated[CallerContext, Depends(get_caller_context)],
    db: Annotated[Session, Depends(get_db)] = None,
    ai_service: Annotated[AttentionSignalAIService, Depends(get_attention_signal_ai_service)] = None,
) -> AttentionSignalResponse:
    """Generates structured, evidence-grounded employee attention signals and recommended human follow-up.

    Guarantees:
    - Enforces manager/hr_admin caller authorization scope.
    - Analyzes ONLY approved employee records (PerformanceRecord, Goal, TaskOutcome, EvaluationTheme).
    - Strict employee data boundary isolation.
    - Pre-LLM fail-closed data sufficiency verification.
    - Returns either:
        - AttentionSignalSuccessResponse (status="success", attention_level, explanation, indicators, follow_up)
        - AttentionSignalInsufficientDataResponse (status="insufficient_data" with reason)
    - Prohibits flight risk / resignation predictions and automatic employment/disciplinary decisions.
    """
    audit = AIAuditService.create_context(
        db=db,
        feature="attention_signal",
        endpoint="/api/attention-signal",
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
            is_manager_only=True,
        )
        audit.set_scope(employee_id=target_employee_id)
        context = AttentionSignalContextBuilder.build_context(
            db=db, employee_id=target_employee_id, target_period=request.target_period
        )
        response = ai_service.generate_attention_signal(
            db=db,
            employee_id=target_employee_id,
            target_period=request.target_period,
        )
        audit.record_response(response)
        persist_insight_snapshot(
            db=db, feature="attention_signal", content=response,
            context=context,
            request_payload={"employee_id": target_employee_id, "target_period": request.target_period},
            scope_employee_id=target_employee_id, scope_department=caller.department,
            period=request.target_period, caller=caller, ai_service=ai_service,
        )
        return response
    except HTTPException as exc:
        if exc.status_code == 403:
            audit.record_outcome(AIAuditOutcome.UNAUTHORIZED)
        raise
    except AttentionSignalAIServiceError:
        error_id = str(uuid.uuid4())
        logger.exception(
            "Attention Signal AI service error [Reference ID: %s]",
            error_id,
        )
        audit.record_outcome(AIAuditOutcome.PROVIDER_ERROR, reference_id=error_id)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"AI service temporarily unavailable. Reference ID: {error_id}",
        ) from None
    finally:
        audit.flush()

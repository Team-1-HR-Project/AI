"""API router for the Evaluation Draft Assistant."""

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
from app.schemas.evaluation_draft import (
    EvaluationDraftInsufficientDataResponse,
    EvaluationDraftRequest,
    EvaluationDraftResponse,
    utc_now,
)
from app.services.audit_service import AIAuditOutcome, AIAuditService
from app.services.evaluation_draft_ai import EvaluationDraftAIService
from app.services.evaluation_draft_context import EvaluationDraftContextBuilder
from app.services.snapshot_service import persist_insight_snapshot

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
    caller: Annotated[CallerContext, Depends(get_caller_context)],
    db: Annotated[Session, Depends(get_db)],
    ai_service: Annotated[EvaluationDraftAIService, Depends(get_evaluation_draft_ai_service)],
) -> EvaluationDraftResponse:
    """Generates an evidence-grounded performance evaluation draft for human manager review.

    Guarantees:
    - Enforces manager/hr_admin authorization scope.
    - Uses ONLY approved employee records.
    - Zero persistence: Never saves or automatically submits the draft.
    - Short-circuits with an insufficient-data response if evidence is inadequate.
    - Demands human manager review and approval before any future use.
    """
    audit = AIAuditService.create_context(
        db=db,
        feature="evaluation_draft",
        endpoint="/api/evaluation-draft",
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

        # 1. Build sanitized approved context
        context = EvaluationDraftContextBuilder.build_context(
            db=db,
            employee_id=target_employee_id,
            period=request.period,
        )

        # 2. Check sufficient data
        if not context.get("has_sufficient_data"):
            insufficient_resp = EvaluationDraftInsufficientDataResponse(
                status="insufficient_data",
                employee_id=target_employee_id,
                period=request.period,
                missing_categories=context.get("missing_categories", []),
                message="Not enough approved employee data to generate a reliable evaluation draft.",
                human_review_required=False,
                created_at=utc_now(),
            )
            audit.record_response(insufficient_resp)
            persist_insight_snapshot(
                db=db, feature="evaluation_draft", content=insufficient_resp,
                context=context,
                request_payload={
                    "employee_id": target_employee_id, "period": request.period,
                    "evaluation_scores": request.evaluation_scores,
                    "manager_notes": request.manager_notes,
                },
                scope_employee_id=target_employee_id, scope_department=caller.department,
                period=request.period, caller=caller, ai_service=ai_service,
            )
            return insufficient_resp

        # 3. Call AI Service with sanitized approved context
        response = ai_service.generate_draft(
            context=context,
            period=request.period,
            entered_scores=request.evaluation_scores,
            manager_notes=request.manager_notes,
        )
        audit.record_response(response)
        persist_insight_snapshot(
            db=db, feature="evaluation_draft", content=response,
            context=context,
            request_payload={
                "employee_id": target_employee_id, "period": request.period,
                "evaluation_scores": request.evaluation_scores,
                "manager_notes": request.manager_notes,
            },
            scope_employee_id=target_employee_id, scope_department=caller.department,
            period=request.period, caller=caller, ai_service=ai_service,
        )
        return response
    except HTTPException as exc:
        if exc.status_code == 403:
            audit.record_outcome(AIAuditOutcome.UNAUTHORIZED)
        raise
    except SQLAlchemyError:
        error_id = str(uuid.uuid4())
        logger.exception("Evaluation Draft database error [Reference ID: %s]", error_id)
        audit.record_outcome(AIAuditOutcome.DEPENDENCY_ERROR, reference_id=error_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"HR data service temporarily unavailable. Reference ID: {error_id}",
        ) from None
    except (RuntimeError, TimeoutError):
        error_id = str(uuid.uuid4())
        logger.exception("Evaluation Draft AI service error [Reference ID: %s]", error_id)
        audit.record_outcome(AIAuditOutcome.PROVIDER_ERROR, reference_id=error_id)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"AI service temporarily unavailable. Reference ID: {error_id}",
        ) from None
    finally:
        audit.flush()

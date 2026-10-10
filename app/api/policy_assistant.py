"""API router for the AI HR Policy Assistant."""

import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.security import (
    CallerContext,
    authorize_chat_session_scope,
    authorize_employee_scope,
    get_caller_context,
)
from app.db.session import get_db
from app.schemas.policy_assistant import (
    PolicyAssistantResponse,
    PolicyQuestionRequest,
)
from app.services.audit_service import AIAuditOutcome, AIAuditService
from app.services.policy_ai import (
    ChatSessionAccessDeniedError,
    ChatSessionNotFoundError,
    PolicyAIService,
    PolicyAIServiceError,
    PolicyDependencyError,
)
from app.services.policy_context import PolicyContextBuilder
from app.services.snapshot_service import persist_insight_snapshot

logger = logging.getLogger(__name__)

router = APIRouter()


def get_policy_ai_service() -> PolicyAIService:
    """Dependency provider for PolicyAIService."""
    return PolicyAIService()


@router.post(
    "/policy-assistant",
    response_model=PolicyAssistantResponse,
    tags=["Policy Assistant"],
    summary="Answer Employee HR Policy Inquiry",
)
def ask_policy_assistant(
    request: PolicyQuestionRequest,
    caller: Annotated[CallerContext, Depends(get_caller_context)],
    db: Annotated[Session, Depends(get_db)],
    ai_service: Annotated[PolicyAIService, Depends(get_policy_ai_service)],
) -> PolicyAssistantResponse:
    """Answers an employee question regarding approved company policies.

    The AI service automatically determines the policy category from the question
    before retrieving approved company policies.

    Returns:
    - PolicyAnswerResponse (status="success"): When answer is grounded in approved policies.
    - PolicyFallbackResponse (status="unsupported"): When question is unsupported or out of scope.
    """
    audit = AIAuditService.create_context(
        db=db,
        feature="policy_assistant",
        endpoint="/api/policy-assistant",
        caller=caller,
        ai_service=ai_service,
        scope_employee_id=request.employee_id,
        scope_department=caller.department if caller else None,
        scope_session_id=request.session_id,
    )
    try:
        target_employee_id = authorize_employee_scope(
            caller=caller,
            target_employee_id=request.employee_id,
            db=db,
            is_manager_only=False,
        )
        audit.set_scope(employee_id=target_employee_id)
        authorize_chat_session_scope(
            caller=caller,
            session_id=request.session_id,
            target_employee_id=target_employee_id,
            db=db,
        )
        context = PolicyContextBuilder.build_context(
            db=db, employee_id=target_employee_id, question=request.question
        )
        response = ai_service.answer_policy_question(
            db=db,
            employee_id=target_employee_id,
            question=request.question,
            session_id=request.session_id,
        )
        audit.record_response(response)
        persist_insight_snapshot(
            db=db, feature="policy_assistant", content=response,
            context=context,
            request_payload={
                "employee_id": target_employee_id,
                "question": request.question,
                "session_id": request.session_id,
            },
            scope_employee_id=target_employee_id, scope_department=caller.department,
            period=None, caller=caller, ai_service=ai_service,
        )
        return response
    except HTTPException as exc:
        if exc.status_code == 403:
            audit.record_outcome(AIAuditOutcome.UNAUTHORIZED)
        raise
    except ChatSessionNotFoundError as exc:
        logger.warning("Policy Assistant session not found: %s", exc)
        audit.record_outcome("not_found")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat session not found.",
        ) from None
    except ChatSessionAccessDeniedError as exc:
        logger.warning("Policy Assistant cross-employee access denied: %s", exc)
        audit.record_outcome(AIAuditOutcome.UNAUTHORIZED)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: session belongs to another employee.",
        ) from None
    except PolicyDependencyError:
        error_id = str(uuid.uuid4())
        logger.exception("Policy Assistant dependency error [Reference ID: %s]", error_id)
        audit.record_outcome(AIAuditOutcome.DEPENDENCY_ERROR, reference_id=error_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Policy service temporarily unavailable. Reference ID: {error_id}",
        ) from None
    except SQLAlchemyError:
        error_id = str(uuid.uuid4())
        logger.exception("Policy Assistant database error [Reference ID: %s]", error_id)
        audit.record_outcome(AIAuditOutcome.DEPENDENCY_ERROR, reference_id=error_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Policy service temporarily unavailable. Reference ID: {error_id}",
        ) from None
    except PolicyAIServiceError:
        error_id = str(uuid.uuid4())
        logger.exception(
            "Policy Assistant AI service error [Reference ID: %s]",
            error_id,
        )
        audit.record_outcome(AIAuditOutcome.PROVIDER_ERROR, reference_id=error_id)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"AI service temporarily unavailable. Reference ID: {error_id}",
        ) from None
    finally:
        audit.flush()

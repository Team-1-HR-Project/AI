"""API router for the AI HR Policy Assistant."""

import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.policy_assistant import (
    PolicyAssistantResponse,
    PolicyQuestionRequest,
)
from app.services.policy_ai import (
    ChatSessionAccessDeniedError,
    ChatSessionNotFoundError,
    PolicyAIService,
    PolicyAIServiceError,
)

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
    try:
        return ai_service.answer_policy_question(
            db=db,
            employee_id=request.employee_id,
            question=request.question,
            session_id=request.session_id,
        )
    except ChatSessionNotFoundError as exc:
        logger.warning("Policy Assistant session not found: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat session not found.",
        ) from None
    except ChatSessionAccessDeniedError as exc:
        logger.warning("Policy Assistant cross-employee access denied: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: session belongs to another employee.",
        ) from None
    except PolicyAIServiceError:
        error_id = str(uuid.uuid4())
        logger.exception(
            "Policy Assistant AI service error [Reference ID: %s]",
            error_id,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"AI service temporarily unavailable. Reference ID: {error_id}",
        ) from None


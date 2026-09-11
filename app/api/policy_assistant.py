"""API router for the AI HR Policy Assistant."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.policy_assistant import (
    PolicyAssistantResponse,
    PolicyQuestionRequest,
)
from app.services.policy_ai import PolicyAIService, PolicyAIServiceError

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
        )
    except PolicyAIServiceError as err:
        # Return HTTP 502 without exposing stack traces, raw provider errors, or secrets
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(err),
        ) from None

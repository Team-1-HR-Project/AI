"""API router for Feature #7: Team Insight Summary - Manager."""

import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.security import (
    CallerContext,
    authorize_department_scope,
    get_caller_context,
)
from app.db.session import get_db
from app.schemas.team_insight import (
    TeamInsightInsufficientDataResponse,
    TeamInsightRequest,
    TeamInsightResponse,
    utc_now,
)
from app.services.audit_service import AIAuditOutcome, AIAuditService
from app.services.snapshot_service import persist_insight_snapshot
from app.services.team_insight_ai import (
    TeamInsightAIService,
    TeamInsightAIServiceError,
)
from app.services.team_insight_context import TeamInsightContextBuilder

logger = logging.getLogger(__name__)

router = APIRouter()


def get_team_insight_ai_service() -> TeamInsightAIService:
    """Dependency provider for TeamInsightAIService instance."""
    return TeamInsightAIService()


@router.post(
    "/team-insight",
    response_model=TeamInsightResponse,
    tags=["Team Insight"],
    summary="Generate Manager Team Insight Summary",
)
def generate_team_insight(
    request: TeamInsightRequest,
    caller: Annotated[CallerContext, Depends(get_caller_context)],
    db: Annotated[Session, Depends(get_db)] = None,
    ai_service: Annotated[TeamInsightAIService, Depends(get_team_insight_ai_service)] = None,
) -> TeamInsightResponse:
    """Generates an executive-level Team Insight Summary for managers.

    Guarantees:
    - Enforces manager/hr_admin authorization scope and department binding.
    - Department-based team identification.
    - Consumes ONLY approved employee and team records.
    - Pre-LLM fail-closed data sufficiency verification.
    - Returns either:
        - TeamInsightSuccessResponse (status="success", team_size, team_findings, drill_down_factors, actions)
        - TeamInsightInsufficientDataResponse (status="insufficient_data", message, missing_categories)
    - Prohibits employee PII exposure (no employee names, IDs, emails).
    - Prohibits flight risk / resignation predictions and automatic employment/disciplinary decisions.
    - Strictly preserves deterministic backend numbers and trend directions.
    """
    audit = AIAuditService.create_context(
        db=db,
        feature="team_insight",
        endpoint="/api/team-insight",
        caller=caller,
        ai_service=ai_service,
        scope_department=request.department,
    )
    try:
        target_department = authorize_department_scope(
            caller=caller,
            requested_department=request.department,
        )
        audit.set_scope(department=target_department)

        # 1. Build and verify deterministic context
        context = TeamInsightContextBuilder.build_context(
            db=db,
            department=target_department,
            period=request.period,
        )

        if not context.get("has_sufficient_data"):
            insufficient_resp = TeamInsightInsufficientDataResponse(
                status="insufficient_data",
                department=context.get("department") or target_department,
                period=context.get("period") or request.period,
                missing_categories=context.get("missing_categories", []),
                message=context.get("message") or "Insufficient approved data to evaluate team insight summary.",
                created_at=utc_now(),
            )
            audit.record_response(insufficient_resp)
            persist_insight_snapshot(
                db=db, feature="team_insight", content=insufficient_resp,
                context=context,
                request_payload={"department": target_department, "period": request.period},
                scope_employee_id=None, scope_department=target_department,
                period=request.period, caller=caller, ai_service=ai_service,
            )
            return insufficient_resp

        # 2. Synthesize manager-facing insights via AI Service
        response = ai_service.generate_team_insight(
            db=db,
            department=target_department,
            period=request.period,
            context=context,
        )
        audit.record_response(response)
        try:
            persist_insight_snapshot(
                db=db,
                feature="team_insight",
                content=response,
                context=context,
                request_payload={"department": target_department, "period": request.period},
                scope_employee_id=None,
                scope_department=target_department,
                period=request.period,
                caller=caller,
                ai_service=ai_service,
            )
        except (SQLAlchemyError, OSError, RuntimeError, ValueError) as exc:
            logger.warning("Failed to record team insight snapshot: %s", exc)
        return response
    except HTTPException as exc:
        if exc.status_code == 403:
            audit.record_outcome(AIAuditOutcome.UNAUTHORIZED)
        raise
    except TeamInsightAIServiceError:
        error_id = str(uuid.uuid4())
        logger.exception(
            "Team Insight AI service error [Reference ID: %s]",
            error_id,
        )
        audit.record_outcome(AIAuditOutcome.PROVIDER_ERROR, reference_id=error_id)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"AI service temporarily unavailable. Reference ID: {error_id}",
        ) from None
    finally:
        audit.flush()

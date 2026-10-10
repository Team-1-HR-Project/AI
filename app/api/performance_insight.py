"""API router for the Performance Insight Generator."""

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
from app.schemas.performance_insight import (
    PerformanceInsightInsufficientDataResponse,
    PerformanceInsightRequest,
    PerformanceInsightResponse,
    utc_now,
)
from app.services.audit_service import AIAuditOutcome, AIAuditService
from app.services.performance_insight_ai import (
    PerformanceInsightAIService,
    PerformanceInsightAIServiceError,
)
from app.services.performance_insight_context import PerformanceInsightContextBuilder
from app.services.snapshot_service import persist_insight_snapshot

logger = logging.getLogger(__name__)

router = APIRouter()


def get_performance_insight_ai_service() -> PerformanceInsightAIService:
    """Dependency provider for PerformanceInsightAIService."""
    return PerformanceInsightAIService()


@router.post(
    "/performance-insight",
    response_model=PerformanceInsightResponse,
    tags=["Performance Insight"],
    summary="Generate Performance Insights",
)
def generate_performance_insight(
    request: PerformanceInsightRequest,
    caller: Annotated[CallerContext, Depends(get_caller_context)],
    db: Annotated[Session, Depends(get_db)],
    ai_service: Annotated[PerformanceInsightAIService, Depends(get_performance_insight_ai_service)],
) -> PerformanceInsightResponse:
    """Generates structured comparative performance insights for an employee.

    Flow:
    1. Validates PerformanceInsightRequest and enforces caller authorization scope.
    2. Builds sanitized context using PerformanceInsightContextBuilder.
    3. If data is insufficient (or employee not found or insufficient periods),
       returns safe PerformanceInsightInsufficientDataResponse without calling the LLM.
    4. Calls PerformanceInsightAIService using ONLY the sanitized context.
    5. Returns the validated PerformanceInsightResponse.
    """
    audit = AIAuditService.create_context(
        db=db,
        feature="performance_insight",
        endpoint="/api/performance-insight",
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

        # 1. Build sanitized context
        context = PerformanceInsightContextBuilder.build_context(
            db=db,
            employee_id=target_employee_id,
            period=request.period,
        )

        # 2. Check sufficient data
        if not context.get("has_sufficient_data") or not context.get("has_trend_data"):
            insufficient_resp = PerformanceInsightInsufficientDataResponse(
                status="insufficient_data",
                employee_id=target_employee_id,
                reason=context.get("reason")
                or "Insufficient approved performance data to generate comparative performance insights.",
                periods_found=context.get("facts", {}).get("periods", []),
                message="Insufficient approved performance data to generate comparative performance insights.",
                created_at=utc_now(),
            )
            audit.record_response(insufficient_resp)
            persist_insight_snapshot(
                db=db, feature="performance_insight", content=insufficient_resp,
                context=context,
                request_payload={"employee_id": target_employee_id, "period": request.period},
                scope_employee_id=target_employee_id, scope_department=caller.department,
                period=request.period, caller=caller, ai_service=ai_service,
            )
            return insufficient_resp

        # 3. Call AI Service with sanitized context
        response = ai_service.generate_insight_from_context(
            context=context,
            employee_id=target_employee_id,
        )
        audit.record_response(response)
        persist_insight_snapshot(
            db=db, feature="performance_insight", content=response,
            context=context,
            request_payload={"employee_id": target_employee_id, "period": request.period},
            scope_employee_id=target_employee_id, scope_department=caller.department,
            period=request.period, caller=caller, ai_service=ai_service,
        )
        return response
    except HTTPException as exc:
        if exc.status_code == 403:
            audit.record_outcome(AIAuditOutcome.UNAUTHORIZED)
        raise
    except PerformanceInsightAIServiceError:
        error_id = str(uuid.uuid4())
        logger.exception(
            "Performance Insight AI service error [Reference ID: %s]",
            error_id,
        )
        audit.record_outcome(AIAuditOutcome.PROVIDER_ERROR, reference_id=error_id)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"AI service temporarily unavailable. Reference ID: {error_id}",
        ) from None
    except SQLAlchemyError:
        error_id = str(uuid.uuid4())
        logger.exception("Performance Insight database error [Reference ID: %s]", error_id)
        audit.record_outcome(AIAuditOutcome.DEPENDENCY_ERROR, reference_id=error_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"HR data service temporarily unavailable. Reference ID: {error_id}",
        ) from None
    finally:
        audit.flush()

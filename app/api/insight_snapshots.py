"""API router for AI insight snapshots, version history, regeneration, and feedback capture."""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from app.api.attention_signal import get_attention_signal_ai_service
from app.api.career_coach import get_career_coach_ai_service
from app.api.evaluation_draft import get_evaluation_draft_ai_service
from app.api.performance_insight import get_performance_insight_ai_service
from app.api.policy_assistant import get_policy_ai_service
from app.api.skill_gap import get_skill_gap_ai_service
from app.api.team_insight import get_team_insight_ai_service
from app.core.security import (
    CallerContext,
    authorize_chat_session_scope,
    authorize_employee_scope,
    get_caller_context,
)
from app.db.session import get_db
from app.models import AIFeedback
from app.schemas.insight_snapshot import (
    AIFeedbackCreateRequest,
    AIFeedbackResponse,
    AIInsightFeature,
    AIInsightHistoryResponse,
    AIInsightSnapshotResponse,
)
from app.services.attention_signal_ai import AttentionSignalAIService
from app.services.attention_signal_context import AttentionSignalContextBuilder
from app.services.career_coach_ai import CareerCoachAIService
from app.services.career_coach_context import CareerCoachContextBuilder
from app.services.evaluation_draft_ai import EvaluationDraftAIService
from app.services.evaluation_draft_context import EvaluationDraftContextBuilder
from app.services.performance_insight_ai import PerformanceInsightAIService
from app.services.performance_insight_context import PerformanceInsightContextBuilder
from app.services.policy_ai import PolicyAIService
from app.services.policy_context import PolicyContextBuilder
from app.services.skill_gap_ai import SkillGapAIService
from app.services.skill_gap_context import SkillGapContextBuilder
from app.services.snapshot_service import (
    deserialize_content,
    get_insight_history,
    get_snapshot_by_id,
    record_ai_feedback,
    regenerate_insight_snapshot,
)
from app.services.team_insight_ai import TeamInsightAIService
from app.services.team_insight_context import TeamInsightContextBuilder

logger = logging.getLogger(__name__)

router = APIRouter(tags=["AI Insights & History"])


def _to_snapshot_response(snapshot: Any) -> AIInsightSnapshotResponse:
    feedback_count = len(snapshot.feedbacks) if hasattr(snapshot, "feedbacks") and snapshot.feedbacks else 0
    return AIInsightSnapshotResponse(
        id=snapshot.id,
        generation_id=snapshot.generation_id,
        version=snapshot.version,
        feature=snapshot.feature,
        source_version=snapshot.source_version,
        source_hash=snapshot.source_hash,
        context_hash=snapshot.context_hash,
        prompt_hash=snapshot.prompt_hash,
        provider=snapshot.provider,
        model=snapshot.model,
        scope_employee_id=snapshot.scope_employee_id,
        scope_department=snapshot.scope_department,
        period=snapshot.period,
        content=deserialize_content(snapshot.content),
        request_payload=deserialize_content(snapshot.request_payload) if snapshot.request_payload else None,
        actor_employee_id=snapshot.actor_employee_id,
        actor_role=snapshot.actor_role,
        previous_snapshot_id=snapshot.previous_snapshot_id,
        regenerated_at=snapshot.regenerated_at,
        regeneration_reason=snapshot.regeneration_reason,
        source_changed=snapshot.source_changed,
        created_at=snapshot.created_at,
        feedback_count=feedback_count,
    )


class RegenerateInsightPayload(BaseModel):
    """Regeneration controls; generated content is never accepted from callers."""

    regeneration_reason: str | None = Field(default=None, max_length=500)

    model_config = ConfigDict(extra="forbid")


def _regenerate_feature(
    *,
    previous: Any,
    caller: CallerContext,
    db: Session,
    services: dict[str, Any],
) -> tuple[Any, Any, Any, dict[str, Any], str | None]:
    """Rebuild the current authorized context and invoke the feature's real AI service."""
    payload = deserialize_content(previous.request_payload) if previous.request_payload else {}
    if not isinstance(payload, dict):
        raise TypeError("Snapshot request metadata is unavailable for regeneration.")
    employee_id = previous.scope_employee_id
    period = payload.get("period") or previous.period
    feature = previous.feature

    if feature == "career_coach":
        context = CareerCoachContextBuilder.build_context(db=db, employee_id=employee_id, period=period)
        response = services[feature].generate_career_plan(db=db, employee_id=employee_id, period=period)
        return response, context, services[feature], payload, period
    if feature == "performance_insight":
        context = PerformanceInsightContextBuilder.build_context(db=db, employee_id=employee_id, period=period)
        response = services[feature].generate_insight_from_context(context=context, employee_id=employee_id)
        return response, context, services[feature], payload, period
    if feature == "policy_assistant":
        question = payload.get("question", "")
        authorize_employee_scope(
            caller=caller, target_employee_id=employee_id, db=db, is_manager_only=False
        )
        authorize_chat_session_scope(
            caller=caller,
            session_id=payload.get("session_id"),
            target_employee_id=employee_id,
            db=db,
        )
        context = PolicyContextBuilder.build_context(db=db, employee_id=employee_id, question=question)
        response = services[feature].answer_policy_question(
            db=db, employee_id=employee_id, question=question, session_id=payload.get("session_id")
        )
        return response, context, services[feature], payload, None
    if feature == "evaluation_draft":
        authorize_employee_scope(
            caller=caller, target_employee_id=employee_id, db=db, is_manager_only=True
        )
        context = EvaluationDraftContextBuilder.build_context(db=db, employee_id=employee_id, period=period)
        response = services[feature].generate_draft(
            context=context,
            period=period,
            entered_scores=payload.get("evaluation_scores"),
            manager_notes=payload.get("manager_notes"),
        )
        return response, context, services[feature], payload, period
    if feature == "skill_gap":
        target_role = payload.get("target_role")
        target_skills = payload.get("target_skills")
        context_result = SkillGapContextBuilder.build_context(
            db=db, employee_id=employee_id, period=period,
            target_role=target_role, target_skills=target_skills,
        )
        context = context_result.get("context", context_result)
        response = services[feature].generate_skill_gap_analysis(
            db=db, employee_id=employee_id, period=period,
            target_role=target_role, target_skills=target_skills,
        )
        return response, context, services[feature], payload, period
    if feature == "attention_signal":
        target_period = payload.get("target_period") or period
        context = AttentionSignalContextBuilder.build_context(
            db=db, employee_id=employee_id, target_period=target_period
        )
        response = services[feature].generate_attention_signal(
            db=db, employee_id=employee_id, target_period=target_period
        )
        return response, context, services[feature], payload, target_period
    if feature == "team_insight":
        department = previous.scope_department or payload.get("department")
        context = TeamInsightContextBuilder.build_context(db=db, department=department, period=period)
        response = services[feature].generate_team_insight(
            db=db, department=department, period=period, context=context
        )
        return response, context, services[feature], payload, period
    raise ValueError(f"Unsupported snapshot feature '{feature}'.")


@router.get(
    "/history",
    response_model=AIInsightHistoryResponse,
    status_code=status.HTTP_200_OK,
)
def get_history(
    caller: Annotated[CallerContext, Depends(get_caller_context)],
    feature: AIInsightFeature = Query(..., description="AI feature name"),  # noqa: B008
    employee_id: str | None = Query(default=None, description="Employee scope filter"),
    department: str | None = Query(default=None, description="Department scope filter"),
    period: str | None = Query(default=None, description="Evaluation period filter"),
    page: int = Query(default=1, ge=1, description="1-based result page"),
    page_size: int = Query(default=50, ge=1, le=100, description="Maximum snapshots per page"),
    db: Annotated[Session, Depends(get_db)] = None,
) -> AIInsightHistoryResponse:
    """Retrieves version history for previously generated AI insights with scope authorization."""
    snapshots = get_insight_history(
        db=db,
        caller=caller,
        feature=feature.value,
        employee_id=employee_id,
        department=department,
        period=period,
        page=page,
        page_size=page_size,
    )
    has_more = len(snapshots) > page_size
    serialized = [_to_snapshot_response(s) for s in snapshots[:page_size]]
    return AIInsightHistoryResponse(
        feature=feature,
        scope_employee_id=employee_id or (caller.employee_id if caller.role == "employee" else None),
        scope_department=department or (caller.department if caller.role == "manager" else None),
        period=period,
        total_versions=len(serialized),
        page=page,
        page_size=page_size,
        has_more=has_more,
        snapshots=serialized,
    )


@router.get(
    "/snapshots/{snapshot_id}",
    response_model=AIInsightSnapshotResponse,
    status_code=status.HTTP_200_OK,
)
def get_snapshot(
    snapshot_id: str,
    caller: Annotated[CallerContext, Depends(get_caller_context)],
    db: Annotated[Session, Depends(get_db)] = None,
) -> AIInsightSnapshotResponse:
    """Retrieves a single AI insight snapshot by ID with scope authorization."""
    snapshot = get_snapshot_by_id(db=db, caller=caller, snapshot_id=snapshot_id)
    return _to_snapshot_response(snapshot)


@router.post(
    "/snapshots/{snapshot_id}/regenerate",
    response_model=AIInsightSnapshotResponse,
    status_code=status.HTTP_201_CREATED,
)
def regenerate_snapshot(
    snapshot_id: str,
    payload: RegenerateInsightPayload,
    caller: Annotated[CallerContext, Depends(get_caller_context)],
    db: Annotated[Session, Depends(get_db)] = None,
    career_service: Annotated[CareerCoachAIService, Depends(get_career_coach_ai_service)] = None,
    performance_service: Annotated[PerformanceInsightAIService, Depends(get_performance_insight_ai_service)] = None,
    policy_service: Annotated[PolicyAIService, Depends(get_policy_ai_service)] = None,
    evaluation_service: Annotated[EvaluationDraftAIService, Depends(get_evaluation_draft_ai_service)] = None,
    skill_service: Annotated[SkillGapAIService, Depends(get_skill_gap_ai_service)] = None,
    attention_service: Annotated[AttentionSignalAIService, Depends(get_attention_signal_ai_service)] = None,
    team_service: Annotated[TeamInsightAIService, Depends(get_team_insight_ai_service)] = None,
) -> AIInsightSnapshotResponse:
    """Creates a new version of the specified AI insight snapshot without overwriting previous versions."""
    new_snapshot = regenerate_insight_snapshot(
        db=db,
        caller=caller,
        previous_snapshot_id=snapshot_id,
        generator=lambda previous: _regenerate_feature(
            previous=previous,
            caller=caller,
            db=db,
            services={
                "career_coach": career_service,
                "performance_insight": performance_service,
                "policy_assistant": policy_service,
                "evaluation_draft": evaluation_service,
                "skill_gap": skill_service,
                "attention_signal": attention_service,
                "team_insight": team_service,
            },
        ),
        regeneration_reason=payload.regeneration_reason,
    )
    return _to_snapshot_response(new_snapshot)


@router.post(
    "/snapshots/{snapshot_id}/feedback",
    response_model=AIFeedbackResponse,
    status_code=status.HTTP_201_CREATED,
)
def submit_feedback(
    snapshot_id: str,
    request: AIFeedbackCreateRequest,
    caller: Annotated[CallerContext, Depends(get_caller_context)],
    db: Annotated[Session, Depends(get_db)] = None,
) -> AIFeedbackResponse:
    """Records helpful/not-helpful feedback and optional commentary linked to an AI insight snapshot."""
    feedback = record_ai_feedback(
        db=db,
        caller=caller,
        snapshot_id=snapshot_id,
        is_helpful=request.is_helpful,
        feedback_text=request.feedback_text,
    )
    return AIFeedbackResponse.model_validate(feedback)


@router.get(
    "/snapshots/{snapshot_id}/feedback",
    response_model=list[AIFeedbackResponse],
    status_code=status.HTTP_200_OK,
)
def get_feedbacks_for_snapshot(
    snapshot_id: str,
    caller: Annotated[CallerContext, Depends(get_caller_context)],
    page: int = Query(default=1, ge=1, description="1-based result page"),
    page_size: int = Query(default=50, ge=1, le=100, description="Maximum feedback entries per page"),
    db: Annotated[Session, Depends(get_db)] = None,
) -> list[AIFeedbackResponse]:
    """Lists feedback entries for a specific AI insight snapshot."""
    snapshot = get_snapshot_by_id(db=db, caller=caller, snapshot_id=snapshot_id)
    feedbacks = (
        db.query(AIFeedback)
        .filter(AIFeedback.snapshot_id == snapshot.id)
        .order_by(AIFeedback.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    return [AIFeedbackResponse.model_validate(f) for f in feedbacks]

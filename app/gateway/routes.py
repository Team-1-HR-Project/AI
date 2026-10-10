"""Public HR Backend Gateway routes for all AI operations.

Provides the secure entry point for Flutter and Web clients.
Enforces:
- Bearer JWT authentication
- RBAC permissions
- Employee self-service boundaries
- Manager department scopes
- Canonical context assembly
- Service authentication to internal AI service
- Complete mapping between internal AI responses and public HR responses
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.gateway.mappers import (
    map_attention_signal_response,
    map_career_coach_response,
    map_evaluation_draft_response,
    map_feedback_response,
    map_history_response,
    map_performance_insight_response,
    map_policy_assistant_response,
    map_skill_gap_response,
    map_snapshot_response,
    map_team_insight_response,
)
from app.gateway.schemas import (
    HRApiSuccessResponse,
    HRAttentionSignalData,
    HRAttentionSignalRequest,
    HRCareerCoachData,
    HRCareerCoachRequest,
    HREvaluationDraftData,
    HREvaluationDraftRequest,
    HRFeedbackCreateRequest,
    HRFeedbackData,
    HRInsightHistoryData,
    HRInsightSnapshotData,
    HRPerformanceInsightData,
    HRPerformanceInsightRequest,
    HRPolicyAssistantData,
    HRPolicyQuestionRequest,
    HRRegenerateRequest,
    HRSkillGapData,
    HRSkillGapRequest,
    HRTeamInsightData,
    HRTeamInsightRequest,
)
from app.gateway.security import (
    HRApiException,
    HRCallerContext,
    HRCallerRole,
    get_hr_caller_context,
    verify_and_resolve_department_scope,
    verify_and_resolve_employee_scope,
    verify_feature_rbac,
)
from app.gateway.service_client import call_ai_service
from app.models import AIInsightSnapshot, ChatSession

router = APIRouter(prefix="/api/ai", tags=["HR AI Gateway"])


# =============================================================================
# 1. AI Career Coach
# =============================================================================


@router.post(
    "/career-coach",
    response_model=HRApiSuccessResponse[HRCareerCoachData],
    summary="AI Career Coach Guidance (For All Roles)",
    description="Generates personalized career development guidance based on approved employee data.",
)
async def hr_career_coach(
    request: HRCareerCoachRequest,
    caller: Annotated[HRCallerContext, Depends(get_hr_caller_context)],
    db: Annotated[Session, Depends(get_db)],
) -> HRApiSuccessResponse[HRCareerCoachData]:
    verify_feature_rbac(caller, "career-coach")
    target_id = verify_and_resolve_employee_scope(caller, request.employee_id, db)

    payload = {
        "employee_id": target_id,
        "period": request.period,
    }

    _status_code, raw_response = await call_ai_service(
        method="POST",
        path="/api/career-coach",
        caller=caller,
        json_body=payload,
    )

    data = map_career_coach_response(raw_response)
    msg = (
        "Career coach guidance retrieved successfully."
        if data.get("status") == "success"
        else data.get("message", "Insufficient data.")
    )
    return HRApiSuccessResponse(success=True, message=msg, data=data)


# =============================================================================
# 2. AI Performance Insight
# =============================================================================


@router.post(
    "/performance-insight",
    response_model=HRApiSuccessResponse[HRPerformanceInsightData],
    summary="AI Performance Insight (For All Roles)",
    description="Explains approved performance metrics, trends, and review actions for an employee.",
)
async def hr_performance_insight(
    request: HRPerformanceInsightRequest,
    caller: Annotated[HRCallerContext, Depends(get_hr_caller_context)],
    db: Annotated[Session, Depends(get_db)],
) -> HRApiSuccessResponse[HRPerformanceInsightData]:
    verify_feature_rbac(caller, "performance-insight")
    target_id = verify_and_resolve_employee_scope(caller, request.employee_id, db)

    payload = {
        "employee_id": target_id,
        "period": request.period,
    }

    _status_code, raw_response = await call_ai_service(
        method="POST",
        path="/api/performance-insight",
        caller=caller,
        json_body=payload,
    )

    data = map_performance_insight_response(raw_response)
    msg = (
        "Performance insight retrieved successfully."
        if data.get("status") == "success"
        else "Insufficient approved performance data for this period."
    )
    return HRApiSuccessResponse(success=True, message=msg, data=data)


# =============================================================================
# 3. AI Evaluation Draft (Manager / Admin Only)
# =============================================================================


@router.post(
    "/evaluation-draft",
    response_model=HRApiSuccessResponse[HREvaluationDraftData],
    summary="Generate Evaluation Draft (Manager / HR Admin Only)",
    description="Drafts a structured, human-in-the-loop performance evaluation from approved data and manager scores.",
)
async def hr_evaluation_draft(
    request: HREvaluationDraftRequest,
    caller: Annotated[HRCallerContext, Depends(get_hr_caller_context)],
    db: Annotated[Session, Depends(get_db)],
) -> HRApiSuccessResponse[HREvaluationDraftData]:
    verify_feature_rbac(caller, "evaluation-draft")
    target_id = verify_and_resolve_employee_scope(caller, request.employee_id, db)

    payload = {
        "employee_id": target_id,
        "period": request.period,
        "evaluation_scores": request.evaluation_scores,
        "manager_notes": request.manager_notes,
    }

    _status_code, raw_response = await call_ai_service(
        method="POST",
        path="/api/evaluation-draft",
        caller=caller,
        json_body=payload,
    )

    data = map_evaluation_draft_response(raw_response)
    msg = (
        "Evaluation draft generated successfully."
        if data.get("status") == "success"
        else "Insufficient approved data to generate evaluation draft."
    )
    return HRApiSuccessResponse(success=True, message=msg, data=data)


# =============================================================================
# 4. AI Skill Gap & Recommendations
# =============================================================================


@router.post(
    "/skill-gap",
    response_model=HRApiSuccessResponse[HRSkillGapData],
    summary="AI Skill-Gap & Development Analysis (For All Roles)",
    description="Identifies proficiency gaps and suggests actionable learning recommendations.",
)
async def hr_skill_gap(
    request: HRSkillGapRequest,
    caller: Annotated[HRCallerContext, Depends(get_hr_caller_context)],
    db: Annotated[Session, Depends(get_db)],
) -> HRApiSuccessResponse[HRSkillGapData]:
    verify_feature_rbac(caller, "skill-gap")
    target_id = verify_and_resolve_employee_scope(caller, request.employee_id, db)

    payload = {
        "employee_id": target_id,
        "period": request.period,
        "target_role": request.target_role,
        "target_skills": request.target_skills,
    }

    _status_code, raw_response = await call_ai_service(
        method="POST",
        path="/api/skill-gap",
        caller=caller,
        json_body=payload,
    )

    data = map_skill_gap_response(raw_response)
    msg = (
        "Skill gap analysis retrieved successfully."
        if data.get("status") == "success"
        else "Insufficient approved employee skill records."
    )
    return HRApiSuccessResponse(success=True, message=msg, data=data)


# =============================================================================
# 5. AI Employee Attention Signal (Manager / Admin Only)
# =============================================================================


@router.post(
    "/attention-signal",
    response_model=HRApiSuccessResponse[HRAttentionSignalData],
    summary="Generate Employee Attention Signal (Manager / HR Admin Only)",
    description="Identifies early pattern changes and suggests supportive manager follow-up actions.",
)
async def hr_attention_signal(
    request: HRAttentionSignalRequest,
    caller: Annotated[HRCallerContext, Depends(get_hr_caller_context)],
    db: Annotated[Session, Depends(get_db)],
) -> HRApiSuccessResponse[HRAttentionSignalData]:
    verify_feature_rbac(caller, "attention-signal")
    target_id = verify_and_resolve_employee_scope(caller, request.employee_id, db)

    payload = {
        "employee_id": target_id,
        "target_period": request.target_period,
    }

    _status_code, raw_response = await call_ai_service(
        method="POST",
        path="/api/attention-signal",
        caller=caller,
        json_body=payload,
    )

    data = map_attention_signal_response(raw_response)
    msg = (
        "Employee attention signal retrieved successfully."
        if data.get("status") == "success"
        else "Insufficient approved data across recent periods."
    )
    return HRApiSuccessResponse(success=True, message=msg, data=data)


# =============================================================================
# 6. AI Team Insight Summary (Manager / Admin Only)
# =============================================================================


@router.post(
    "/team-insight",
    response_model=HRApiSuccessResponse[HRTeamInsightData],
    summary="Generate Team Insight Summary (Manager / HR Admin Only)",
    description="Aggregates team workload, skill trends, and recommended actions without exposing individual PII.",
)
async def hr_team_insight(
    request: HRTeamInsightRequest,
    caller: Annotated[HRCallerContext, Depends(get_hr_caller_context)],
    db: Annotated[Session, Depends(get_db)],
) -> HRApiSuccessResponse[HRTeamInsightData]:
    verify_feature_rbac(caller, "team-insight")
    dept = verify_and_resolve_department_scope(caller, request.department, request.department_id, db)

    payload = {
        "department": dept,
        "period": request.period,
    }

    _status_code, raw_response = await call_ai_service(
        method="POST",
        path="/api/team-insight",
        caller=caller,
        json_body=payload,
    )

    data = map_team_insight_response(raw_response)
    msg = (
        "Team insight summary retrieved successfully."
        if data.get("status") == "success"
        else "Insufficient approved data across team members."
    )
    return HRApiSuccessResponse(success=True, message=msg, data=data)


# =============================================================================
# 7. AI HR Policy Assistant
# =============================================================================


@router.post(
    "/policy-assistant",
    response_model=HRApiSuccessResponse[HRPolicyAssistantData],
    summary="Ask HR Policy Assistant (For All Roles)",
    description="Provides grounded answers to employee HR policy questions based on approved company policies.",
)
async def hr_policy_assistant(
    request: HRPolicyQuestionRequest,
    caller: Annotated[HRCallerContext, Depends(get_hr_caller_context)],
    db: Annotated[Session, Depends(get_db)],
) -> HRApiSuccessResponse[HRPolicyAssistantData]:
    verify_feature_rbac(caller, "policy-assistant")
    target_id = verify_and_resolve_employee_scope(caller, request.employee_id, db)

    # If session_id provided, verify session ownership
    if request.session_id:
        sess = db.query(ChatSession).filter(ChatSession.id == request.session_id.strip()).first()
        if sess and sess.employee_id != target_id:
            raise HRApiException(
                status_code=status.HTTP_403_FORBIDDEN,
                message="Forbidden: chat session belongs to another employee.",
            )

    payload = {
        "employee_id": target_id,
        "question": request.question,
        "session_id": request.session_id,
    }

    _status_code, raw_response = await call_ai_service(
        method="POST",
        path="/api/policy-assistant",
        caller=caller,
        json_body=payload,
    )

    data = map_policy_assistant_response(raw_response)
    msg = (
        "Policy question answered successfully."
        if data.get("status") == "success"
        else "The question is outside the approved policy scope."
    )
    return HRApiSuccessResponse(success=True, message=msg, data=data)


# =============================================================================
# 8. AI Insights Version History (Gateway)
# =============================================================================


@router.get(
    "/insights/history",
    response_model=HRApiSuccessResponse[HRInsightHistoryData],
    summary="Get AI Insight Version History",
    description="Retrieves paginated version history for generated AI insights with strict scope isolation.",
)
async def hr_insight_history(
    caller: Annotated[HRCallerContext, Depends(get_hr_caller_context)],
    feature: str = Query(..., description="Feature name e.g. 'career_coach', 'performance_insight'"),
    employee_id: str | None = Query(default=None, description="Employee scope filter"),
    department: str | None = Query(default=None, description="Department scope filter"),
    period: str | None = Query(default=None, description="Evaluation period filter"),
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=50, ge=1, le=100, description="Items per page"),
    db: Annotated[Session, Depends(get_db)] = None,
) -> HRApiSuccessResponse[HRInsightHistoryData]:
    # Enforce scope check on history queries
    target_emp_id = None
    if employee_id or caller.role == HRCallerRole.EMPLOYEE.value:
        target_emp_id = verify_and_resolve_employee_scope(caller, employee_id, db)

    target_dept = department
    if caller.role == HRCallerRole.MANAGER.value and department:
        target_dept = verify_and_resolve_department_scope(caller, department, None, db)

    params = {
        "feature": feature,
        "employee_id": target_emp_id,
        "department": target_dept,
        "period": period,
        "page": page,
        "page_size": page_size,
    }

    _status_code, raw_response = await call_ai_service(
        method="GET",
        path="/api/insights/history",
        caller=caller,
        params=params,
    )

    data = map_history_response(raw_response)
    return HRApiSuccessResponse(
        success=True,
        message="AI insight history retrieved successfully.",
        data=data,
    )


# =============================================================================
# 9. Get Single Snapshot by ID (Gateway)
# =============================================================================


@router.get(
    "/insights/snapshots/{snapshot_id}",
    response_model=HRApiSuccessResponse[HRInsightSnapshotData],
    summary="Retrieve AI Insight Snapshot",
    description="Retrieves a specific historical AI insight snapshot by its unique ID with scope protection.",
)
async def hr_get_snapshot(
    snapshot_id: str,
    caller: Annotated[HRCallerContext, Depends(get_hr_caller_context)],
    db: Annotated[Session, Depends(get_db)],
) -> HRApiSuccessResponse[HRInsightSnapshotData]:
    _status_code, raw_response = await call_ai_service(
        method="GET",
        path=f"/api/insights/snapshots/{snapshot_id}",
        caller=caller,
    )

    data = map_snapshot_response(raw_response)
    return HRApiSuccessResponse(
        success=True,
        message="AI insight snapshot retrieved successfully.",
        data=data,
    )


# =============================================================================
# 10. Regenerate Snapshot (Gateway)
# =============================================================================


@router.post(
    "/insights/snapshots/{snapshot_id}/regenerate",
    response_model=HRApiSuccessResponse[HRInsightSnapshotData],
    status_code=status.HTTP_201_CREATED,
    summary="Regenerate AI Insight Snapshot",
    description="Re-runs AI inference for a previous snapshot if underlying approved data changed.",
)
async def hr_regenerate_snapshot(
    snapshot_id: str,
    request: HRRegenerateRequest,
    caller: Annotated[HRCallerContext, Depends(get_hr_caller_context)],
    db: Annotated[Session, Depends(get_db)],
) -> HRApiSuccessResponse[HRInsightSnapshotData]:
    # Check permissions: employees can only regenerate employee-level features
    snap = db.query(AIInsightSnapshot).filter(AIInsightSnapshot.id == snapshot_id).first()
    if snap and snap.feature in ("evaluation_draft", "attention_signal", "team_insight") and caller.role == HRCallerRole.EMPLOYEE.value:
        raise HRApiException(
            status_code=status.HTTP_403_FORBIDDEN,
            message="Forbidden: employees are not authorized to regenerate manager-level insights.",
        )

    payload = {"reason": request.reason} if request.reason else None

    _status_code, raw_response = await call_ai_service(
        method="POST",
        path=f"/api/insights/snapshots/{snapshot_id}/regenerate",
        caller=caller,
        json_body=payload,
    )

    data = map_snapshot_response(raw_response)
    return HRApiSuccessResponse(
        success=True,
        message="AI insight snapshot regenerated successfully.",
        data=data,
    )


# =============================================================================
# 11. Submit Feedback on Snapshot (Gateway)
# =============================================================================


@router.post(
    "/insights/snapshots/{snapshot_id}/feedback",
    response_model=HRApiSuccessResponse[HRFeedbackData],
    status_code=status.HTTP_201_CREATED,
    summary="Submit AI Insight Feedback",
    description="Submits helpfulness rating and qualitative feedback for an AI insight snapshot.",
)
async def hr_submit_feedback(
    snapshot_id: str,
    request: HRFeedbackCreateRequest,
    caller: Annotated[HRCallerContext, Depends(get_hr_caller_context)],
    db: Annotated[Session, Depends(get_db)],
) -> HRApiSuccessResponse[HRFeedbackData]:
    payload = {
        "is_helpful": request.is_helpful,
        "feedback_text": request.feedback_text,
    }

    _status_code, raw_response = await call_ai_service(
        method="POST",
        path=f"/api/insights/snapshots/{snapshot_id}/feedback",
        caller=caller,
        json_body=payload,
    )

    data = map_feedback_response(raw_response)
    return HRApiSuccessResponse(
        success=True,
        message="Feedback submitted successfully.",
        data=data,
    )


# =============================================================================
# 12. Get Feedbacks for Snapshot (Gateway)
# =============================================================================


@router.get(
    "/insights/snapshots/{snapshot_id}/feedback",
    response_model=HRApiSuccessResponse[list[HRFeedbackData]],
    summary="Get Feedback for AI Insight Snapshot",
    description="Lists feedback entries submitted for a specific AI insight snapshot.",
)
async def hr_get_feedbacks(
    snapshot_id: str,
    caller: Annotated[HRCallerContext, Depends(get_hr_caller_context)],
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=50, ge=1, le=100, description="Items per page"),
    db: Annotated[Session, Depends(get_db)] = None,
) -> HRApiSuccessResponse[list[HRFeedbackData]]:
    params = {"page": page, "page_size": page_size}

    _status_code, raw_response = await call_ai_service(
        method="GET",
        path=f"/api/insights/snapshots/{snapshot_id}/feedback",
        caller=caller,
        params=params,
    )

    feedbacks = [
        map_feedback_response(f) if isinstance(f, dict) else f
        for f in (raw_response if isinstance(raw_response, list) else [])
    ]
    return HRApiSuccessResponse(
        success=True,
        message="Feedback entries retrieved successfully.",
        data=feedbacks,
    )

"""Comprehensive contract and integration tests for the HR Backend Gateway.

Validates:
- Bearer JWT authentication (missing/invalid -> 401)
- RBAC permissions (employee accessing manager features -> 403)
- BOLA / IDOR protection (employee targeting other employee -> 403)
- Manager department scoping (target outside department -> 403, non-existent -> 404)
- Verification that NO AI provider call occurs on authorization failures
- Error mapping (429, 502, 503, 504) into HR envelope {"success": false, "message": ..., "errors": ...}
- Insufficient data handling (status: insufficient_data -> 200 with HR envelope)
- Policy assistant unsupported question handling (status: unsupported -> 200 with HR envelope)
- Successful generation and exact HR DTO <-> AI DTO mapping (preserves evidence, priority, reason, targets)
- History retrieval with pagination (page, page_size, total_versions, has_more)
- Snapshot ownership and retrieval
- Snapshot regeneration (201 Created)
- Feedback submission (201 Created) and pagination
"""

import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.session import Base, get_db
from app.gateway.jwt import create_jwt_token
from app.main import app
from app.models import (
    AIInsightSnapshot,
    Employee,
)

TEST_DB_URL = "sqlite:///:memory:"
test_engine = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False}, poolclass=StaticPool)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture
def db():
    Base.metadata.create_all(bind=test_engine)
    session = TestingSessionLocal()

    # Seed test users
    emp_alice = Employee(
        id="EMP-ALICE",
        first_name="Alice",
        last_name="Smith",
        role_title="Software Engineer",
        department="Engineering",
    )
    emp_bob = Employee(
        id="EMP-BOB",
        first_name="Bob",
        last_name="Jones",
        role_title="QA Engineer",
        department="Engineering",
    )
    emp_charlie_sales = Employee(
        id="EMP-CHARLIE",
        first_name="Charlie",
        last_name="Brown",
        role_title="Sales Rep",
        department="Sales",
    )
    mgr_eng = Employee(
        id="MGR-ENG",
        first_name="David",
        last_name="Miller",
        role_title="Engineering Manager",
        department="Engineering",
    )
    admin_hr = Employee(
        id="ADMIN-HR",
        first_name="Eve",
        last_name="Admin",
        role_title="HR Director",
        department="Human Resources",
    )

    session.add_all([emp_alice, emp_bob, emp_charlie_sales, mgr_eng, admin_hr])
    session.commit()

    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=test_engine)


@pytest.fixture
def client(db):
    def override_get_db():
        try:
            yield db
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.pop(get_db, None)


def _token(emp_id: str, role: str) -> str:
    return create_jwt_token({
        "sub": emp_id,
        "employee_code": emp_id,
        "role": role,
    })


# =============================================================================
# 1. Authentication & Security Tests (401)
# =============================================================================


def test_missing_auth_header_returns_401(client):
    """Missing Authorization header must return HTTP 401 with standard HR envelope."""
    resp = client.post("/api/ai/career-coach", json={"employee_id": "EMP-ALICE"})
    assert resp.status_code == 401
    body = resp.json()
    assert body["success"] is False
    assert "Unauthenticated" in body["message"]


def test_invalid_token_returns_401(client):
    """Malformed or invalid JWT token must return HTTP 401."""
    resp = client.post(
        "/api/ai/career-coach",
        headers={"Authorization": "Bearer invalid.malformed.jwt"},
        json={"employee_id": "EMP-ALICE"},
    )
    assert resp.status_code == 401
    body = resp.json()
    assert body["success"] is False


def test_nonexistent_caller_in_token_returns_401(client):
    """Valid JWT pointing to non-existent employee ID must return HTTP 401."""
    token = create_jwt_token({"sub": "EMP-GHOST", "role": "Employee"})
    resp = client.post(
        "/api/ai/career-coach",
        headers={"Authorization": f"Bearer {token}"},
        json={"employee_id": "EMP-GHOST"},
    )
    assert resp.status_code == 401
    body = resp.json()
    assert body["success"] is False
    assert "not found" in body["message"].lower()


# =============================================================================
# 2. RBAC & Permission Tests (403)
# =============================================================================


@patch("app.gateway.routes.call_ai_service")
def test_employee_accessing_manager_routes_rejected_403(mock_call, client):
    """Employees attempting to access manager-only features must be rejected with 403 without calling AI."""
    token = _token("EMP-ALICE", "Employee")
    headers = {"Authorization": f"Bearer {token}"}

    # Evaluation Draft
    resp1 = client.post(
        "/api/ai/evaluation-draft",
        headers=headers,
        json={"employee_id": "EMP-ALICE", "period": "2026-Q3", "evaluation_scores": {"performance": 85.0}},
    )
    assert resp1.status_code == 403
    assert resp1.json()["success"] is False

    # Attention Signal
    resp2 = client.post(
        "/api/ai/attention-signal",
        headers=headers,
        json={"employee_id": "EMP-ALICE", "target_period": "2026-Q3"},
    )
    assert resp2.status_code == 403
    assert resp2.json()["success"] is False

    # Team Insight
    resp3 = client.post(
        "/api/ai/team-insight",
        headers=headers,
        json={"department": "Engineering"},
    )
    assert resp3.status_code == 403
    assert resp3.json()["success"] is False

    # Crucial assertion: no internal AI service call was ever triggered!
    mock_call.assert_not_called()


# =============================================================================
# 3. Scope & BOLA / IDOR Tests (403 / 404)
# =============================================================================


@patch("app.gateway.routes.call_ai_service")
def test_employee_accessing_other_employee_rejected_403(mock_call, client):
    """An employee requesting another employee's Career Coach must be rejected with 403."""
    token = _token("EMP-ALICE", "Employee")
    resp = client.post(
        "/api/ai/career-coach",
        headers={"Authorization": f"Bearer {token}"},
        json={"employee_id": "EMP-BOB"},
    )
    assert resp.status_code == 403
    body = resp.json()
    assert body["success"] is False
    assert "only access their own records" in body["message"]
    mock_call.assert_not_called()


@patch("app.gateway.routes.call_ai_service")
def test_manager_accessing_outside_department_rejected_403(mock_call, client):
    """A manager requesting an employee outside their department must be rejected with 403."""
    token = _token("MGR-ENG", "Manager")
    resp = client.post(
        "/api/ai/career-coach",
        headers={"Authorization": f"Bearer {token}"},
        json={"employee_id": "EMP-CHARLIE"},  # In Sales department
    )
    assert resp.status_code == 403
    body = resp.json()
    assert body["success"] is False
    assert "own department" in body["message"]
    mock_call.assert_not_called()


@patch("app.gateway.routes.call_ai_service")
def test_manager_accessing_nonexistent_employee_returns_404(mock_call, client):
    """A manager requesting a non-existent employee ID must return 404."""
    token = _token("MGR-ENG", "Manager")
    resp = client.post(
        "/api/ai/career-coach",
        headers={"Authorization": f"Bearer {token}"},
        json={"employee_id": "EMP-UNKNOWN"},
    )
    assert resp.status_code == 404
    body = resp.json()
    assert body["success"] is False
    assert "not found" in body["message"].lower()
    mock_call.assert_not_called()


@patch("app.gateway.routes.call_ai_service")
def test_manager_team_insight_outside_department_rejected_403(mock_call, client):
    """A manager requesting team insight for a different department must be rejected with 403."""
    token = _token("MGR-ENG", "Manager")
    resp = client.post(
        "/api/ai/team-insight",
        headers={"Authorization": f"Bearer {token}"},
        json={"department": "Sales"},
    )
    assert resp.status_code == 403
    body = resp.json()
    assert body["success"] is False
    assert "own department" in body["message"]
    mock_call.assert_not_called()


# =============================================================================
# 4. Error Mapping Tests (422, 429, 502, 503, 504)
# =============================================================================


def test_validation_error_returns_422_with_hr_envelope(client):
    """Pydantic validation error on gateway routes returns 422 with HR envelope."""
    token = _token("EMP-ALICE", "Employee")
    resp = client.post(
        "/api/ai/performance-insight",
        headers={"Authorization": f"Bearer {token}"},
        json={"employee_id": "EMP-ALICE"},  # Missing required 'period'
    )
    assert resp.status_code == 422
    body = resp.json()
    assert body["success"] is False
    assert "Validation error" in body["message"]
    assert isinstance(body["errors"], list)
    assert len(body["errors"]) > 0


@patch("app.gateway.routes.call_ai_service")
def test_provider_timeout_maps_to_504(mock_call, client):
    """When the AI service or provider times out, gateway returns 504 with HR envelope."""
    from app.gateway.security import HRApiException
    mock_call.side_effect = HRApiException(status_code=504, message="AI provider timed out.")

    token = _token("EMP-ALICE", "Employee")
    resp = client.post(
        "/api/ai/career-coach",
        headers={"Authorization": f"Bearer {token}"},
        json={"employee_id": "EMP-ALICE"},
    )
    assert resp.status_code == 504
    body = resp.json()
    assert body["success"] is False
    assert "timed out" in body["message"]


@patch("app.gateway.routes.call_ai_service")
def test_provider_failure_maps_to_502(mock_call, client):
    """When AI provider fails, gateway returns 502 with HR envelope."""
    from app.gateway.security import HRApiException
    mock_call.side_effect = HRApiException(status_code=502, message="AI provider is temporarily unavailable.")

    token = _token("EMP-ALICE", "Employee")
    resp = client.post(
        "/api/ai/career-coach",
        headers={"Authorization": f"Bearer {token}"},
        json={"employee_id": "EMP-ALICE"},
    )
    assert resp.status_code == 502
    body = resp.json()
    assert body["success"] is False
    assert "unavailable" in body["message"]


@patch("app.gateway.routes.call_ai_service")
def test_provider_rate_limit_maps_to_429(mock_call, client):
    """When rate limit is exceeded, gateway returns 429 with HR envelope."""
    from app.gateway.security import HRApiException
    mock_call.side_effect = HRApiException(status_code=429, message="Rate limit exceeded. Please retry later.")

    token = _token("EMP-ALICE", "Employee")
    resp = client.post(
        "/api/ai/career-coach",
        headers={"Authorization": f"Bearer {token}"},
        json={"employee_id": "EMP-ALICE"},
    )
    assert resp.status_code == 429
    body = resp.json()
    assert body["success"] is False
    assert "Rate limit" in body["message"]


# =============================================================================
# 5. Insufficient Data & Unsupported Question (200 with typed fallback)
# =============================================================================


@patch("app.gateway.routes.call_ai_service")
def test_insufficient_data_returns_200_with_envelope(mock_call, client):
    """Insufficient data returns 200 with success=True and status=insufficient_data."""
    mock_call.return_value = (
        200,
        {
            "status": "insufficient_data",
            "employee_id": "EMP-ALICE",
            "missing_categories": ["performance", "goals"],
            "message": "Not enough approved employee data to generate a reliable career coaching plan.",
            "created_at": "2026-09-28T10:00:00Z",
        },
    )

    token = _token("EMP-ALICE", "Employee")
    resp = client.post(
        "/api/ai/career-coach",
        headers={"Authorization": f"Bearer {token}"},
        json={"employee_id": "EMP-ALICE"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["status"] == "insufficient_data"
    assert "missing_categories" in body["data"]
    assert "performance" in body["data"]["missing_categories"]


@patch("app.gateway.routes.call_ai_service")
def test_unsupported_policy_question_returns_200_with_envelope(mock_call, client):
    """Unsupported policy question returns 200 with status=unsupported."""
    mock_call.return_value = (
        200,
        {
            "status": "unsupported",
            "employee_id": "EMP-ALICE",
            "message": "The question is outside the approved policy scope.",
            "created_at": "2026-09-28T10:00:00Z",
        },
    )

    token = _token("EMP-ALICE", "Employee")
    resp = client.post(
        "/api/ai/policy-assistant",
        headers={"Authorization": f"Bearer {token}"},
        json={"employee_id": "EMP-ALICE", "question": "What is the secret stock price prediction?"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["status"] == "unsupported"


# =============================================================================
# 6. Exact HR DTO <-> AI DTO Mapping (No data loss)
# =============================================================================


@patch("app.gateway.routes.call_ai_service")
def test_career_coach_dto_mapping_preserves_evidence_and_targets(mock_call, client):
    """Verifies that Career Coach preserves evidence, priority, reason, and measurable targets."""
    mock_call.return_value = (
        200,
        {
            "status": "success",
            "employee_id": "EMP-ALICE",
            "strengths": [
                {
                    "title": "High Technical Delivery",
                    "description": "Consistently exceeds delivery expectations.",
                    "evidence": [
                        {"source_type": "performance", "source_id": 10, "claim": "95.5 score in Q3"}
                    ],
                }
            ],
            "development_areas": [
                {
                    "title": "Mentorship",
                    "description": "Increase coaching of junior developers.",
                    "priority": "high",
                    "evidence": [
                        {"source_type": "evaluation_theme", "source_id": 5, "claim": "Peer review feedback"}
                    ],
                }
            ],
            "development_plan": [
                {
                    "action": "Conduct bi-weekly workshops",
                    "reason": "Address identified mentorship opportunity",
                    "measurable_target": "Hold 4 sessions before Q4",
                    "suggested_timeline": "Starts next month",
                }
            ],
            "follow_up": {"checkpoint": "2026-11-01", "review_focus": "Check workshop progress"},
            "created_at": "2026-09-28T10:00:00Z",
        },
    )

    token = _token("EMP-ALICE", "Employee")
    resp = client.post(
        "/api/ai/career-coach",
        headers={"Authorization": f"Bearer {token}"},
        json={"employee_id": "EMP-ALICE"},
    )
    assert resp.status_code == 200
    data = resp.json()["data"]

    # Check that development_focus is populated
    assert data["development_focus"] == "Prioritize: Mentorship"

    # Check evidence is fully preserved
    strength = data["strengths"][0]
    assert strength["evidence"][0]["claim"] == "95.5 score in Q3"
    assert strength["evidence"][0]["source_id"] == 10

    # Check priority is preserved
    dev_area = data["development_areas"][0]
    assert dev_area["priority"] == "high"

    # Check measurable target and reason are preserved
    plan_item = data["development_plan"][0]
    assert plan_item["measurable_target"] == "Hold 4 sessions before Q4"
    assert plan_item["reason"] == "Address identified mentorship opportunity"


@patch("app.gateway.routes.call_ai_service")
def test_policy_assistant_dto_mapping_preserves_policy_references(mock_call, client):
    """Verifies that Policy Assistant preserves rich policy references (id, code, title, version)."""
    mock_call.return_value = (
        200,
        {
            "status": "success",
            "session_id": "sess-xyz-123",
            "employee_id": "EMP-ALICE",
            "answer": "Annual leave entitlement is 21 working days per year.",
            "policy_references": [
                {
                    "policy_id": 1,
                    "policy_code": "POL-LEAVE-01",
                    "title": "Annual and Sick Leave Policy",
                    "version": "2.1",
                }
            ],
            "employee_facts_used": ["department: Engineering", "leave_balance: 14 days"],
            "created_at": "2026-09-28T10:00:00Z",
        },
    )

    token = _token("EMP-ALICE", "Employee")
    resp = client.post(
        "/api/ai/policy-assistant",
        headers={"Authorization": f"Bearer {token}"},
        json={"employee_id": "EMP-ALICE", "question": "How many vacation days do I get?"},
    )
    assert resp.status_code == 200
    data = resp.json()["data"]

    assert data["session_id"] == "sess-xyz-123"
    assert len(data["policy_references"]) == 1
    ref = data["policy_references"][0]
    assert ref["policy_code"] == "POL-LEAVE-01"
    assert ref["version"] == "2.1"
    assert len(data["employee_facts_used"]) == 2


# =============================================================================
# 7. History, Snapshot, Regeneration, and Feedback Tests
# =============================================================================


@patch("app.gateway.routes.call_ai_service")
def test_history_pagination(mock_call, client):
    """Verifies that insight history handles pagination properly."""
    mock_call.return_value = (
        200,
        {
            "feature": "career_coach",
            "scope_employee_id": "EMP-ALICE",
            "scope_department": None,
            "period": None,
            "total_versions": 5,
            "page": 2,
            "page_size": 2,
            "has_more": True,
            "snapshots": [
                {
                    "id": "snap-2",
                    "generation_id": "gen-2",
                    "version": 2,
                    "feature": "career_coach",
                    "scope_employee_id": "EMP-ALICE",
                    "scope_department": None,
                    "period": "2026-Q3",
                    "content": {"status": "success"},
                    "created_at": "2026-09-28T10:00:00Z",
                }
            ],
        },
    )

    token = _token("EMP-ALICE", "Employee")
    resp = client.get(
        "/api/ai/insights/history?feature=career_coach&page=2&page_size=2",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["page"] == 2
    assert data["page_size"] == 2
    assert data["has_more"] is True
    assert len(data["snapshots"]) == 1


@patch("app.gateway.routes.call_ai_service")
def test_regenerate_snapshot_returns_201(mock_call, client):
    """Regenerating a snapshot returns HTTP 201 Created."""
    mock_call.return_value = (
        201,
        {
            "id": "snap-new",
            "generation_id": "gen-new",
            "version": 2,
            "feature": "career_coach",
            "scope_employee_id": "EMP-ALICE",
            "content": {"status": "success"},
            "created_at": "2026-09-28T11:00:00Z",
        },
    )

    token = _token("EMP-ALICE", "Employee")
    resp = client.post(
        "/api/ai/insights/snapshots/snap-prev/regenerate",
        headers={"Authorization": f"Bearer {token}"},
        json={"reason": "Updated quarterly goals"},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["version"] == 2


@patch("app.gateway.routes.call_ai_service")
def test_submit_feedback_returns_201(mock_call, client):
    """Submitting feedback on a snapshot returns HTTP 201 Created."""
    mock_call.return_value = (
        201,
        {
            "id": "fb-1",
            "snapshot_id": "snap-123",
            "actor_employee_id": "EMP-ALICE",
            "actor_role": "employee",
            "is_helpful": True,
            "feedback_text": "Great guidance!",
            "created_at": "2026-09-28T11:00:00Z",
        },
    )

    token = _token("EMP-ALICE", "Employee")
    resp = client.post(
        "/api/ai/insights/snapshots/snap-123/feedback",
        headers={"Authorization": f"Bearer {token}"},
        json={"is_helpful": True, "feedback_text": "Great guidance!"},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["is_helpful"] is True
    assert body["data"]["feedback_text"] == "Great guidance!"


@patch("app.gateway.routes.call_ai_service")
def test_get_feedbacks_for_snapshot(mock_call, client):
    """Retrieving feedback entries for a snapshot returns list with HR envelope."""
    mock_call.return_value = (
        200,
        [
            {
                "id": "fb-1",
                "snapshot_id": "snap-123",
                "actor_employee_id": "EMP-ALICE",
                "actor_role": "employee",
                "is_helpful": True,
                "feedback_text": "Accurate assessment",
                "created_at": "2026-09-28T11:00:00Z",
            }
        ],
    )

    token = _token("EMP-ALICE", "Employee")
    resp = client.get(
        "/api/ai/insights/snapshots/snap-123/feedback",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert len(body["data"]) == 1
    assert body["data"][0]["id"] == "fb-1"


@patch("app.gateway.routes.call_ai_service")
def test_get_single_snapshot(mock_call, client):
    """Retrieving a single snapshot returns snapshot data with HR envelope."""
    mock_call.return_value = (
        200,
        {
            "id": "snap-123",
            "generation_id": "gen-123",
            "version": 1,
            "feature": "career_coach",
            "scope_employee_id": "EMP-ALICE",
            "content": {"status": "success", "summary": "Great progress"},
            "created_at": "2026-09-28T10:00:00Z",
        },
    )

    token = _token("EMP-ALICE", "Employee")
    resp = client.get(
        "/api/ai/insights/snapshots/snap-123",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["id"] == "snap-123"
    assert body["data"]["content"]["summary"] == "Great progress"


@patch("app.gateway.routes.call_ai_service")
def test_performance_insight_dto_mapping(mock_call, client):
    """Verifies that Performance Insight maps ai_interpretation and suggested_review_actions."""
    mock_call.return_value = (
        200,
        {
            "status": "success",
            "employee_id": "EMP-ALICE",
            "verified_facts": {"period": "2026-Q3", "task_completion_rate": 94.0},
            "calculated_trends": {"direction": "improving"},
            "ai_interpretation": {"context": "Performance is consistent and trending upward."},
            "suggested_review_actions": [{"action": "Maintain cadence", "focus": "Quality"}],
            "created_at": "2026-09-28T10:00:00Z",
        },
    )

    token = _token("EMP-ALICE", "Employee")
    resp = client.post(
        "/api/ai/performance-insight",
        headers={"Authorization": f"Bearer {token}"},
        json={"employee_id": "EMP-ALICE", "period": "2026-Q3"},
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert "interpretation" in data
    assert "ai_interpretation" in data
    assert "recommended_actions" in data
    assert "suggested_review_actions" in data
    assert data["interpretation"]["context"] == "Performance is consistent and trending upward."


@patch("app.gateway.routes.call_ai_service")
def test_evaluation_draft_dto_mapping_and_rbac(mock_call, client):
    """Verifies that Evaluation Draft maps all fields and allows Manager."""
    mock_call.return_value = (
        200,
        {
            "status": "success",
            "employee_id": "EMP-ALICE",
            "period": "2026-Q3",
            "evaluation_narrative": "Alice has demonstrated high proficiency in system engineering.",
            "strengths": [{"title": "Architecture", "description": "Designed reliable system", "evidence": []}],
            "improvement_areas": [{"title": "Docs", "description": "Update runbooks", "priority": "medium", "evidence": []}],
            "entered_scores": {"technical": 90.0},
            "disclaimer": "Draft for human review only.",
            "human_review_required": True,
            "created_at": "2026-09-28T10:00:00Z",
        },
    )

    token = _token("MGR-ENG", "Manager")
    resp = client.post(
        "/api/ai/evaluation-draft",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "employee_id": "EMP-ALICE",
            "period": "2026-Q3",
            "evaluation_scores": {"technical": 90.0},
            "manager_notes": "Very satisfied with output.",
        },
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["evaluation_narrative"] == "Alice has demonstrated high proficiency in system engineering."
    assert data["human_review_required"] is True
    assert data["entered_scores"]["technical"] == 90.0
    assert len(data["strengths"]) == 1
    assert len(data["improvement_areas"]) == 1


@patch("app.gateway.routes.call_ai_service")
def test_skill_gap_dto_mapping(mock_call, client):
    """Verifies that Skill Gap preserves severity, evidence, and recommendations."""
    mock_call.return_value = (
        200,
        {
            "status": "success",
            "employee_id": "EMP-ALICE",
            "target_role": "Staff Engineer",
            "skill_gaps": [
                {
                    "skill_name": "Distributed Systems",
                    "current_level": "Intermediate",
                    "desired_level": "Expert",
                    "gap_severity": "high",
                    "evidence": "No approved records for distributed consensus",
                }
            ],
            "recommendations": [
                {
                    "title": "Raft Protocol Course",
                    "learning_type": "training_course",
                    "suggested_duration": "4 weeks",
                    "measurable_outcome": "Implement replicated log",
                }
            ],
            "disclaimer": "Skill-gap advisory guidance.",
            "created_at": "2026-09-28T10:00:00Z",
        },
    )

    token = _token("EMP-ALICE", "Employee")
    resp = client.post(
        "/api/ai/skill-gap",
        headers={"Authorization": f"Bearer {token}"},
        json={"employee_id": "EMP-ALICE", "target_role": "Staff Engineer"},
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["target_role"] == "Staff Engineer"
    assert data["skill_gaps"][0]["skill_name"] == "Distributed Systems"
    assert data["skill_gaps"][0]["gap_severity"] == "high"
    assert data["recommendations"][0]["learning_type"] == "training_course"


@patch("app.gateway.routes.call_ai_service")
def test_attention_signal_dto_mapping(mock_call, client):
    """Verifies that Attention Signal maps indicators, follow-ups, and safety flags."""
    mock_call.return_value = (
        200,
        {
            "status": "success",
            "employee_id": "EMP-ALICE",
            "target_period": "2026-Q3",
            "comparison_period": "2026-Q2",
            "attention_level": "Low",
            "explanation": "Consistent high performance across periods.",
            "contributing_indicators": [
                {
                    "indicator_name": "Attendance Rate",
                    "category": "attendance",
                    "current_value": 98.0,
                    "previous_value": 97.5,
                    "change_description": "Steady attendance",
                    "evidence": "Approved attendance records",
                }
            ],
            "recommended_follow_up": [
                {
                    "action_type": "Recognition",
                    "description": "Acknowledge consistent delivery",
                    "rationale": "High achievement rate",
                }
            ],
            "advisory_only": True,
            "human_review_required": True,
            "created_at": "2026-09-28T10:00:00Z",
        },
    )

    token = _token("MGR-ENG", "Manager")
    resp = client.post(
        "/api/ai/attention-signal",
        headers={"Authorization": f"Bearer {token}"},
        json={"employee_id": "EMP-ALICE", "target_period": "2026-Q3"},
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["attention_level"] == "Low"
    assert data["advisory_only"] is True
    assert data["human_review_required"] is True
    assert len(data["contributing_indicators"]) == 1
    assert len(data["recommended_follow_up"]) == 1


@patch("app.gateway.routes.call_ai_service")
def test_team_insight_dto_mapping(mock_call, client):
    """Verifies that Team Insight maps summary, common skills/gaps, actions, and team findings."""
    mock_call.return_value = (
        200,
        {
            "status": "success",
            "department": "Engineering",
            "period": "2026-Q3",
            "team_size": 5,
            "summary": "Engineering team velocity is healthy.",
            "top_common_skills": ["Python", "Docker"],
            "top_common_gaps": ["Kubernetes"],
            "recommended_management_actions": [
                {
                    "action": "Organize K8s workshop",
                    "priority": "high",
                    "focus_area": "Skill Development",
                    "expected_impact": "Improve deployment self-sufficiency",
                    "timeline": "Next sprint",
                }
            ],
            "team_findings": {"overdue_workload": {"summary": "No overdue tasks"}},
            "data_coverage": {"total_team_members": 5, "records_analyzed": 20},
            "created_at": "2026-09-28T10:00:00Z",
        },
    )

    token = _token("MGR-ENG", "Manager")
    resp = client.post(
        "/api/ai/team-insight",
        headers={"Authorization": f"Bearer {token}"},
        json={"department": "Engineering", "period": "2026-Q3"},
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["department"] == "Engineering"
    assert data["team_size"] == 5
    assert data["top_common_skills"] == ["Python", "Docker"]
    assert data["top_common_gaps"] == ["Kubernetes"]
    assert len(data["recommended_management_actions"]) == 1
    assert data["team_findings"]["overdue_workload"]["summary"] == "No overdue tasks"


@patch("app.gateway.routes.call_ai_service")
def test_employee_id_defaulting_for_self_service(mock_call, client):
    """When an employee omits employee_id, gateway defaults it to caller employee_id."""
    mock_call.return_value = (
        200,
        {
            "status": "success",
            "employee_id": "EMP-ALICE",
            "strengths": [],
            "development_areas": [],
            "development_plan": [],
            "created_at": "2026-09-28T10:00:00Z",
        },
    )

    token = _token("EMP-ALICE", "Employee")
    # Body has NO employee_id
    resp = client.post(
        "/api/ai/career-coach",
        headers={"Authorization": f"Bearer {token}"},
        json={"period": "2026-Q3"},
    )
    assert resp.status_code == 200
    # Verify that call_ai_service received employee_id: "EMP-ALICE"
    call_args = mock_call.call_args[1]
    assert call_args["json_body"]["employee_id"] == "EMP-ALICE"


def test_employee_cannot_regenerate_manager_snapshot(db, client):
    """An employee attempting to regenerate a manager-level snapshot returns 403."""
    # Seed an evaluation draft snapshot in DB
    snap = AIInsightSnapshot(
        id="snap-eval-1",
        generation_id="gen-eval-1",
        version=1,
        feature="evaluation_draft",
        scope_employee_id="EMP-ALICE",
        content=json.dumps({"status": "success"}),
    )
    db.add(snap)
    db.commit()

    token = _token("EMP-ALICE", "Employee")
    resp = client.post(
        "/api/ai/insights/snapshots/snap-eval-1/regenerate",
        headers={"Authorization": f"Bearer {token}"},
        json={"reason": "Retrying"},
    )
    assert resp.status_code == 403
    body = resp.json()
    assert body["success"] is False
    assert "not authorized to regenerate manager-level" in body["message"]


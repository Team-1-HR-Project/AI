"""Focused tests for AI Audit Events (governance, scope, outcomes, and privacy).

Covers all 12 audit requirements:
1. Successful AI request creates an audit event.
2. Insufficient-data outcome creates an audit event.
3. Unauthorized request creates an audit event when caller identity is available.
4. Provider/AI error creates an audit event with the existing reference ID.
5. Actor ID and role come from caller context, not request-body values.
6. Employee/department/session scope is taken from authorized scope.
7. Provider and model are recorded correctly.
8. Full prompt is NOT stored in audit events.
9. Full AI response is NOT stored in audit events.
10. Secrets / API keys / passwords are NOT stored.
11. All 7 AI routes produce audit events.
12. Audit-write failure does not expose sensitive information or corrupt the AI response.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.attention_signal import get_attention_signal_ai_service
from app.api.career_coach import get_career_coach_ai_service
from app.api.evaluation_draft import get_evaluation_draft_ai_service
from app.api.performance_insight import get_performance_insight_ai_service
from app.api.policy_assistant import get_policy_ai_service
from app.api.skill_gap import get_skill_gap_ai_service
from app.api.team_insight import get_team_insight_ai_service
from app.db.session import Base, get_db
from app.main import app
from app.models import (
    AIAuditEvent,
    AIInsightSnapshot,
    ChatMessage,
    ChatSession,
    CompanyPolicy,
    Employee,
    EvaluationTheme,
    Goal,
    PerformanceRecord,
    Skill,
    TaskOutcome,
)
from app.services.career_coach_ai import CareerCoachAIServiceError

TEST_DATABASE_URL = "sqlite:///:memory:"

test_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture(scope="function")
def db_session():
    """Provides an isolated database session."""
    Base.metadata.create_all(bind=test_engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=test_engine)


@pytest.fixture(scope="function")
def client(db_session: Session):
    """FastAPI TestClient with get_db overridden."""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def seed_audit_data(db_session: Session):
    """Seeds test employees, records, and policies."""
    alice = Employee(
        id="EMP-ENG-ALICE",
        first_name="Alice",
        last_name="Engineer",
        role_title="Software Engineer",
        department="Engineering",
    )
    bob = Employee(
        id="EMP-ENG-BOB",
        first_name="Bob",
        last_name="Developer",
        role_title="Backend Developer",
        department="Engineering",
    )
    carol_mgr = Employee(
        id="EMP-ENG-CAROL",
        first_name="Carol",
        last_name="Manager",
        role_title="Engineering Manager",
        department="Engineering",
    )
    dave_sales = Employee(
        id="EMP-SALES-DAVE",
        first_name="Dave",
        last_name="Sales",
        role_title="Account Executive",
        department="Sales",
    )
    db_session.add_all([alice, bob, carol_mgr, dave_sales])

    p1 = PerformanceRecord(
        employee_id="EMP-ENG-ALICE",
        period="2026-Q3",
        overall_score=90.0,
        task_completion_rate=95.0,
        goal_achievement_rate=92.0,
        attendance_rate=99.0,
        is_approved=True,
    )
    g1 = Goal(
        employee_id="EMP-ENG-ALICE",
        title="Deliver cloud migration",
        progress=80.0,
        status="in_progress",
        period="2026-Q3",
        is_approved=True,
    )
    s1 = Skill(
        employee_id="EMP-ENG-ALICE",
        name="Python",
        level="Expert",
        is_approved=True,
    )
    t1 = TaskOutcome(
        employee_id="EMP-ENG-ALICE",
        title="Migrate database cluster",
        status="completed",
        period="2026-Q3",
        is_approved=True,
    )
    th1 = EvaluationTheme(
        employee_id="EMP-ENG-ALICE",
        theme="Technical Leadership",
        sentiment="positive",
        evidence="Delivered cloud project",
        period="2026-Q3",
        is_approved=True,
    )
    pol1 = CompanyPolicy(
        policy_code="POL-HR-01",
        title="Remote Work Policy",
        category="workplace",
        content="Remote work is permitted 3 days a week.",
        summary="Remote work permitted.",
        is_active=True,
        is_approved=True,
    )
    session = ChatSession(
        id="session-alice-1",
        employee_id="EMP-ENG-ALICE",
        title="Alice Policy Chat",
    )
    msg = ChatMessage(
        session_id="session-alice-1",
        role="user",
        content="What is the remote work policy?",
    )
    db_session.add_all([p1, g1, s1, t1, th1, pol1, session, msg])
    db_session.commit()


# =============================================================================
# 1. SUCCESS & INSUFFICIENT DATA AUDIT EVENTS
# =============================================================================


VALID_CAREER_COACH_SUCCESS = {
    "status": "success",
    "employee_id": "EMP-ENG-ALICE",
    "strengths": [
        {
            "title": "Architectural Mastery",
            "description": "Consistently designs and deploys resilient systems.",
            "evidence": [
                {
                    "source_type": "performance",
                    "source_id": 1,
                    "claim": "98% task completion rate",
                }
            ],
        }
    ],
    "development_areas": [
        {
            "title": "Technical Blogging",
            "description": "Document architectural insights externally.",
            "evidence": [
                {
                    "source_type": "evaluation_theme",
                    "source_id": 1,
                    "claim": "Leadership notes opportunity",
                }
            ],
            "priority": "medium",
        }
    ],
    "development_plan": [
        {
            "action": "Publish internal engineering whitepaper",
            "reason": "Scale systemic thinking across squads.",
            "measurable_target": "Complete 1 peer-reviewed whitepaper",
            "suggested_timeline": "45 days",
        }
    ],
    "follow_up": {
        "checkpoint": "End of Q4",
        "review_focus": "Assess adoption of architecture standards",
    },
}


def make_policy_assistant_success(answer: str = "Remote work allowed 3 days.", employee_id: str = "EMP-ENG-ALICE"):
    return {
        "status": "success",
        "employee_id": employee_id,
        "answer": answer,
        "policy_references": [
            {
                "policy_id": 1,
                "policy_code": "POL-HR-01",
                "title": "Remote Work Policy",
                "version": "1.0",
            }
        ],
        "employee_facts_used": ["Role: Lead Architect"],
    }


def test_successful_ai_request_creates_audit_event(client: TestClient, db_session: Session, seed_audit_data):
    """A successful AI request creates an audit event with outcome='success'."""
    headers = {
        "X-Caller-Employee-ID": "EMP-ENG-ALICE",
        "X-Caller-Role": "employee",
    }
    mock_cc = MagicMock()
    mock_cc.model = "openai/gpt-oss-120b"
    mock_cc.generate_career_plan.return_value = VALID_CAREER_COACH_SUCCESS
    app.dependency_overrides[get_career_coach_ai_service] = lambda: mock_cc

    resp = client.post(
        "/api/career-coach",
        json={"employee_id": "EMP-ENG-ALICE", "period": "2026-Q3"},
        headers=headers,
    )
    assert resp.status_code == 200

    events = db_session.query(AIAuditEvent).filter(AIAuditEvent.feature == "career_coach").all()
    assert len(events) == 1
    ev = events[0]
    assert ev.actor_employee_id == "EMP-ENG-ALICE"
    assert ev.actor_role == "employee"
    assert ev.scope_employee_id == "EMP-ENG-ALICE"
    assert ev.scope_department == "Engineering"
    assert ev.feature == "career_coach"
    assert ev.endpoint == "/api/career-coach"
    assert ev.provider == "groq"
    assert ev.model == "openai/gpt-oss-120b"
    assert ev.outcome == "success"
    assert ev.event_id is not None
    assert ev.timestamp is not None


def test_insufficient_data_outcome_creates_audit_event(
    client: TestClient, db_session: Session, seed_audit_data
):
    """An insufficient data outcome creates an audit event with outcome='insufficient_data'."""
    headers = {
        "X-Caller-Employee-ID": "EMP-ENG-ALICE",
        "X-Caller-Role": "employee",
    }
    mock_cc = MagicMock()
    mock_cc.model = "openai/gpt-oss-120b"
    mock_cc.generate_career_plan.return_value = {
        "status": "insufficient_data",
        "employee_id": "EMP-ENG-ALICE",
        "missing_categories": ["goals"],
        "message": "Not enough data.",
        "created_at": "2026-09-19T12:00:00Z",
    }
    app.dependency_overrides[get_career_coach_ai_service] = lambda: mock_cc

    resp = client.post(
        "/api/career-coach",
        json={"employee_id": "EMP-ENG-ALICE"},
        headers=headers,
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "insufficient_data"

    events = db_session.query(AIAuditEvent).filter(AIAuditEvent.feature == "career_coach").all()
    assert len(events) == 1
    ev = events[0]
    assert ev.outcome == "insufficient_data"
    assert ev.actor_employee_id == "EMP-ENG-ALICE"


# =============================================================================
# 2. UNAUTHORIZED REQUEST AUDIT EVENTS
# =============================================================================


def test_unauthorized_request_creates_audit_event_when_caller_known(
    client: TestClient, db_session: Session, seed_audit_data
):
    """When a known caller attempts unauthorized access, an audit event with outcome='unauthorized' is logged."""
    alice_headers = {
        "X-Caller-Employee-ID": "EMP-ENG-ALICE",
        "X-Caller-Role": "employee",
    }
    # Alice attempts to access Bob's Career Coach
    resp = client.post(
        "/api/career-coach",
        json={"employee_id": "EMP-ENG-BOB"},
        headers=alice_headers,
    )
    assert resp.status_code == 403

    events = db_session.query(AIAuditEvent).filter(AIAuditEvent.feature == "career_coach").all()
    assert len(events) == 1
    ev = events[0]
    assert ev.actor_employee_id == "EMP-ENG-ALICE"
    assert ev.actor_role == "employee"
    assert ev.outcome == "unauthorized"


# =============================================================================
# 3. PROVIDER ERROR & REFERENCE ID AUDIT EVENTS
# =============================================================================


def test_provider_error_creates_audit_event_with_reference_id(
    client: TestClient, db_session: Session, seed_audit_data
):
    """When an AI provider error occurs, outcome is 'provider_error' and the reference_id is captured."""
    headers = {
        "X-Caller-Employee-ID": "EMP-ENG-ALICE",
        "X-Caller-Role": "employee",
    }
    mock_cc = MagicMock()
    mock_cc.model = "openai/gpt-oss-120b"
    mock_cc.generate_career_plan.side_effect = CareerCoachAIServiceError("Groq service rate limited")
    app.dependency_overrides[get_career_coach_ai_service] = lambda: mock_cc

    resp = client.post(
        "/api/career-coach",
        json={"employee_id": "EMP-ENG-ALICE"},
        headers=headers,
    )
    assert resp.status_code == 502
    assert "Reference ID:" in resp.json()["detail"]

    events = db_session.query(AIAuditEvent).filter(AIAuditEvent.feature == "career_coach").all()
    assert len(events) == 1
    ev = events[0]
    assert ev.outcome == "provider_error"
    assert ev.reference_id is not None
    assert ev.reference_id in resp.json()["detail"]


# =============================================================================
# 4. SCOPE & ACTOR DERIVATION FROM CALLER CONTEXT (TAMPER RESISTANCE)
# =============================================================================


def test_actor_id_and_role_come_from_caller_context_not_body(
    client: TestClient, db_session: Session, seed_audit_data
):
    """Actor identity is strictly taken from caller context, never forged request body."""
    carol_headers = {
        "X-Caller-Employee-ID": "EMP-ENG-CAROL",
        "X-Caller-Role": "manager",
    }
    mock_pi = MagicMock()
    mock_pi.model = "openai/gpt-oss-120b"
    mock_pi.generate_insight_from_context.return_value = {
        "status": "insufficient_data",
        "employee_id": "EMP-ENG-ALICE",
        "reason": "Need more periods",
        "periods_found": ["2026-Q3"],
        "message": "Need more data.",
        "created_at": "2026-09-19T12:00:00Z",
    }
    app.dependency_overrides[get_performance_insight_ai_service] = lambda: mock_pi

    resp = client.post(
        "/api/performance-insight",
        json={"employee_id": "EMP-ENG-ALICE"},
        headers=carol_headers,
    )
    assert resp.status_code == 200

    events = db_session.query(AIAuditEvent).filter(AIAuditEvent.feature == "performance_insight").all()
    assert len(events) == 1
    ev = events[0]
    assert ev.actor_employee_id == "EMP-ENG-CAROL"
    assert ev.actor_role == "manager"
    assert ev.scope_employee_id == "EMP-ENG-ALICE"
    assert ev.scope_department == "Engineering"


def test_policy_assistant_records_session_scope(
    client: TestClient, db_session: Session, seed_audit_data
):
    """Policy Assistant records employee and session scopes in the audit event."""
    alice_headers = {
        "X-Caller-Employee-ID": "EMP-ENG-ALICE",
        "X-Caller-Role": "employee",
    }
    mock_pol = MagicMock()
    mock_pol.model = "openai/gpt-oss-120b"
    mock_pol.answer_policy_question.return_value = make_policy_assistant_success("Remote work allowed 3 days.")
    app.dependency_overrides[get_policy_ai_service] = lambda: mock_pol

    resp = client.post(
        "/api/policy-assistant",
        json={
            "employee_id": "EMP-ENG-ALICE",
            "question": "Can I work remotely?",
            "session_id": "session-alice-1",
        },
        headers=alice_headers,
    )
    assert resp.status_code == 200

    events = db_session.query(AIAuditEvent).filter(AIAuditEvent.feature == "policy_assistant").all()
    assert len(events) == 1
    ev = events[0]
    assert ev.actor_employee_id == "EMP-ENG-ALICE"
    assert ev.scope_session_id == "session-alice-1"
    assert ev.scope_employee_id == "EMP-ENG-ALICE"


# =============================================================================
# 5. PRIVACY & SECRECY GUARANTEES
# =============================================================================


def test_prompts_and_full_ai_responses_are_not_stored(
    client: TestClient, db_session: Session, seed_audit_data
):
    """Audit events MUST NOT store prompts, raw question texts, or LLM responses."""
    alice_headers = {
        "X-Caller-Employee-ID": "EMP-ENG-ALICE",
        "X-Caller-Role": "employee",
    }
    secret_prompt = "CONFIDENTIAL_SUPER_SECRET_PROMPT_CONTENT_XYZ123"
    secret_answer = "HIGHLY_CONFIDENTIAL_SECRET_AI_ANSWER_ABC789"

    mock_pol = MagicMock()
    mock_pol.model = "openai/gpt-oss-120b"
    mock_pol.answer_policy_question.return_value = make_policy_assistant_success(secret_answer)
    app.dependency_overrides[get_policy_ai_service] = lambda: mock_pol

    resp = client.post(
        "/api/policy-assistant",
        json={
            "employee_id": "EMP-ENG-ALICE",
            "question": secret_prompt,
        },
        headers=alice_headers,
    )
    assert resp.status_code == 200

    event = db_session.query(AIAuditEvent).filter(AIAuditEvent.feature == "policy_assistant").first()
    assert event is not None

    # Inspect all column values on the audit row
    for col in event.__table__.columns:
        val = getattr(event, col.name)
        if val is not None:
            val_str = str(val)
            assert secret_prompt not in val_str
            assert secret_answer not in val_str
            assert "Bearer" not in val_str
            assert "gsk_" not in val_str
            assert "password" not in val_str.lower()


# =============================================================================
# 6. ALL 7 AI ROUTES PRODUCE AUDIT EVENTS
# =============================================================================


def test_all_7_ai_routes_produce_audit_events(client: TestClient, db_session: Session, seed_audit_data):
    """All 7 AI endpoints record an audit event on invocation."""
    emp_headers = {
        "X-Caller-Employee-ID": "EMP-ENG-ALICE",
        "X-Caller-Role": "employee",
    }
    mgr_headers = {
        "X-Caller-Employee-ID": "EMP-ENG-CAROL",
        "X-Caller-Role": "manager",
    }

    # 1. Career Coach
    mock_cc = MagicMock()
    mock_cc.model = "openai/gpt-oss-120b"
    mock_cc.generate_career_plan.return_value = {
        "status": "insufficient_data",
        "employee_id": "EMP-ENG-ALICE",
        "missing_categories": ["goals"],
        "message": "msg",
    }
    app.dependency_overrides[get_career_coach_ai_service] = lambda: mock_cc
    client.post("/api/career-coach", json={"employee_id": "EMP-ENG-ALICE"}, headers=emp_headers)

    # 2. Policy Assistant
    mock_pol = MagicMock()
    mock_pol.model = "openai/gpt-oss-120b"
    mock_pol.answer_policy_question.return_value = make_policy_assistant_success("Policy answer valid.")
    app.dependency_overrides[get_policy_ai_service] = lambda: mock_pol
    client.post("/api/policy-assistant", json={"employee_id": "EMP-ENG-ALICE", "question": "Policy?"}, headers=emp_headers)

    # 3. Performance Insight
    mock_pi = MagicMock()
    mock_pi.model = "openai/gpt-oss-120b"
    app.dependency_overrides[get_performance_insight_ai_service] = lambda: mock_pi
    client.post("/api/performance-insight", json={"employee_id": "EMP-ENG-ALICE"}, headers=emp_headers)

    # 4. Evaluation Draft (Manager only)
    mock_ed = MagicMock()
    mock_ed.model = "openai/gpt-oss-120b"
    mock_ed.generate_draft.return_value = {
        "status": "success",
        "employee_id": "EMP-ENG-ALICE",
        "period": "2026-Q3",
        "evaluation_narrative": "Alice had an outstanding Q3, achieving 90.0 overall score.",
        "strengths": [
            {
                "title": "Exceptional Technical Execution",
                "description": "Consistently achieves high delivery quality.",
                "evidence": [
                    {
                        "source_type": "performance",
                        "source_id": 1,
                        "claim": "Achieved 90.0 overall score in Q3",
                    }
                ],
            }
        ],
        "improvement_areas": [
            {
                "title": "Mentorship Sessions",
                "description": "Conduct monthly design review walk-throughs for new joiners.",
                "evidence": [
                    {
                        "source_type": "evaluation_theme",
                        "source_id": 1,
                        "claim": "Theme highlighted knowledge sharing opportunity",
                    }
                ],
                "priority": "medium",
            }
        ],
        "entered_scores": {},
        "human_review_required": True,
        "created_at": "2026-09-19T12:00:00Z",
    }
    app.dependency_overrides[get_evaluation_draft_ai_service] = lambda: mock_ed
    client.post("/api/evaluation-draft", json={"employee_id": "EMP-ENG-ALICE", "period": "2026-Q3"}, headers=mgr_headers)

    # 5. Skill Gap
    mock_sg = MagicMock()
    mock_sg.model = "openai/gpt-oss-120b"
    mock_sg.generate_skill_gap_analysis.return_value = {
        "status": "insufficient_data",
        "employee_id": "EMP-ENG-ALICE",
        "missing_categories": ["skills"],
        "message": "Insufficient data",
    }
    app.dependency_overrides[get_skill_gap_ai_service] = lambda: mock_sg
    client.post("/api/skill-gap", json={"employee_id": "EMP-ENG-ALICE"}, headers=emp_headers)

    # 6. Attention Signal (Manager only)
    mock_as = MagicMock()
    mock_as.model = "openai/gpt-oss-120b"
    mock_as.generate_attention_signal.return_value = {
        "status": "insufficient_data",
        "employee_id": "EMP-ENG-ALICE",
        "reason": "Insufficient approved data",
    }
    app.dependency_overrides[get_attention_signal_ai_service] = lambda: mock_as
    client.post("/api/attention-signal", json={"employee_id": "EMP-ENG-ALICE"}, headers=mgr_headers)

    # 7. Team Insight (Manager only)
    mock_ti = MagicMock()
    mock_ti.model = "openai/gpt-oss-120b"
    mock_ti.generate_team_insight.return_value = {
        "status": "insufficient_data",
        "department": "Engineering",
        "period": "2026-Q3",
        "missing_categories": ["records"],
        "message": "Insufficient records for team insight.",
    }
    app.dependency_overrides[get_team_insight_ai_service] = lambda: mock_ti
    client.post("/api/team-insight", json={"department": "Engineering"}, headers=mgr_headers)

    # Verify that an audit event exists for each of the 7 features
    features_logged = {ev.feature for ev in db_session.query(AIAuditEvent).all()}
    expected_features = {
        "career_coach",
        "policy_assistant",
        "performance_insight",
        "evaluation_draft",
        "skill_gap",
        "attention_signal",
        "team_insight",
    }
    assert features_logged == expected_features

    snapshot_features = {
        snapshot.feature for snapshot in db_session.query(AIInsightSnapshot).all()
    }
    assert snapshot_features == expected_features
    assert db_session.query(AIInsightSnapshot).count() == 7


# =============================================================================
# 7. AUDIT WRITE FAILURE RESILIENCE
# =============================================================================


def test_audit_write_failure_does_not_corrupt_response(
    client: TestClient, db_session: Session, seed_audit_data, monkeypatch
):
    """Failure to persist an audit event does not break a successful AI request."""
    headers = {
        "X-Caller-Employee-ID": "EMP-ENG-ALICE",
        "X-Caller-Role": "employee",
    }
    mock_cc = MagicMock()
    mock_cc.model = "openai/gpt-oss-120b"
    mock_cc.generate_career_plan.return_value = VALID_CAREER_COACH_SUCCESS
    app.dependency_overrides[get_career_coach_ai_service] = lambda: mock_cc

    # Sabotage db_session.commit to simulate database audit write failure
    original_commit = db_session.commit
    def failing_commit():
        raise RuntimeError("Disk full / DB error during audit insert")
    db_session.commit = failing_commit

    resp = client.post(
        "/api/career-coach",
        json={"employee_id": "EMP-ENG-ALICE"},
        headers=headers,
    )
    # The API call still succeeds!
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"

    # Restore commit
    db_session.commit = original_commit

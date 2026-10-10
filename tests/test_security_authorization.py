"""Comprehensive security and authorization tests for AI routes.

Covers:
- Missing caller headers -> 401
- Invalid caller employee ID -> 401
- Invalid caller role -> 401
- Employee accessing another employee -> 403
- Employee accessing manager-only functionality -> 403
- Manager accessing employee in same department -> allowed (200)
- Manager accessing employee in another department -> 403
- Manager requesting another department in Team Insight -> 403
- Manager requesting own department in Team Insight -> allowed (200)
- hr_admin accessing another department / employee -> allowed (200)
- Policy session hijacking attempt -> 403
- Tampering with request-body employee_id / department cannot expand authorization scope
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.career_coach import get_career_coach_ai_service
from app.api.performance_insight import get_performance_insight_ai_service
from app.api.skill_gap import get_skill_gap_ai_service
from app.api.team_insight import get_team_insight_ai_service
from app.db.session import Base, get_db
from app.main import app
from app.models import (
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
def seed_auth_data(db_session: Session):
    """Seeds employees across multiple departments and roles."""
    # Engineering employees
    alice_dev = Employee(
        id="EMP-ENG-ALICE",
        first_name="Alice",
        last_name="Engineer",
        role_title="Senior Software Engineer",
        department="Engineering",
    )
    bob_dev = Employee(
        id="EMP-ENG-BOB",
        first_name="Bob",
        last_name="Developer",
        role_title="Software Engineer",
        department="Engineering",
    )
    carol_mgr = Employee(
        id="EMP-ENG-CAROL-MGR",
        first_name="Carol",
        last_name="Manager",
        role_title="Engineering Manager",
        department="Engineering",
    )

    # Sales employees
    dave_sales = Employee(
        id="EMP-SALES-DAVE",
        first_name="Dave",
        last_name="Closer",
        role_title="Sales Executive",
        department="Sales",
    )
    eve_sales_mgr = Employee(
        id="EMP-SALES-EVE-MGR",
        first_name="Eve",
        last_name="Director",
        role_title="Sales Director",
        department="Sales",
    )

    # HR Admin
    helen_admin = Employee(
        id="EMP-HR-HELEN",
        first_name="Helen",
        last_name="Admin",
        role_title="Chief People Officer",
        department="Human Resources",
    )

    db_session.add_all([alice_dev, bob_dev, carol_mgr, dave_sales, eve_sales_mgr, helen_admin])
    db_session.commit()

    # Seed basic records for Alice so AI calls succeed when authorized
    p1 = PerformanceRecord(
        employee_id="EMP-ENG-ALICE",
        period="2026-Q2",
        overall_score=85.0,
        task_completion_rate=90.0,
        goal_achievement_rate=85.0,
        attendance_rate=95.0,
        is_approved=True,
    )
    p2 = PerformanceRecord(
        employee_id="EMP-ENG-ALICE",
        period="2026-Q3",
        overall_score=90.0,
        task_completion_rate=95.0,
        goal_achievement_rate=90.0,
        attendance_rate=98.0,
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
        name="Python Backend",
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
        theme="Technical Architecture",
        sentiment="positive",
        evidence="Exemplary system design",
        period="2026-Q3",
        is_approved=True,
    )
    pol1 = CompanyPolicy(
        policy_code="POL-SEC-01",
        title="Data Protection Policy",
        category="security",
        content="All data must be encrypted.",
        summary="Data encryption required.",
        is_active=True,
        is_approved=True,
    )

    # Bob's ChatSession (for session hijacking tests)
    bob_session = ChatSession(
        id="session-bob-12345",
        employee_id="EMP-ENG-BOB",
        title="Bob's policy chat",
        summary="Discussion on PTO",
    )
    bob_msg = ChatMessage(
        session_id="session-bob-12345",
        role="user",
        content="What is the remote work policy?",
    )

    db_session.add_all([p1, p2, g1, s1, t1, th1, pol1, bob_session, bob_msg])
    db_session.commit()


# =============================================================================
# 1. AUTHENTICATION (401) TESTS
# =============================================================================


@pytest.mark.parametrize(
    "endpoint,payload",
    [
        ("/api/career-coach", {"employee_id": "EMP-ENG-ALICE"}),
        ("/api/policy-assistant", {"employee_id": "EMP-ENG-ALICE", "question": "What is the policy?"}),
        ("/api/performance-insight", {"employee_id": "EMP-ENG-ALICE"}),
        ("/api/evaluation-draft", {"employee_id": "EMP-ENG-ALICE"}),
        ("/api/skill-gap", {"employee_id": "EMP-ENG-ALICE"}),
        ("/api/attention-signal", {"employee_id": "EMP-ENG-ALICE"}),
        ("/api/team-insight", {"department": "Engineering"}),
    ],
)
def test_missing_caller_headers_returns_401(client: TestClient, seed_auth_data, endpoint, payload):
    """Missing X-Caller-Employee-ID or X-Caller-Role headers returns HTTP 401."""
    # 1. No headers at all
    resp = client.post(endpoint, json=payload)
    assert resp.status_code == 401
    assert "Missing or empty required header" in resp.json()["detail"]

    # 2. Only Employee ID header
    resp = client.post(endpoint, json=payload, headers={"X-Caller-Employee-ID": "EMP-ENG-ALICE"})
    assert resp.status_code == 401
    assert "X-Caller-Role" in resp.json()["detail"]

    # 3. Only Role header
    resp = client.post(endpoint, json=payload, headers={"X-Caller-Role": "employee"})
    assert resp.status_code == 401
    assert "X-Caller-Employee-ID" in resp.json()["detail"]


def test_invalid_caller_employee_id_returns_401(client: TestClient, seed_auth_data):
    """Caller Employee ID not found in database returns HTTP 401."""
    headers = {
        "X-Caller-Employee-ID": "NONEXISTENT-EMPLOYEE-ID",
        "X-Caller-Role": "employee",
    }
    resp = client.post(
        "/api/career-coach",
        json={"employee_id": "EMP-ENG-ALICE"},
        headers=headers,
    )
    assert resp.status_code == 401
    assert "Caller does not exist" in resp.json()["detail"]


def test_invalid_caller_role_returns_401(client: TestClient, seed_auth_data):
    """Caller Role not in allowed roles returns HTTP 401."""
    headers = {
        "X-Caller-Employee-ID": "EMP-ENG-ALICE",
        "X-Caller-Role": "superuser",  # Invalid role
    }
    resp = client.post(
        "/api/career-coach",
        json={"employee_id": "EMP-ENG-ALICE"},
        headers=headers,
    )
    assert resp.status_code == 401
    assert "Invalid caller role" in resp.json()["detail"]


# =============================================================================
# 2. EMPLOYEE ROLE AUTHORIZATION (403 vs ALLOWED)
# =============================================================================


def test_employee_accessing_another_employee_returns_403(client: TestClient, seed_auth_data):
    """An employee caller attempting to access another employee's records returns HTTP 403."""
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
    assert "employees may only access their own records" in resp.json()["detail"]

    # Alice attempts to access Bob's Performance Insight
    resp = client.post(
        "/api/performance-insight",
        json={"employee_id": "EMP-ENG-BOB"},
        headers=alice_headers,
    )
    assert resp.status_code == 403
    assert "employees may only access their own records" in resp.json()["detail"]

    # Alice attempts to access Bob's Skill Gap
    resp = client.post(
        "/api/skill-gap",
        json={"employee_id": "EMP-ENG-BOB"},
        headers=alice_headers,
    )
    assert resp.status_code == 403
    assert "employees may only access their own records" in resp.json()["detail"]


@pytest.mark.parametrize(
    "endpoint,payload",
    [
        ("/api/evaluation-draft", {"employee_id": "EMP-ENG-ALICE", "period": "2026-Q3"}),
        ("/api/attention-signal", {"employee_id": "EMP-ENG-ALICE"}),
        ("/api/team-insight", {"department": "Engineering"}),
    ],
)
def test_employee_accessing_manager_only_features_returns_403(
    client: TestClient, seed_auth_data, endpoint, payload
):
    """An employee caller attempting to access manager-only features returns HTTP 403."""
    alice_headers = {
        "X-Caller-Employee-ID": "EMP-ENG-ALICE",
        "X-Caller-Role": "employee",
    }
    resp = client.post(endpoint, json=payload, headers=alice_headers)
    assert resp.status_code == 403
    assert "not authorized to access" in resp.json()["detail"]


def test_employee_accessing_own_records_is_allowed(client: TestClient, seed_auth_data):
    """An employee caller accessing their own records is authorized."""
    alice_headers = {
        "X-Caller-Employee-ID": "EMP-ENG-ALICE",
        "X-Caller-Role": "employee",
    }
    mock_cc = MagicMock()
    mock_cc.generate_career_plan.return_value = {
        "status": "insufficient_data",
        "employee_id": "EMP-ENG-ALICE",
        "missing_categories": ["goals"],
        "message": "Authorized test response",
    }
    app.dependency_overrides[get_career_coach_ai_service] = lambda: mock_cc

    resp = client.post(
        "/api/career-coach",
        json={"employee_id": "EMP-ENG-ALICE"},
        headers=alice_headers,
    )
    assert resp.status_code == 200
    assert resp.json()["employee_id"] == "EMP-ENG-ALICE"


# =============================================================================
# 3. MANAGER ROLE AUTHORIZATION
# =============================================================================


def test_manager_accessing_employee_in_same_department_is_allowed(client: TestClient, seed_auth_data):
    """Carol (Engineering Manager) accessing Alice (Engineering) is allowed."""
    carol_headers = {
        "X-Caller-Employee-ID": "EMP-ENG-CAROL-MGR",
        "X-Caller-Role": "manager",
    }
    mock_pi = MagicMock()
    mock_pi.generate_insight_from_context.return_value = {
        "status": "insufficient_data",
        "employee_id": "EMP-ENG-ALICE",
        "reason": "Authorized manager test response",
        "periods_found": [],
        "message": "Authorized manager test response",
    }
    app.dependency_overrides[get_performance_insight_ai_service] = lambda: mock_pi

    resp = client.post(
        "/api/performance-insight",
        json={"employee_id": "EMP-ENG-ALICE"},
        headers=carol_headers,
    )
    assert resp.status_code == 200
    assert resp.json()["employee_id"] == "EMP-ENG-ALICE"


def test_manager_accessing_employee_in_another_department_returns_403(client: TestClient, seed_auth_data):
    """Carol (Engineering Manager) accessing Dave (Sales) returns HTTP 403."""
    carol_headers = {
        "X-Caller-Employee-ID": "EMP-ENG-CAROL-MGR",
        "X-Caller-Role": "manager",
    }
    # Attempt to access Dave in Sales
    resp = client.post(
        "/api/evaluation-draft",
        json={"employee_id": "EMP-SALES-DAVE", "period": "2026-Q3"},
        headers=carol_headers,
    )
    assert resp.status_code == 403
    assert "managers may only access employees in their own department" in resp.json()["detail"]


def test_manager_requesting_another_department_in_team_insight_returns_403(
    client: TestClient, seed_auth_data
):
    """Carol (Engineering Manager) requesting Team Insight for Sales returns HTTP 403."""
    carol_headers = {
        "X-Caller-Employee-ID": "EMP-ENG-CAROL-MGR",
        "X-Caller-Role": "manager",
    }
    resp = client.post(
        "/api/team-insight",
        json={"department": "Sales"},
        headers=carol_headers,
    )
    assert resp.status_code == 403
    assert "managers may only access Team Insight for their own department" in resp.json()["detail"]


def test_manager_requesting_own_department_in_team_insight_is_allowed(client: TestClient, seed_auth_data):
    """Carol (Engineering Manager) requesting Team Insight for Engineering is allowed."""
    carol_headers = {
        "X-Caller-Employee-ID": "EMP-ENG-CAROL-MGR",
        "X-Caller-Role": "manager",
    }
    mock_ti = MagicMock()
    mock_ti.generate_team_insight.return_value = {
        "status": "success",
        "department": "Engineering",
        "team_size": 3,
        "team_findings": {
            "executive_summary": "Solid velocity across the team.",
            "overdue_workload": {
                "summary": "No blockers.",
                "total_blocked_tasks": 0,
                "total_delayed_goals": 0,
                "affected_member_count": 0,
            },
            "completion_trends": {
                "summary": "Strong metrics.",
                "team_avg_task_completion": 92.0,
                "team_avg_goal_achievement": 88.0,
                "team_avg_overall_score": 88.0,
                "direction": "improved",
            },
            "skill_gap_patterns": {
                "summary": "Good skills.",
                "top_common_gaps": ["Python Backend"],
            },
            "evaluation_theme_patterns": {
                "summary": "Positive leadership.",
                "top_positive_themes": ["Technical Architecture"],
                "top_needs_improvement_themes": [],
            },
        },
        "drill_down_factors": [],
        "recommended_management_actions": [],
        "advisory_disclaimer": "Advisory only.",
        "created_at": "2026-09-19T12:00:00Z",
    }
    app.dependency_overrides[get_team_insight_ai_service] = lambda: mock_ti

    resp = client.post(
        "/api/team-insight",
        json={"department": "Engineering"},
        headers=carol_headers,
    )
    assert resp.status_code == 200
    assert resp.json()["department"] == "Engineering"


# =============================================================================
# 4. HR ADMIN ROLE AUTHORIZATION
# =============================================================================


def test_hr_admin_accessing_any_department_or_employee_is_allowed(client: TestClient, seed_auth_data):
    """Helen (HR Admin) can access employees and departments across the organization."""
    admin_headers = {
        "X-Caller-Employee-ID": "EMP-HR-HELEN",
        "X-Caller-Role": "hr_admin",
    }
    # 1. Admin accessing Engineering employee
    mock_sg = MagicMock()
    mock_sg.generate_skill_gap_analysis.return_value = {
        "status": "success",
        "employee_id": "EMP-ENG-ALICE",
        "skill_gaps": [],
        "recommendations": [],
    }
    app.dependency_overrides[get_skill_gap_ai_service] = lambda: mock_sg

    resp = client.post(
        "/api/skill-gap",
        json={"employee_id": "EMP-ENG-ALICE"},
        headers=admin_headers,
    )
    assert resp.status_code == 200

    # 2. Admin accessing Sales department for Team Insight
    mock_ti = MagicMock()
    mock_ti.generate_team_insight.return_value = {
        "status": "success",
        "department": "Sales",
        "team_size": 2,
        "team_findings": {
            "executive_summary": "Sales team overview.",
            "overdue_workload": {
                "summary": "Workload summary.",
                "total_blocked_tasks": 0,
                "total_delayed_goals": 0,
                "affected_member_count": 0,
            },
            "completion_trends": {
                "summary": "Sales completion.",
                "team_avg_task_completion": 85.0,
                "team_avg_goal_achievement": 80.0,
                "team_avg_overall_score": 82.0,
                "direction": "stable",
            },
            "skill_gap_patterns": {
                "summary": "Sales gaps.",
                "top_common_gaps": [],
            },
            "evaluation_theme_patterns": {
                "summary": "Themes.",
                "top_positive_themes": [],
                "top_needs_improvement_themes": [],
            },
        },
        "drill_down_factors": [],
        "recommended_management_actions": [],
        "advisory_disclaimer": "Advisory only.",
        "created_at": "2026-09-19T12:00:00Z",
    }
    app.dependency_overrides[get_team_insight_ai_service] = lambda: mock_ti

    resp = client.post(
        "/api/team-insight",
        json={"department": "Sales"},
        headers=admin_headers,
    )
    assert resp.status_code == 200
    assert resp.json()["department"] == "Sales"


# =============================================================================
# 5. POLICY ASSISTANT SESSION HIJACKING ATTEMPT (403)
# =============================================================================


def test_policy_session_hijacking_attempt_returns_403(client: TestClient, seed_auth_data):
    """Alice attempting to provide Bob's session_id in Policy Assistant is rejected with 403."""
    alice_headers = {
        "X-Caller-Employee-ID": "EMP-ENG-ALICE",
        "X-Caller-Role": "employee",
    }
    # Alice passes her own employee_id, but Bob's session_id
    resp = client.post(
        "/api/policy-assistant",
        json={
            "employee_id": "EMP-ENG-ALICE",
            "question": "Can I work remotely?",
            "session_id": "session-bob-12345",
        },
        headers=alice_headers,
    )
    assert resp.status_code == 403
    assert "chat session belongs to another employee" in resp.json()["detail"]


# =============================================================================
# 6. TAMPERING WITH REQUEST BODY CANNOT EXPAND SCOPE
# =============================================================================


def test_tampering_with_request_body_cannot_expand_scope(client: TestClient, seed_auth_data):
    """Employee attempting to tamper with employee_id or department cannot bypass authorization."""
    alice_headers = {
        "X-Caller-Employee-ID": "EMP-ENG-ALICE",
        "X-Caller-Role": "employee",
    }
    # 1. Tampering with employee_id on Attention Signal -> 403 (employee cannot access)
    resp = client.post(
        "/api/attention-signal",
        json={"employee_id": "EMP-SALES-DAVE"},
        headers=alice_headers,
    )
    assert resp.status_code == 403

    # 2. Tampering with employee_id on Skill Gap -> 403
    resp = client.post(
        "/api/skill-gap",
        json={"employee_id": "EMP-SALES-DAVE"},
        headers=alice_headers,
    )
    assert resp.status_code == 403

    # 3. Manager tampering with department on Team Insight -> 403
    carol_headers = {
        "X-Caller-Employee-ID": "EMP-ENG-CAROL-MGR",
        "X-Caller-Role": "manager",
    }
    resp = client.post(
        "/api/team-insight",
        json={"department": "Human Resources"},
        headers=carol_headers,
    )
    assert resp.status_code == 403


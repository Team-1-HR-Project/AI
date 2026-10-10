"""API integration tests for Feature #7: Team Insight Summary - Manager.

Covers:
1. Successful valid request.
2. Successful request with explicit period.
3. Successful request with period omitted.
4. Unknown department -> insufficient_data.
5. Department with no approved relevant records -> insufficient_data.
6. Department isolation.
7. Approved-only behavior.
8. Response contains team_size and team-level findings.
9. Response does not expose employee names.
10. Response does not expose employee IDs.
11. Response does not expose individual-level findings/rankings.
12. Validation error for missing department -> 422.
13. Validation error for invalid request fields -> 422.
14. Extra forbidden request fields -> 422.
15. AI grounding failure -> 502 with safe opaque reference.
16. AI/provider failure -> 502 with safe opaque reference.
17. Ensure insufficient-data path does not call the AI service.
18. Ensure advisory/human-review fields remain enforced.
"""

from __future__ import annotations

import json
import re
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.team_insight import get_team_insight_ai_service
from app.db.session import Base, get_db
from app.main import app
from app.models import (
    Employee,
    EvaluationTheme,
    Goal,
    PerformanceRecord,
    Skill,
    TaskOutcome,
)
from app.schemas.career_coach import PriorityLevel
from app.schemas.performance_insight import TrendDirection
from app.services.team_insight_ai import (
    TeamInsightAIService,
    TeamInsightAIServiceError,
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
    """Provides an isolated in-memory database session."""
    Base.metadata.create_all(bind=test_engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=test_engine)


@pytest.fixture(scope="function")
def client(db_session: Session):
    """FastAPI TestClient with overridden get_db dependency and hr_admin caller context.

    Uses hr_admin role so the caller can query any department without restriction.
    The caller employee is seeded in the "Management" department so it is never
    counted inside the Engineering team_size.
    """
    # Seed the caller in "Management" — not counted in Engineering team_size
    mgr = Employee(
        id="EMP-MGR-ENGINEERING",
        first_name="Manager",
        last_name="Test",
        role_title="Engineering Manager",
        department="Management",
    )
    db_session.merge(mgr)
    db_session.commit()

    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(
        app,
        headers={
            "X-Caller-Employee-ID": "EMP-MGR-ENGINEERING",
            "X-Caller-Role": "hr_admin",
        },
    ) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture(scope="function")
def hr_admin_client(db_session: Session):
    """FastAPI TestClient using hr_admin role for cross-department access tests.

    hr_admin callers can request any department, unlike managers who are restricted
    to their own department. Used for tests 4, 5, and 17 that send unknown/cross-dept requests.
    """
    admin = Employee(
        id="EMP-HRADMIN-001",
        first_name="Admin",
        last_name="User",
        role_title="HR Admin",
        department="Human Resources",
    )
    db_session.merge(admin)
    db_session.commit()

    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(
        app,
        headers={
            "X-Caller-Employee-ID": "EMP-HRADMIN-001",
            "X-Caller-Role": "hr_admin",
        },
    ) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def seed_department_data(db_session: Session):
    """Seeds Engineering and Sales employees with approved and unapproved records."""
    emp_alice = Employee(
        id="EMP-ENG-ALICE",
        first_name="Alice",
        last_name="Smith",
        role_title="Lead Software Engineer",
        department="Engineering",
    )
    emp_bob = Employee(
        id="EMP-ENG-BOB",
        first_name="Bob",
        last_name="Jones",
        role_title="Senior DevOps Engineer",
        department="Engineering",
    )
    emp_charlie = Employee(
        id="EMP-ENG-CHARLIE",
        first_name="Charlie",
        last_name="Brown",
        role_title="Software Engineer",
        department="Engineering",
    )
    emp_sarah = Employee(
        id="EMP-SALES-SARAH",
        first_name="Sarah",
        last_name="Miller",
        role_title="Account Executive",
        department="Sales",
    )
    emp_hr = Employee(
        id="EMP-HR-HOPE",
        first_name="Hope",
        last_name="Vance",
        role_title="HR Coordinator",
        department="Human Resources",
    )
    db_session.add_all([emp_alice, emp_bob, emp_charlie, emp_sarah, emp_hr])
    db_session.commit()

    # 1. Performance Records (Engineering: 2026-Q2 and 2026-Q3)
    p_alice_q2 = PerformanceRecord(
        employee_id="EMP-ENG-ALICE",
        period="2026-Q2",
        overall_score=85.0,
        task_completion_rate=90.0,
        goal_achievement_rate=80.0,
        attendance_rate=95.0,
        is_approved=True,
    )
    p_alice_q3 = PerformanceRecord(
        employee_id="EMP-ENG-ALICE",
        period="2026-Q3",
        overall_score=95.0,
        task_completion_rate=98.0,
        goal_achievement_rate=92.0,
        attendance_rate=96.0,
        is_approved=True,
    )
    p_bob_q2 = PerformanceRecord(
        employee_id="EMP-ENG-BOB",
        period="2026-Q2",
        overall_score=80.0,
        task_completion_rate=80.0,
        goal_achievement_rate=70.0,
        attendance_rate=90.0,
        is_approved=True,
    )
    p_bob_q3 = PerformanceRecord(
        employee_id="EMP-ENG-BOB",
        period="2026-Q3",
        overall_score=85.0,
        task_completion_rate=92.0,
        goal_achievement_rate=88.0,
        attendance_rate=92.0,
        is_approved=True,
    )
    p_charlie_q3 = PerformanceRecord(
        employee_id="EMP-ENG-CHARLIE",
        period="2026-Q3",
        overall_score=75.0,
        task_completion_rate=80.0,
        goal_achievement_rate=90.0,
        attendance_rate=90.0,
        is_approved=True,
    )

    # Unapproved Performance Record
    p_unapproved = PerformanceRecord(
        employee_id="EMP-ENG-CHARLIE",
        period="2026-Q3",
        overall_score=10.0,
        task_completion_rate=10.0,
        goal_achievement_rate=10.0,
        attendance_rate=10.0,
        is_approved=False,
    )

    # Sales Performance Record (Isolation)
    p_sales = PerformanceRecord(
        employee_id="EMP-SALES-SARAH",
        period="2026-Q3",
        overall_score=50.0,
        task_completion_rate=50.0,
        goal_achievement_rate=50.0,
        attendance_rate=50.0,
        is_approved=True,
    )

    # 2. Tasks
    t_alice_blocked = TaskOutcome(
        employee_id="EMP-ENG-ALICE",
        title="Async Queue Cutover",
        status="blocked",
        outcome="Blocked pending IAM security approval",
        period="2026-Q3",
        is_approved=True,
    )
    t_bob_blocked = TaskOutcome(
        employee_id="EMP-ENG-BOB",
        title="Database failover test",
        status="blocked",
        outcome="Blocked on storage provisioning",
        period="2026-Q3",
        is_approved=True,
    )
    t_unapproved_blocked = TaskOutcome(
        employee_id="EMP-ENG-CHARLIE",
        title="Secret project cutover",
        status="blocked",
        outcome="Unapproved blocker",
        period="2026-Q3",
        is_approved=False,
    )
    t_sales_blocked = TaskOutcome(
        employee_id="EMP-SALES-SARAH",
        title="Sales contract signature",
        status="blocked",
        outcome="Blocked on legal",
        period="2026-Q3",
        is_approved=True,
    )

    # 3. Goals
    g_bob_delayed = Goal(
        employee_id="EMP-ENG-BOB",
        title="Upgrade Kubernetes clusters",
        status="delayed",
        progress=45.0,
        period="2026-Q3",
        is_approved=True,
    )
    g_charlie_delayed = Goal(
        employee_id="EMP-ENG-CHARLIE",
        title="Automate integration tests",
        status="delayed",
        progress=30.0,
        period="2026-Q3",
        is_approved=True,
    )
    g_sales_delayed = Goal(
        employee_id="EMP-SALES-SARAH",
        title="Close enterprise deals",
        status="delayed",
        progress=20.0,
        period="2026-Q3",
        is_approved=True,
    )

    # 4. Skills
    s_alice1 = Skill(
        employee_id="EMP-ENG-ALICE",
        name="Python Backend",
        level="Expert",
        is_approved=True,
    )
    s_bob1 = Skill(
        employee_id="EMP-ENG-BOB",
        name="Python Backend",
        level="Intermediate",
        is_approved=True,
    )
    s_bob2 = Skill(
        employee_id="EMP-ENG-BOB",
        name="Kubernetes",
        level="Expert",
        is_approved=True,
    )
    s_sales = Skill(
        employee_id="EMP-SALES-SARAH",
        name="Enterprise Sales",
        level="Expert",
        is_approved=True,
    )

    # 5. Evaluation Themes
    th_alice_pos = EvaluationTheme(
        employee_id="EMP-ENG-ALICE",
        theme="Technical Leadership",
        sentiment="positive",
        evidence="Exemplary leadership on architecture.",
        period="2026-Q3",
        is_approved=True,
    )
    th_bob_needs_imp = EvaluationTheme(
        employee_id="EMP-ENG-BOB",
        theme="Documentation",
        sentiment="needs_improvement",
        evidence="Runbooks lack sequence diagrams.",
        period="2026-Q3",
        is_approved=True,
    )
    th_sales = EvaluationTheme(
        employee_id="EMP-SALES-SARAH",
        theme="Cold Calling",
        sentiment="positive",
        evidence="Sales performance.",
        period="2026-Q3",
        is_approved=True,
    )

    db_session.add_all([
        p_alice_q2, p_alice_q3,
        p_bob_q2, p_bob_q3,
        p_charlie_q3, p_unapproved, p_sales,
        t_alice_blocked, t_bob_blocked, t_unapproved_blocked, t_sales_blocked,
        g_bob_delayed, g_charlie_delayed, g_sales_delayed,
        s_alice1, s_bob1, s_bob2, s_sales,
        th_alice_pos, th_bob_needs_imp, th_sales,
    ])
    db_session.commit()


def _get_mock_ai_service(override_output: dict | None = None) -> TeamInsightAIService:
    """Creates a mock TeamInsightAIService returning grounded JSON."""
    default_output = {
        "executive_summary": (
            "The Engineering team showed an improved trajectory in 2026-Q3, achieving an average overall "
            "score of 85.0 across 3 evaluated members. Delivery velocity remains strong with task completion at 90.0, "
            "while 2 blocked tasks and 2 delayed goals require managerial coordination."
        ),
        "overdue_workload_summary": (
            "Operational blockers impacted 3 team members, with 2 blocked tasks and 2 delayed goals noted. "
            "Key dependencies involve IAM permissions and storage provisioning."
        ),
        "completion_trends_summary": (
            "Task completion averaged 90.0 and goal achievement averaged 90.0. "
            "The period-over-period direction improved relative to the comparison period."
        ),
        "skill_gap_summary": (
            "The primary skill identified across the team is Python Backend, with Kubernetes also representing "
            "a key operational competency."
        ),
        "top_common_gaps": ["Python Backend", "Kubernetes"],
        "evaluation_theme_summary": (
            "Feedback highlighted Technical Leadership as a primary strength, while Documentation was noted "
            "as a key growth area for runbook maintenance."
        ),
        "top_positive_themes": ["Technical Leadership"],
        "top_needs_improvement_themes": ["Documentation"],
        "drill_down_factors": [
            {
                "category": "workload_blockers",
                "factor_title": "Async Queue Cutover Blocker",
                "observation": "Task blocked pending IAM security approval.",
                "supporting_metrics": "Status: blocked, Period: 2026-Q3",
                "anonymized_role": "Lead Software Engineer",
            }
        ],
        "recommended_management_actions": [
            {
                "action_title": "Escalate Security Approvals",
                "description": "Engage cross-functional IAM security leads to expedite outstanding queue cutover reviews.",
                "priority": PriorityLevel.HIGH.value,
            }
        ],
    }
    payload = override_output if override_output is not None else default_output

    choice = MagicMock()
    choice.message.content = json.dumps(payload)
    mock_resp = MagicMock()
    mock_resp.choices = [choice]

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_resp
    return TeamInsightAIService(api_key="mock_key", client=mock_client)


# =============================================================================
# API TESTS
# =============================================================================


def test_successful_valid_request(client: TestClient, seed_department_data):
    """1. Successful valid request returns HTTP 200 and TeamInsightSuccessResponse."""
    app.dependency_overrides[get_team_insight_ai_service] = lambda: _get_mock_ai_service()

    resp = client.post("/api/team-insight", json={"department": "Engineering", "period": "2026-Q3"})
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "success"
    assert data["department"] == "Engineering"
    assert data["period"] == "2026-Q3"
    assert data["team_size"] == 3
    assert "team_findings" in data
    assert "executive_summary" in data["team_findings"]
    assert "overdue_workload" in data["team_findings"]
    assert "completion_trends" in data["team_findings"]
    assert "skill_gap_patterns" in data["team_findings"]
    assert "evaluation_theme_patterns" in data["team_findings"]
    assert len(data["drill_down_factors"]) >= 1
    assert len(data["recommended_management_actions"]) >= 1
    assert "advisory_disclaimer" in data


def test_successful_request_with_explicit_period(client: TestClient, seed_department_data):
    """2. Explicit period is respected and passed to context/response."""
    app.dependency_overrides[get_team_insight_ai_service] = lambda: _get_mock_ai_service()

    resp = client.post("/api/team-insight", json={"department": "Engineering", "period": "2026-Q3"})
    assert resp.status_code == 200
    assert resp.json()["period"] == "2026-Q3"


def test_successful_request_with_period_omitted(client: TestClient, seed_department_data):
    """3. When period is omitted, latest approved period (2026-Q3) is auto-selected."""
    app.dependency_overrides[get_team_insight_ai_service] = lambda: _get_mock_ai_service()

    resp = client.post("/api/team-insight", json={"department": "Engineering"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["period"] == "2026-Q3"


def test_unknown_department_insufficient_data(hr_admin_client: TestClient, seed_department_data):
    """4. Unknown department returns HTTP 200 with status='insufficient_data'."""
    mock_ai = MagicMock()
    app.dependency_overrides[get_team_insight_ai_service] = lambda: mock_ai

    resp = hr_admin_client.post("/api/team-insight", json={"department": "NonexistentDept", "period": "2026-Q3"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "insufficient_data"
    assert "employees" in data["missing_categories"]
    assert "No employees found" in data["message"]
    # AI service must not be called
    mock_ai.generate_team_insight.assert_not_called()


def test_department_with_no_approved_records_insufficient_data(hr_admin_client: TestClient, seed_department_data):
    """5. Department with employees but no approved records returns insufficient_data."""
    mock_ai = MagicMock()
    app.dependency_overrides[get_team_insight_ai_service] = lambda: mock_ai

    resp = hr_admin_client.post("/api/team-insight", json={"department": "Human Resources", "period": "2026-Q3"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "insufficient_data"
    assert "No approved performance" in data["message"]
    mock_ai.generate_team_insight.assert_not_called()


def test_department_isolation(client: TestClient, seed_department_data):
    """6. Records from outside the department (Sales) are excluded."""
    app.dependency_overrides[get_team_insight_ai_service] = lambda: _get_mock_ai_service()

    resp = client.post("/api/team-insight", json={"department": "Engineering", "period": "2026-Q3"})
    assert resp.status_code == 200
    text = json.dumps(resp.json())
    assert "Enterprise Sales" not in text
    assert "Cold Calling" not in text
    assert "Sales contract signature" not in text


def test_approved_only_behavior(client: TestClient, seed_department_data):
    """7. Unapproved tasks, goals, and records are excluded."""
    app.dependency_overrides[get_team_insight_ai_service] = lambda: _get_mock_ai_service()

    resp = client.post("/api/team-insight", json={"department": "Engineering", "period": "2026-Q3"})
    assert resp.status_code == 200
    data = resp.json()
    # Unapproved blocked task 'Secret project cutover' must not be counted in total_blocked_tasks
    assert data["team_findings"]["overdue_workload"]["total_blocked_tasks"] == 2
    assert "Secret project cutover" not in json.dumps(data)


def test_response_contains_team_size_and_findings(client: TestClient, seed_department_data):
    """8. Response contains team_size and all team-level finding dimensions."""
    app.dependency_overrides[get_team_insight_ai_service] = lambda: _get_mock_ai_service()

    resp = client.post("/api/team-insight", json={"department": "Engineering", "period": "2026-Q3"})
    assert resp.status_code == 200
    data = resp.json()

    assert data["team_size"] == 3
    tf = data["team_findings"]
    assert tf["overdue_workload"]["total_blocked_tasks"] == 2
    assert tf["overdue_workload"]["total_delayed_goals"] == 2
    assert tf["overdue_workload"]["affected_member_count"] == 3
    assert tf["completion_trends"]["team_avg_overall_score"] == 85.0
    assert tf["completion_trends"]["team_avg_task_completion"] == 90.0
    assert tf["completion_trends"]["team_avg_goal_achievement"] == 90.0
    assert tf["completion_trends"]["direction"] == TrendDirection.IMPROVED.value


def test_response_does_not_expose_employee_names(client: TestClient, seed_department_data):
    """9. Employee personal names must not be exposed in the response payload."""
    app.dependency_overrides[get_team_insight_ai_service] = lambda: _get_mock_ai_service()

    resp = client.post("/api/team-insight", json={"department": "Engineering", "period": "2026-Q3"})
    assert resp.status_code == 200
    payload_str = json.dumps(resp.json())

    # Ensure individual employee names are not present
    for name in ["Alice Smith", "Bob Jones", "Charlie Brown", "Sarah Miller"]:
        assert name not in payload_str


def test_response_does_not_expose_employee_ids(client: TestClient, seed_department_data):
    """10. Employee IDs must not appear anywhere in the response payload."""
    app.dependency_overrides[get_team_insight_ai_service] = lambda: _get_mock_ai_service()

    resp = client.post("/api/team-insight", json={"department": "Engineering", "period": "2026-Q3"})
    assert resp.status_code == 200
    payload_str = json.dumps(resp.json())

    assert re.search(r"\bEMP-[A-Z0-9-]+\b", payload_str) is None


def test_response_does_not_expose_individual_rankings(client: TestClient, seed_department_data):
    """11. Drill-down factors must be role-anonymized and not contain rankings."""
    app.dependency_overrides[get_team_insight_ai_service] = lambda: _get_mock_ai_service()

    resp = client.post("/api/team-insight", json={"department": "Engineering", "period": "2026-Q3"})
    assert resp.status_code == 200
    data = resp.json()

    for factor in data["drill_down_factors"]:
        assert factor.get("anonymized_role") in [
            "Lead Software Engineer",
            "Senior DevOps Engineer",
            "Software Engineer",
            None,
        ]
        assert re.search(r"\b(rank|top performer|worst performer)\b", json.dumps(factor), re.IGNORECASE) is None


def test_validation_error_for_missing_department(client: TestClient):
    """12. Missing required 'department' field triggers HTTP 422."""
    resp = client.post("/api/team-insight", json={"period": "2026-Q3"})
    assert resp.status_code == 422
    assert "department" in resp.text


def test_validation_error_for_invalid_request_fields(client: TestClient):
    """13. Empty department string triggers HTTP 422."""
    resp = client.post("/api/team-insight", json={"department": "", "period": "2026-Q3"})
    assert resp.status_code == 422


def test_extra_forbidden_request_fields(client: TestClient):
    """14. Extra unpermitted request fields trigger HTTP 422."""
    resp = client.post(
        "/api/team-insight",
        json={"department": "Engineering", "manager_id": "MGR-001"},
    )
    assert resp.status_code == 422
    assert "extra_forbidden" in resp.text or "Extra inputs are not permitted" in resp.text


def test_ai_grounding_failure_returns_502(client: TestClient, seed_department_data):
    """15. Grounding failures in AI service return 502 with safe opaque reference ID."""
    mock_ai = MagicMock()
    mock_ai.generate_team_insight.side_effect = TeamInsightAIServiceError(
        "Grounding validation failure: numeric value '999.0' in narrative is not supported by context."
    )
    app.dependency_overrides[get_team_insight_ai_service] = lambda: mock_ai

    resp = client.post("/api/team-insight", json={"department": "Engineering", "period": "2026-Q3"})
    assert resp.status_code == 502
    data = resp.json()
    assert "AI service temporarily unavailable" in data["detail"]
    assert "Reference ID:" in data["detail"]
    # Ensure internal grounding error message is NOT exposed to caller
    assert "999.0" not in data["detail"]


def test_ai_provider_failure_returns_502(client: TestClient, seed_department_data):
    """16. Provider communication failures return 502 with safe opaque reference ID."""
    mock_ai = MagicMock()
    mock_ai.generate_team_insight.side_effect = TeamInsightAIServiceError(
        "AI provider failed after 3 attempts. Last error: Connection refused to api.groq.com"
    )
    app.dependency_overrides[get_team_insight_ai_service] = lambda: mock_ai

    resp = client.post("/api/team-insight", json={"department": "Engineering", "period": "2026-Q3"})
    assert resp.status_code == 502
    data = resp.json()
    assert "AI service temporarily unavailable" in data["detail"]
    assert "Reference ID:" in data["detail"]
    # Ensure raw provider URL / error details are NOT exposed
    assert "api.groq.com" not in data["detail"]


def test_insufficient_data_path_does_not_call_ai(hr_admin_client: TestClient, seed_department_data):
    """17. Insufficient-data path short-circuits before calling AI service."""
    mock_ai = MagicMock()
    app.dependency_overrides[get_team_insight_ai_service] = lambda: mock_ai

    resp = hr_admin_client.post("/api/team-insight", json={"department": "UnknownTeam"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "insufficient_data"
    mock_ai.generate_team_insight.assert_not_called()


def test_advisory_disclaimer_enforced(client: TestClient, seed_department_data):
    """18. Success response enforces advisory disclaimer."""
    app.dependency_overrides[get_team_insight_ai_service] = lambda: _get_mock_ai_service()

    resp = client.post("/api/team-insight", json={"department": "Engineering", "period": "2026-Q3"})
    assert resp.status_code == 200
    data = resp.json()
    assert "advisory_disclaimer" in data
    assert "AI-assisted advisory analysis" in data["advisory_disclaimer"]

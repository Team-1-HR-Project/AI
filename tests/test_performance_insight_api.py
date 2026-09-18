"""API integration tests for Performance Insight Generator endpoint (POST /api/performance-insight).

Covers:
- Successful request and verified response structure
- Employee with sufficient approved data across periods
- Employee with insufficient data (zero records, single record)
- Nonexistent employee handling
- Unapproved records exclusion and isolation against data leakage
- AI service failure (502 Bad Gateway with safe reference ID)
- Malformed/invalid request validation (HTTP 422)
- Verifying the AI service receives only sanitized context (no raw DB models)
- Verifying insufficient-data requests do NOT call the AI service
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.performance_insight import get_performance_insight_ai_service
from app.db.session import Base, get_db
from app.main import app
from app.models import Employee, PerformanceRecord
from app.schemas.career_coach import PriorityLevel
from app.schemas.performance_insight import (
    AIInterpretation,
    CalculatedTrends,
    ContributingIndicator,
    MetricTrend,
    PerformanceImprovementItem,
    PerformanceInsightSuccessResponse,
    PerformancePeriodMetrics,
    ReviewActionItem,
    TrendDirection,
    VerifiedFacts,
)
from app.services.performance_insight_ai import (
    PerformanceInsightAIService,
    PerformanceInsightAIServiceError,
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
    """Provides an isolated database session for each test."""
    Base.metadata.create_all(bind=test_engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=test_engine)


@pytest.fixture(scope="function")
def client(db_session):
    """Provides a TestClient with overridden get_db dependency."""
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
def mock_ai_service():
    """Provides a mock PerformanceInsightAIService."""
    service = MagicMock(spec=PerformanceInsightAIService)
    app.dependency_overrides[get_performance_insight_ai_service] = lambda: service
    yield service
    if get_performance_insight_ai_service in app.dependency_overrides:
        del app.dependency_overrides[get_performance_insight_ai_service]


@pytest.fixture
def seed_data(db_session):
    """Seeds test employees and performance records."""
    emp1 = Employee(
        id="EMP-API-01",
        first_name="Diana",
        last_name="Prince",
        role_title="Lead Architect",
        department="Engineering",
    )
    emp2 = Employee(
        id="EMP-API-02",
        first_name="Bruce",
        last_name="Wayne",
        role_title="Security Lead",
        department="Security",
    )
    emp_empty = Employee(
        id="EMP-EMPTY",
        first_name="Barry",
        last_name="Allen",
        role_title="Research Associate",
        department="Research",
    )
    emp_single = Employee(
        id="EMP-SINGLE",
        first_name="Arthur",
        last_name="Curry",
        role_title="Ocean Specialist",
        department="Operations",
    )
    db_session.add_all([emp1, emp2, emp_empty, emp_single])
    db_session.commit()

    # Emp1: Two approved quarters (Q1, Q2) and one unapproved draft (Q3)
    p1 = PerformanceRecord(
        employee_id=emp1.id,
        period="2026-Q1",
        overall_score=82.0,
        task_completion_rate=85.0,
        goal_achievement_rate=80.0,
        attendance_rate=96.0,
        is_approved=True,
    )
    p2 = PerformanceRecord(
        employee_id=emp1.id,
        period="2026-Q2",
        overall_score=88.0,
        task_completion_rate=92.0,
        goal_achievement_rate=86.0,
        attendance_rate=94.0,
        is_approved=True,
    )
    p3_unapproved = PerformanceRecord(
        employee_id=emp1.id,
        period="2026-Q3",
        overall_score=99.0,
        task_completion_rate=100.0,
        goal_achievement_rate=100.0,
        attendance_rate=100.0,
        is_approved=False,
    )

    # Emp2: Distinct approved record
    p_emp2 = PerformanceRecord(
        employee_id=emp2.id,
        period="2026-Q2",
        overall_score=75.0,
        task_completion_rate=78.0,
        goal_achievement_rate=72.0,
        attendance_rate=90.0,
        is_approved=True,
    )

    # Emp single: only 1 approved record
    p_single = PerformanceRecord(
        employee_id=emp_single.id,
        period="2026-Q1",
        overall_score=85.0,
        task_completion_rate=88.0,
        goal_achievement_rate=85.0,
        attendance_rate=95.0,
        is_approved=True,
    )

    db_session.add_all([p1, p2, p3_unapproved, p_emp2, p_single])
    db_session.commit()
    return emp1, emp2, emp_empty, emp_single


def _create_mock_success_response(employee_id: str) -> PerformanceInsightSuccessResponse:
    """Creates a valid PerformanceInsightSuccessResponse object."""
    return PerformanceInsightSuccessResponse(
        status="success",
        employee_id=employee_id,
        verified_facts=VerifiedFacts(
            target_period="2026-Q2",
            comparison_period="2026-Q1",
            metrics_by_period=[
                PerformancePeriodMetrics(
                    period="2026-Q1",
                    overall_score=82.0,
                    task_completion_rate=85.0,
                    goal_achievement_rate=80.0,
                    attendance_rate=96.0,
                ),
                PerformancePeriodMetrics(
                    period="2026-Q2",
                    overall_score=88.0,
                    task_completion_rate=92.0,
                    goal_achievement_rate=86.0,
                    attendance_rate=94.0,
                ),
            ],
        ),
        calculated_trends=CalculatedTrends(
            from_period="2026-Q1",
            to_period="2026-Q2",
            metrics={
                "overall_score": MetricTrend(
                    metric_name="overall_score",
                    previous_value=82.0,
                    current_value=88.0,
                    delta=6.0,
                    direction=TrendDirection.IMPROVED,
                    percent_change=7.32,
                )
            },
            improved_metrics=["overall_score"],
            declined_metrics=[],
            stable_metrics=[],
        ),
        ai_interpretation=AIInterpretation(
            summary="Strong progression observed across delivery throughput metrics.",
            improvements=[
                PerformanceImprovementItem(
                    metric="overall_score",
                    summary="Overall score improved by 6 points from Q1 to Q2.",
                    contributing_indicators=[
                        ContributingIndicator(
                            indicator_name="Sprint execution cadence",
                            category="tasks",
                            observation="Observed steady reduction in sprint rollover items to review.",
                        )
                    ],
                )
            ],
            declines=[],
        ),
        suggested_review_actions=[
            ReviewActionItem(
                priority=PriorityLevel.MEDIUM,
                focus_area="Architecture Reviews",
                recommended_action="Schedule bi-weekly architecture checkpoint meetings.",
                rationale="High delivery execution supports taking lead on system designs.",
            )
        ],
        created_at=datetime.now(timezone.utc),
    )


# 1. Successful Request & Correct Response Structure
def test_performance_insight_success(client, seed_data, mock_ai_service):
    emp1, _, _, _ = seed_data
    mock_ai_service.generate_insight_from_context.return_value = _create_mock_success_response(emp1.id)

    response = client.post(
        "/api/performance-insight",
        json={"employee_id": emp1.id},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["employee_id"] == emp1.id
    assert "verified_facts" in data
    assert "calculated_trends" in data
    assert "ai_interpretation" in data
    assert "suggested_review_actions" in data
    assert len(data["verified_facts"]["metrics_by_period"]) == 2
    assert data["calculated_trends"]["from_period"] == "2026-Q1"
    assert data["calculated_trends"]["to_period"] == "2026-Q2"

    # Verify AI service was called with sanitized context (not raw models)
    mock_ai_service.generate_insight_from_context.assert_called_once()
    call_kwargs = mock_ai_service.generate_insight_from_context.call_args.kwargs
    context_passed = call_kwargs["context"]

    assert isinstance(context_passed, dict)
    assert not isinstance(context_passed.get("employee"), Employee)
    assert context_passed["employee"]["id"] == emp1.id
    assert context_passed["target_period"] == "2026-Q2"
    assert context_passed["comparison_period"] == "2026-Q1"


# 2. Target Period Support
def test_performance_insight_with_target_period(client, seed_data, mock_ai_service):
    emp1, _, _, _ = seed_data
    mock_ai_service.generate_insight_from_context.return_value = _create_mock_success_response(emp1.id)

    response = client.post(
        "/api/performance-insight",
        json={"employee_id": emp1.id, "period": "2026-Q2"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"

    call_kwargs = mock_ai_service.generate_insight_from_context.call_args.kwargs
    context_passed = call_kwargs["context"]
    assert context_passed["target_period"] == "2026-Q2"


# 3. Employee with Insufficient Data: Zero Records
def test_performance_insight_zero_records_no_ai_call(client, seed_data, mock_ai_service):
    _, _, emp_empty, _ = seed_data

    response = client.post(
        "/api/performance-insight",
        json={"employee_id": emp_empty.id},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "insufficient_data"
    assert data["employee_id"] == emp_empty.id
    assert "No approved performance records found" in data["reason"]

    # AI service must NOT be called
    mock_ai_service.generate_insight_from_context.assert_not_called()


# 4. Employee with Insufficient Data: Single Period
def test_performance_insight_single_period_no_ai_call(client, seed_data, mock_ai_service):
    _, _, _, emp_single = seed_data

    response = client.post(
        "/api/performance-insight",
        json={"employee_id": emp_single.id},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "insufficient_data"
    assert data["employee_id"] == emp_single.id
    assert "Only one approved performance period available" in data["reason"]

    # AI service must NOT be called
    mock_ai_service.generate_insight_from_context.assert_not_called()


# 5. Nonexistent Employee Handling
def test_performance_insight_nonexistent_employee(client, mock_ai_service):
    response = client.post(
        "/api/performance-insight",
        json={"employee_id": "EMP-DOES-NOT-EXIST"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "insufficient_data"
    assert data["employee_id"] == "EMP-DOES-NOT-EXIST"
    assert "not found" in data["reason"]

    mock_ai_service.generate_insight_from_context.assert_not_called()


# 6. Unapproved Records Are Not Exposed & Cross-Employee Isolation
def test_performance_insight_unapproved_records_excluded_and_isolated(client, seed_data, mock_ai_service):
    emp1, _emp2, _, _ = seed_data
    mock_ai_service.generate_insight_from_context.return_value = _create_mock_success_response(emp1.id)

    response = client.post(
        "/api/performance-insight",
        json={"employee_id": emp1.id},
    )

    assert response.status_code == 200

    # Inspect context passed to AI service
    call_kwargs = mock_ai_service.generate_insight_from_context.call_args.kwargs
    context_passed = call_kwargs["context"]

    periods_in_context = context_passed["facts"]["periods"]
    # 2026-Q3 is unapproved, so it must NOT appear
    assert "2026-Q3" not in periods_in_context
    assert periods_in_context == ["2026-Q1", "2026-Q2"]

    # Emp2's data must NOT leak into emp1's context
    for m in context_passed["facts"]["metrics_by_period"]:
        assert m["overall_score"] != 75.0


# 7. AI Service Failure (502 Bad Gateway with Safe Reference ID)
def test_performance_insight_ai_service_failure_returns_502(client, seed_data, mock_ai_service):
    emp1, _, _, _ = seed_data
    mock_ai_service.generate_insight_from_context.side_effect = PerformanceInsightAIServiceError(
        "Groq rate limit exceeded."
    )

    response = client.post(
        "/api/performance-insight",
        json={"employee_id": emp1.id},
    )

    assert response.status_code == 502
    data = response.json()
    assert "AI service temporarily unavailable" in data["detail"]
    assert "Reference ID:" in data["detail"]
    # Verify no raw provider or exception tracebacks leaked
    assert "Groq" not in data["detail"]


# 8. Malformed / Invalid Requests (HTTP 422)
def test_performance_insight_missing_employee_id(client):
    response = client.post(
        "/api/performance-insight",
        json={},
    )
    assert response.status_code == 422


def test_performance_insight_empty_employee_id(client):
    response = client.post(
        "/api/performance-insight",
        json={"employee_id": ""},
    )
    assert response.status_code == 422


def test_performance_insight_unexpected_field_forbidden(client):
    response = client.post(
        "/api/performance-insight",
        json={"employee_id": "EMP-01", "malicious_field": "injection"},
    )
    assert response.status_code == 422

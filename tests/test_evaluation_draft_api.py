"""API integration tests for Evaluation Draft Assistant endpoint (POST /api/evaluation-draft)."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.evaluation_draft import get_evaluation_draft_ai_service
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
from app.schemas.career_coach import EvidenceItem, PriorityLevel
from app.schemas.evaluation_draft import (
    EvaluationDraftSuccessResponse,
    EvaluationImprovementItem,
    EvaluationStrengthItem,
)
from app.services.evaluation_draft_ai import EvaluationDraftAIService

TEST_DATABASE_URL = "sqlite:///:memory:"
test_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture
def db_session():
    Base.metadata.create_all(bind=test_engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=test_engine)


@pytest.fixture
def client(db_session):
    db_session.add(Employee(
        id="EMP-TEST-CALLER",
        first_name="Test",
        last_name="Caller",
        role_title="HR Administrator",
        department="Human Resources",
    ))
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
            "X-Caller-Employee-ID": "EMP-TEST-CALLER",
            "X-Caller-Role": "hr_admin",
        },
    ) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def mock_ai_service():
    service = MagicMock(spec=EvaluationDraftAIService)
    return service


@pytest.fixture
def seed_data(db_session):
    emp = Employee(
        id="EMP-API-EVAL-01",
        first_name="Diana",
        last_name="Prince",
        role_title="Senior Principal Engineer",
        department="Engineering",
    )
    db_session.add(emp)
    db_session.commit()

    p1 = PerformanceRecord(
        employee_id=emp.id,
        period="2026-Q3",
        overall_score=94.0,
        task_completion_rate=97.0,
        goal_achievement_rate=93.0,
        attendance_rate=99.0,
        is_approved=True,
    )
    g1 = Goal(
        employee_id=emp.id,
        title="Scalable Storage Migration",
        progress=100.0,
        status="completed",
        period="2026-Q3",
        is_approved=True,
    )
    s1 = Skill(
        employee_id=emp.id,
        name="Distributed Systems",
        level="Expert",
        evidence="Led storage architecture",
        is_approved=True,
    )
    t1 = TaskOutcome(
        employee_id=emp.id,
        title="Zero-downtime cutover",
        status="completed",
        outcome="Cutover completed with 0 errors",
        period="2026-Q3",
        is_approved=True,
    )
    th1 = EvaluationTheme(
        employee_id=emp.id,
        theme="System Reliability",
        sentiment="positive",
        evidence="Exceeded operational uptime targets",
        period="2026-Q3",
        is_approved=True,
    )
    db_session.add_all([p1, g1, s1, t1, th1])
    db_session.commit()
    return emp


def _create_mock_success_response(employee_id: str, period: str) -> EvaluationDraftSuccessResponse:
    return EvaluationDraftSuccessResponse(
        employee_id=employee_id,
        period=period,
        evaluation_narrative="Diana had an outstanding Q3, achieving 94.0 overall score and zero cutover errors.",
        strengths=[
            EvaluationStrengthItem(
                title="Exceptional Technical Execution",
                description="Consistently achieves high delivery quality.",
                evidence=[
                    EvidenceItem(
                        source_type="performance",
                        source_id=1,
                        claim="Achieved 94.0 overall score in Q3",
                    )
                ],
            )
        ],
        improvement_areas=[
            EvaluationImprovementItem(
                title="Mentorship Sessions",
                description="Conduct monthly design review walk-throughs for new joiners.",
                evidence=[
                    EvidenceItem(
                        source_type="evaluation_theme",
                        source_id=1,
                        claim="Theme highlighted knowledge sharing opportunity",
                    )
                ],
                priority=PriorityLevel.MEDIUM,
            )
        ],
        entered_scores={"overall": 94.0},
        human_review_required=True,
        created_at=datetime.now(timezone.utc),
    )


def test_api_evaluation_draft_success(client, seed_data, mock_ai_service):
    emp = seed_data
    app.dependency_overrides[get_evaluation_draft_ai_service] = lambda: mock_ai_service
    mock_ai_service.generate_draft.return_value = _create_mock_success_response(emp.id, "2026-Q3")

    response = client.post(
        "/api/evaluation-draft",
        json={
            "employee_id": emp.id,
            "period": "2026-Q3",
            "evaluation_scores": {"overall": 94.0},
            "manager_notes": "Exceeded expectations on core cutover.",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["employee_id"] == emp.id
    assert data["period"] == "2026-Q3"
    assert data["human_review_required"] is True
    assert "draft" in data["review_disclaimer"].lower()
    assert len(data["strengths"]) == 1
    assert len(data["improvement_areas"]) == 1

    # Verify AI service received sanitized context
    call_kwargs = mock_ai_service.generate_draft.call_args.kwargs
    context_passed = call_kwargs["context"]
    assert context_passed["employee"]["id"] == emp.id
    assert call_kwargs["period"] == "2026-Q3"
    assert call_kwargs["manager_notes"] == "Exceeded expectations on core cutover."


def test_api_evaluation_draft_stateless_no_db_persistence(client, seed_data, mock_ai_service, db_session):
    emp = seed_data
    app.dependency_overrides[get_evaluation_draft_ai_service] = lambda: mock_ai_service
    mock_ai_service.generate_draft.return_value = _create_mock_success_response(emp.id, "2026-Q3")

    initial_perf_count = db_session.query(PerformanceRecord).count()
    initial_theme_count = db_session.query(EvaluationTheme).count()

    response = client.post(
        "/api/evaluation-draft",
        json={"employee_id": emp.id, "period": "2026-Q3"},
    )
    assert response.status_code == 200

    # Verify no new records were inserted into the database
    assert db_session.query(PerformanceRecord).count() == initial_perf_count
    assert db_session.query(EvaluationTheme).count() == initial_theme_count


def test_api_evaluation_draft_insufficient_data(client, db_session, mock_ai_service):
    # Empty employee with zero records
    emp_empty = Employee(
        id="EMP-EMPTY-01",
        first_name="Empty",
        last_name="User",
        role_title="Junior Engineer",
        department="Engineering",
    )
    db_session.add(emp_empty)
    db_session.commit()

    app.dependency_overrides[get_evaluation_draft_ai_service] = lambda: mock_ai_service

    response = client.post(
        "/api/evaluation-draft",
        json={"employee_id": emp_empty.id, "period": "2026-Q3"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "insufficient_data"
    assert data["employee_id"] == emp_empty.id
    assert data["human_review_required"] is False
    assert "performance" in data["missing_categories"]

    # AI service must not be invoked
    mock_ai_service.generate_draft.assert_not_called()


def test_api_evaluation_draft_nonexistent_employee(client, mock_ai_service):
    app.dependency_overrides[get_evaluation_draft_ai_service] = lambda: mock_ai_service

    response = client.post(
        "/api/evaluation-draft",
        json={"employee_id": "EMP-NONEXISTENT", "period": "2026-Q3"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "insufficient_data"
    mock_ai_service.generate_draft.assert_not_called()


def test_api_evaluation_draft_validation_error(client):
    # Missing period
    response = client.post(
        "/api/evaluation-draft",
        json={"employee_id": "EMP-001"},
    )
    assert response.status_code == 422

    # Out-of-range score
    response_score = client.post(
        "/api/evaluation-draft",
        json={
            "employee_id": "EMP-001",
            "period": "2026-Q3",
            "evaluation_scores": {"overall": 120.0},
        },
    )
    assert response_score.status_code == 422


def test_api_evaluation_draft_ai_error_returns_502(client, seed_data, mock_ai_service):
    emp = seed_data
    app.dependency_overrides[get_evaluation_draft_ai_service] = lambda: mock_ai_service
    mock_ai_service.generate_draft.side_effect = RuntimeError("AI service temporarily unavailable. Reference ID: test-ref-id")

    response = client.post(
        "/api/evaluation-draft",
        json={"employee_id": emp.id, "period": "2026-Q3"},
    )

    assert response.status_code == 502
    data = response.json()
    assert "AI service temporarily unavailable" in data["detail"]
    assert "Reference ID:" in data["detail"]

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.career_coach import get_career_coach_ai_service
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
from app.schemas.career_coach import (
    CareerCoachInsufficientDataResponse,
    CareerCoachSuccessResponse,
)
from app.services.career_coach_ai import (
    CareerCoachAIService,
    CareerCoachAIServiceError,
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
    Base.metadata.create_all(bind=test_engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=test_engine)

@pytest.fixture(scope="function")
def client(db_session):
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
def seed_employee(db_session):
    emp = Employee(
        id="EMP-API-001",
        first_name="Diana",
        last_name="Prince",
        role_title="Lead Architect",
        department="Engineering",
    )
    db_session.add(emp)
    db_session.commit()

    db_session.add(PerformanceRecord(
        employee_id="EMP-API-001",
        period="2026-Q3",
        overall_score=95.0,
        task_completion_rate=98.0,
        goal_achievement_rate=94.0,
        attendance_rate=99.0,
    ))
    db_session.add(Goal(
        employee_id="EMP-API-001",
        title="Deploy resilient cloud topology",
        progress=90.0,
        status="in_progress",
        period="2026-Q3",
    ))
    db_session.add(Skill(
        employee_id="EMP-API-001",
        name="Distributed Systems",
        level="Expert",
        evidence="Delivered multi-region failover architecture",
    ))
    db_session.add(TaskOutcome(
        employee_id="EMP-API-001",
        title="Zero-downtime database cutover",
        status="completed",
        outcome="100% data integrity with zero downtime",
        period="2026-Q3",
    ))
    db_session.add(EvaluationTheme(
        employee_id="EMP-API-001",
        theme="Systemic Thinking",
        sentiment="positive",
        evidence="Exemplary foresight during infrastructure planning",
        period="2026-Q3",
    ))
    db_session.commit()
    return emp

MOCK_SUCCESS_PLAN = CareerCoachSuccessResponse(
    status="success",
    employee_id="EMP-API-001",
    strengths=[
        {
            "title": "Architectural Mastery",
            "description": "Consistently designs and deploys resilient systems.",
            "evidence": [
                {
                    "source_type": "performance",
                    "source_id": 1,
                    "claim": "98% task completion rate"
                },
                {
                    "source_type": "skill",
                    "source_id": 1,
                    "claim": "Delivered multi-region failover"
                }
            ]
        }
    ],
    development_areas=[
        {
            "title": "Technical Blogging & Advocacy",
            "description": "Document architectural insights externally.",
            "evidence": [
                {
                    "source_type": "evaluation_theme",
                    "source_id": 1,
                    "claim": "Evaluation notes opportunity for wider thought leadership"
                }
            ],
            "priority": "medium"
        }
    ],
    development_plan=[
        {
            "action": "Publish internal engineering whitepaper",
            "reason": "Scale systemic thinking across squads.",
            "measurable_target": "Complete 1 peer-reviewed whitepaper",
            "suggested_timeline": "45 days"
        }
    ],
    follow_up={
        "checkpoint": "End of Q4",
        "review_focus": "Assess adoption of multi-region architecture standards"
    }
)

# 1. Successful Career Coach request
def test_successful_career_coach_api_request(client, seed_employee):
    mock_ai_service = MagicMock(spec=CareerCoachAIService)
    mock_ai_service.generate_career_plan.return_value = MOCK_SUCCESS_PLAN

    app.dependency_overrides[get_career_coach_ai_service] = lambda: mock_ai_service

    response = client.post(
        "/api/career-coach",
        json={"employee_id": "EMP-API-001", "period": "2026-Q3"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["employee_id"] == "EMP-API-001"
    assert len(data["strengths"]) == 1
    assert data["strengths"][0]["title"] == "Architectural Mastery"
    assert len(data["development_plan"]) == 1
    assert data["development_plan"][0]["suggested_timeline"] == "45 days"

    # Verify parameters from request body were correctly passed down
    mock_ai_service.generate_career_plan.assert_called_once()
    call_kwargs = mock_ai_service.generate_career_plan.call_args.kwargs
    assert call_kwargs["employee_id"] == "EMP-API-001"
    assert call_kwargs["period"] == "2026-Q3"


# 2. Insufficient-data response
def test_insufficient_data_api_response(client, db_session):
    # Employee with no records
    emp = Employee(
        id="EMP-SPARSE",
        first_name="Arthur",
        last_name="Curry",
        role_title="Junior Analyst",
        department="Operations",
    )
    db_session.add(emp)
    db_session.commit()

    mock_insufficient_plan = CareerCoachInsufficientDataResponse(
        status="insufficient_data",
        employee_id="EMP-SPARSE",
        missing_categories=["performance", "goals", "skills", "task_outcomes", "evaluation_themes"],
        message="Not enough approved employee data to generate a reliable career coaching plan."
    )

    mock_ai_service = MagicMock(spec=CareerCoachAIService)
    mock_ai_service.generate_career_plan.return_value = mock_insufficient_plan

    app.dependency_overrides[get_career_coach_ai_service] = lambda: mock_ai_service

    response = client.post(
        "/api/career-coach",
        json={"employee_id": "EMP-SPARSE"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "insufficient_data"
    assert data["employee_id"] == "EMP-SPARSE"
    assert "performance" in data["missing_categories"]
    assert "Not enough approved employee data" in data["message"]


# 3. AI service error handled cleanly with HTTP error (no secrets or stack traces leaked)
def test_ai_service_error_handling(client, seed_employee):
    mock_ai_service = MagicMock(spec=CareerCoachAIService)
    mock_ai_service.generate_career_plan.side_effect = CareerCoachAIServiceError(
        "Groq API error encountered (RateLimitError). Unable to complete Career Coach generation."
    )

    app.dependency_overrides[get_career_coach_ai_service] = lambda: mock_ai_service

    response = client.post(
        "/api/career-coach",
        json={"employee_id": "EMP-API-001"},
    )

    assert response.status_code == 502
    data = response.json()
    assert "RateLimitError" in data["detail"]
    # Verify no raw python traceback or internal keys in response
    assert "Traceback" not in response.text
    assert "gsk_" not in response.text


# 4. Request validation: missing employee_id
def test_career_coach_validation_missing_employee_id(client):
    response = client.post(
        "/api/career-coach",
        json={},
    )
    assert response.status_code == 422


# 5. Request validation: extra fields rejected
def test_career_coach_validation_extra_fields_rejected(client):
    response = client.post(
        "/api/career-coach",
        json={"employee_id": "EMP-API-001", "unexpected": "disallowed"},
    )
    assert response.status_code == 422


# 6. OpenAPI contract: no path or query parameters, body has employee_id and period
def test_career_coach_openapi_contract():
    openapi_schema = app.openapi()
    assert "/api/career-coach" in openapi_schema["paths"]
    endpoint_spec = openapi_schema["paths"]["/api/career-coach"]["post"]

    # Verify no path or query parameters
    params = endpoint_spec.get("parameters", [])
    assert len(params) == 0

    # Verify request body schema
    schema_ref = endpoint_spec["requestBody"]["content"]["application/json"]["schema"]["$ref"]
    schema_name = schema_ref.split("/")[-1]
    body_schema = openapi_schema["components"]["schemas"][schema_name]
    assert "employee_id" in body_schema["properties"]
    assert "period" in body_schema["properties"]
    assert body_schema["required"] == ["employee_id"]

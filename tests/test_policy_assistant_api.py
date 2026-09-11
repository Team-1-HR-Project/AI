"""Integration tests for the AI HR Policy Assistant API endpoint."""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.policy_assistant import get_policy_ai_service
from app.db.session import Base, get_db
from app.main import app
from app.schemas.policy_assistant import (
    PolicyAnswerResponse,
    PolicyFallbackResponse,
    PolicyReference,
)
from app.services.policy_ai import (
    PolicyAIService,
    PolicyAIServiceError,
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
def mock_policy_ai_service():
    mock_service = MagicMock(spec=PolicyAIService)
    app.dependency_overrides[get_policy_ai_service] = lambda: mock_service
    yield mock_service
    app.dependency_overrides.pop(get_policy_ai_service, None)


# 1. Successful policy answer test
def test_successful_policy_assistant_api_request(client, mock_policy_ai_service):
    mock_policy_ai_service.answer_policy_question.return_value = PolicyAnswerResponse(
        status="success",
        employee_id="EMP-001",
        answer="Employees may roll over up to 5 days of unused annual leave into the next calendar year.",
        policy_references=[
            PolicyReference(
                policy_id=1,
                policy_code="POL-LEAVE-001",
                title="Annual Leave & Time Off Policy",
                version="1.0",
            )
        ],
        employee_facts_used=["Role: Senior Software Engineer"],
    )

    response = client.post(
        "/api/policy-assistant",
        json={
            "employee_id": "EMP-001",
            "question": "What is the annual leave rollover limit?",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["employee_id"] == "EMP-001"
    assert "roll over up to 5 days" in data["answer"]
    assert len(data["policy_references"]) == 1
    assert data["policy_references"][0]["policy_code"] == "POL-LEAVE-001"
    assert data["policy_references"][0]["policy_id"] == 1
    assert data["employee_facts_used"] == ["Role: Senior Software Engineer"]
    assert "created_at" in data

    # Verify mock was called with employee_id and question from request body
    _args, kwargs = mock_policy_ai_service.answer_policy_question.call_args
    assert kwargs["employee_id"] == "EMP-001"
    assert kwargs["question"] == "What is the annual leave rollover limit?"


# 2. Unsupported question test
def test_unsupported_question_api_response(client, mock_policy_ai_service):
    mock_policy_ai_service.answer_policy_question.return_value = PolicyFallbackResponse(
        status="unsupported",
        employee_id="EMP-001",
        message="No approved company policies cover recipes or cooking.",
    )

    response = client.post(
        "/api/policy-assistant",
        json={
            "employee_id": "EMP-001",
            "question": "Can you share a good recipe?",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "unsupported"
    assert data["employee_id"] == "EMP-001"
    assert "No approved company policies" in data["message"]
    assert "created_at" in data


# 3. Service error maps cleanly to HTTP 502
def test_policy_ai_service_error_handling(client, mock_policy_ai_service):
    mock_policy_ai_service.answer_policy_question.side_effect = PolicyAIServiceError(
        "Provider rate limit reached. Service temporarily unavailable."
    )

    response = client.post(
        "/api/policy-assistant",
        json={
            "employee_id": "EMP-001",
            "question": "What is the leave policy?",
        },
    )

    assert response.status_code == 502
    data = response.json()
    assert "Provider rate limit reached" in data["detail"]


# 4. Request validation failure (question too short)
def test_request_validation_failure_short_question(client):
    response = client.post(
        "/api/policy-assistant",
        json={
            "employee_id": "EMP-001",
            "question": "ab",  # min_length is 3
        },
    )

    assert response.status_code == 422


# 5. Request validation failure (missing request body)
def test_request_validation_failure_missing_body(client):
    response = client.post(
        "/api/policy-assistant",
        json={},
    )

    assert response.status_code == 422


# 6. Request validation failure (missing employee_id)
def test_request_validation_failure_missing_employee_id(client):
    response = client.post(
        "/api/policy-assistant",
        json={
            "question": "What is the leave policy?",
        },
    )

    assert response.status_code == 422


# 7. Endpoint returns the exact response shape
def test_endpoint_returns_correct_response_shape(client, mock_policy_ai_service):
    mock_policy_ai_service.answer_policy_question.return_value = PolicyAnswerResponse(
        status="success",
        employee_id="EMP-001",
        answer="You may work remotely up to 2 days per week.",
        policy_references=[
            PolicyReference(
                policy_id=2,
                policy_code="POL-REMOTE-001",
                title="Hybrid & Remote Work Policy",
                version="1.0",
            )
        ],
        employee_facts_used=["Department: Engineering"],
    )

    response = client.post(
        "/api/policy-assistant",
        json={
            "employee_id": "EMP-001",
            "question": "How many remote days do we get?",
        },
    )

    assert response.status_code == 200
    data = response.json()
    expected_top_keys = {
        "status",
        "employee_id",
        "answer",
        "policy_references",
        "employee_facts_used",
        "created_at",
    }
    assert expected_top_keys.issubset(set(data.keys()))

    expected_ref_keys = {"policy_id", "policy_code", "title", "version"}
    assert expected_ref_keys.issubset(set(data["policy_references"][0].keys()))


# 8. Request body rejects category (extra forbidden)
def test_request_body_rejects_category_field(client):
    response = client.post(
        "/api/policy-assistant",
        json={
            "employee_id": "EMP-001",
            "question": "What is the leave policy?",
            "category": "Leave",
        },
    )
    assert response.status_code == 422


# 9. Request body rejects extra arbitrary fields
def test_request_body_rejects_extra_fields(client):
    response = client.post(
        "/api/policy-assistant",
        json={
            "employee_id": "EMP-001",
            "question": "What is the leave policy?",
            "arbitrary_field": "disallowed",
        },
    )
    assert response.status_code == 422


# 10. Endpoint invokes service with only employee_id and question
def test_endpoint_invokes_service_with_only_employee_id_and_question(client, mock_policy_ai_service):
    mock_policy_ai_service.answer_policy_question.return_value = PolicyFallbackResponse(
        status="unsupported",
        employee_id="EMP-001",
        message="No policies matched.",
    )

    response = client.post(
        "/api/policy-assistant",
        json={
            "employee_id": "EMP-001",
            "question": "What is the leave policy?",
        },
    )

    assert response.status_code == 200
    _args, kwargs = mock_policy_ai_service.answer_policy_question.call_args
    assert kwargs["employee_id"] == "EMP-001"
    assert kwargs["question"] == "What is the leave policy?"
    assert "category" not in kwargs


# 11. OpenAPI schema validation: zero path/query parameters, correct request body
def test_openapi_schema_matches_contract():
    openapi_schema = app.openapi()
    assert "/api/policy-assistant" in openapi_schema["paths"]
    assert "/api/policy-assistant/{employee_id}" not in openapi_schema["paths"]
    endpoint_spec = openapi_schema["paths"]["/api/policy-assistant"]["post"]

    # Verify no parameters in path or query
    params = endpoint_spec.get("parameters", [])
    assert len(params) == 0

    # Verify request body schema has question and employee_id, and not category
    schema_ref = endpoint_spec["requestBody"]["content"]["application/json"]["schema"]["$ref"]
    schema_name = schema_ref.split("/")[-1]
    body_schema = openapi_schema["components"]["schemas"][schema_name]
    assert "question" in body_schema["properties"]
    assert "employee_id" in body_schema["properties"]
    assert "category" not in body_schema["properties"]
    assert set(body_schema["required"]) == {"employee_id", "question"}

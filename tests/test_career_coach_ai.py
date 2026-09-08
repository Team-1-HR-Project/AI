import json
import os
from unittest.mock import MagicMock, patch
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from groq import APIConnectionError

from app.db.session import Base
from app.models import (
    Employee,
    PerformanceRecord,
    Goal,
    Skill,
    TaskOutcome,
    EvaluationTheme,
)
from app.schemas.career_coach import (
    CareerCoachSuccessResponse,
    CareerCoachInsufficientDataResponse,
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
def db():
    Base.metadata.create_all(bind=test_engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=test_engine)

@pytest.fixture
def seed_data(db):
    """Seed two employees: Employee A with full data, Employee B with different department."""
    emp_a = Employee(
        id="EMP-A",
        first_name="Alice",
        last_name="Johnson",
        role_title="Senior Backend Engineer",
        department="Engineering",
    )
    emp_b = Employee(
        id="EMP-B",
        first_name="Bob",
        last_name="Williams",
        role_title="Sales Executive",
        department="Sales",
    )
    db.add_all([emp_a, emp_b])
    db.commit()

    # Employee A complete records
    db.add(PerformanceRecord(
        employee_id="EMP-A",
        period="2026-Q3",
        overall_score=94.0,
        task_completion_rate=96.0,
        goal_achievement_rate=92.0,
        attendance_rate=99.0,
    ))
    db.add(Goal(
        employee_id="EMP-A",
        title="Modernize API Gateway",
        progress=85.0,
        status="in_progress",
        period="2026-Q3",
    ))
    db.add(Skill(
        employee_id="EMP-A",
        name="Python & FastAPI",
        level="Expert",
        evidence="Built high throughput microservices",
    ))
    db.add(TaskOutcome(
        employee_id="EMP-A",
        title="Optimize database indexing",
        status="completed",
        outcome="Latency reduced by 50%",
        period="2026-Q3",
    ))
    db.add(EvaluationTheme(
        employee_id="EMP-A",
        theme="Technical Problem Solving",
        sentiment="positive",
        evidence="Rapid debugging under high traffic",
        period="2026-Q3",
    ))

    db.commit()
    return emp_a, emp_b

MOCK_VALID_GROQ_JSON = json.dumps({
    "status": "success",
    "employee_id": "EMP-A",
    "strengths": [
        {
            "title": "Backend Optimization & Resiliency",
            "description": "Consistently delivers high-efficiency service refactors.",
            "evidence": ["96% task completion rate", "Latency reduced by 50%"]
        }
    ],
    "development_areas": [
        {
            "title": "Cross-Functional System Documentation",
            "description": "Expand architecture documentation for junior peers.",
            "evidence": ["Evaluation theme highlighted opportunities for peer guidance"],
            "priority": "medium"
        }
    ],
    "development_plan": [
        {
            "action": "Author API Gateway RFC Document",
            "reason": "Ensure cross-functional alignment before rollout.",
            "measurable_target": "Publish draft RFC and gather 3 peer reviews",
            "suggested_timeline": "30 days"
        }
    ],
    "follow_up": {
        "checkpoint": "End of Q3 Sprint 4",
        "review_focus": "Review RFC feedback and benchmark latency improvements"
    },
    "created_at": "2026-09-07T12:00:00Z"
})


# 1. Successful Career Coach generation
def test_successful_career_coach_generation(db, seed_data):
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = MOCK_VALID_GROQ_JSON
    mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

    service = CareerCoachAIService(api_key="mock-key", client=mock_client)
    result = service.generate_career_plan(db, employee_id="EMP-A", period="2026-Q3")

    assert isinstance(result, CareerCoachSuccessResponse)
    assert result.status == "success"
    assert result.employee_id == "EMP-A"
    assert len(result.strengths) == 1
    assert result.development_areas[0].priority.value == "medium"
    assert len(result.development_plan) == 1
    assert result.development_plan[0].suggested_timeline == "30 days"
    mock_client.chat.completions.create.assert_called_once()


# 2. Insufficient data: Verify Groq is NOT called and CareerCoachInsufficientDataResponse returned
def test_insufficient_data_skips_groq(db):
    emp_empty = Employee(
        id="EMP-EMPTY",
        first_name="New",
        last_name="Hire",
        role_title="Junior Engineer",
        department="Engineering"
    )
    db.add(emp_empty)
    db.commit()

    mock_client = MagicMock()
    service = CareerCoachAIService(api_key="mock-key", client=mock_client)
    result = service.generate_career_plan(db, employee_id="EMP-EMPTY")

    assert isinstance(result, CareerCoachInsufficientDataResponse)
    assert result.status == "insufficient_data"
    assert result.employee_id == "EMP-EMPTY"
    assert len(result.missing_categories) > 0
    # Crucial: Groq client must NOT be called
    mock_client.chat.completions.create.assert_not_called()


# 3. Invalid AI response: Mock Groq returns invalid JSON/schema -> rejected
def test_invalid_ai_response_rejected(db, seed_data):
    # Missing required fields
    invalid_json = json.dumps({
        "status": "success",
        "employee_id": "EMP-A"
        # Missing strengths, development_areas, development_plan, follow_up
    })

    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = invalid_json
    mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

    service = CareerCoachAIService(api_key="mock-key", client=mock_client)

    with pytest.raises(CareerCoachAIServiceError) as exc_info:
        service.generate_career_plan(db, employee_id="EMP-A", period="2026-Q3")
    assert "schema validation" in str(exc_info.value)


# 4. Sensitive data protection: Verify sent prompt contains only sanitized fields
def test_sensitive_data_protection(db, seed_data):
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = MOCK_VALID_GROQ_JSON
    mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

    service = CareerCoachAIService(api_key="mock-key", client=mock_client)
    service.generate_career_plan(db, employee_id="EMP-A", period="2026-Q3")

    # Inspect the exact payload sent to Groq
    call_args = mock_client.chat.completions.create.call_args[1]
    messages = call_args["messages"]
    user_prompt = messages[1]["content"]
    system_prompt = messages[0]["content"]

    # Verify no sensitive keywords in the user prompt
    forbidden_terms = ["salary", "password", "token", "ssn", "national_id", "bank", "bonus", "phone", "address"]
    for term in forbidden_terms:
        assert f'"{term}"' not in user_prompt.lower()

    # Verify system prompt explicitly prohibits employment decisions
    assert "PROHIBITION ON EMPLOYMENT DECISIONS" in system_prompt


# 5. Employee isolation: Verify only requested employee's data is sent to Groq
def test_employee_isolation(db, seed_data):
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = MOCK_VALID_GROQ_JSON
    mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

    service = CareerCoachAIService(api_key="mock-key", client=mock_client)
    service.generate_career_plan(db, employee_id="EMP-A", period="2026-Q3")

    call_args = mock_client.chat.completions.create.call_args[1]
    user_prompt = call_args["messages"][1]["content"]

    # Must contain EMP-A data and NOT EMP-B data
    assert "EMP-A" in user_prompt
    assert "EMP-B" not in user_prompt
    assert "Bob" not in user_prompt
    assert "Sales" not in user_prompt


# 6. Configuration: Verify GROQ_API_KEY and GROQ_MODEL are read from environment
def test_configuration_from_env():
    with patch.dict(os.environ, {"GROQ_API_KEY": "test-env-key", "GROQ_MODEL": "llama-3.3-70b-versatile"}):
        service = CareerCoachAIService()
        assert service.api_key == "test-env-key"
        assert service.model == "llama-3.3-70b-versatile"

def test_missing_api_key_raises_error(db, seed_data):
    with patch.dict(os.environ, {}, clear=True):
        service = CareerCoachAIService(api_key=None)
        with pytest.raises(CareerCoachAIServiceError) as exc_info:
            service.generate_career_plan(db, employee_id="EMP-A", period="2026-Q3")
        assert "GROQ_API_KEY is not configured" in str(exc_info.value)


# 7. Groq API failure: Mock a Groq API exception and verify it is handled safely
def test_groq_api_failure_handled_safely(db, seed_data):
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = APIConnectionError(request=MagicMock())

    service = CareerCoachAIService(api_key="secret-api-key-12345", client=mock_client)

    with pytest.raises(CareerCoachAIServiceError) as exc_info:
        service.generate_career_plan(db, employee_id="EMP-A", period="2026-Q3")

    # Crucial: Error message must NEVER leak the API key
    assert "secret-api-key-12345" not in str(exc_info.value)
    assert "Groq API error encountered" in str(exc_info.value)

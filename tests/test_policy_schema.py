"""Unit tests for AI HR Policy Assistant Pydantic validation schemas."""

from datetime import datetime, timezone

import pytest
from pydantic import TypeAdapter, ValidationError

from app.schemas.policy_assistant import (
    PolicyAnswerResponse,
    PolicyAssistantResponse,
    PolicyFallbackResponse,
    PolicyQuestionRequest,
    PolicyReference,
)

# --- 1. PolicyQuestionRequest Tests ---


def test_valid_policy_question_request():
    req = PolicyQuestionRequest(
        employee_id="EMP-001",
        question="What is the annual leave rollover limit?",
    )
    assert req.employee_id == "EMP-001"
    assert req.question == "What is the annual leave rollover limit?"


def test_policy_question_request_rejects_category_in_body():
    with pytest.raises(ValidationError) as exc:
        PolicyQuestionRequest(
            employee_id="EMP-001",
            question="What is the annual leave rollover limit?",
            category="Leave & Attendance",
        )
    assert "extra_forbidden" in str(exc.value) or "category" in str(exc.value)


def test_policy_question_request_missing_required_fields():
    # Missing both
    with pytest.raises(ValidationError) as exc:
        PolicyQuestionRequest()
    assert "employee_id" in str(exc.value)

    # Missing question
    with pytest.raises(ValidationError) as exc:
        PolicyQuestionRequest(employee_id="EMP-001")
    assert "question" in str(exc.value)

    # Missing employee_id
    with pytest.raises(ValidationError) as exc:
        PolicyQuestionRequest(question="What is the annual leave rollover limit?")
    assert "employee_id" in str(exc.value)


def test_policy_question_request_empty_strings_rejected():
    with pytest.raises(ValidationError) as exc:
        PolicyQuestionRequest(employee_id="EMP-001", question="")
    assert "question" in str(exc.value)

    with pytest.raises(ValidationError) as exc:
        PolicyQuestionRequest(employee_id="EMP-001", question="ab")
    assert "question" in str(exc.value)

    with pytest.raises(ValidationError) as exc:
        PolicyQuestionRequest(employee_id="", question="Valid question text?")
    assert "employee_id" in str(exc.value)


def test_policy_question_request_max_length_limits():
    # question > 1000
    with pytest.raises(ValidationError) as exc:
        PolicyQuestionRequest(
            employee_id="EMP-001",
            question="Q" * 1001,
        )
    assert "question" in str(exc.value)

    # employee_id > 100
    with pytest.raises(ValidationError) as exc:
        PolicyQuestionRequest(
            employee_id="E" * 101,
            question="Valid question text?",
        )
    assert "employee_id" in str(exc.value)


def test_policy_question_request_rejects_extra_fields():
    # arbitrary unexpected fields rejected
    with pytest.raises(ValidationError) as exc:
        PolicyQuestionRequest(
            employee_id="EMP-001",
            question="Valid question text?",
            unexpected_field="disallowed",
        )
    assert "extra_forbidden" in str(exc.value) or "unexpected_field" in str(exc.value)


# --- 2. PolicyReference Tests ---


def test_valid_policy_reference():
    ref = PolicyReference(
        policy_id=1,
        policy_code="POL-LEAVE-001",
        title="Annual Leave & Time Off Policy",
        version="1.0",
    )
    assert ref.policy_id == 1
    assert ref.policy_code == "POL-LEAVE-001"
    assert ref.title == "Annual Leave & Time Off Policy"
    assert ref.version == "1.0"


def test_policy_reference_invalid_id():
    with pytest.raises(ValidationError) as exc:
        PolicyReference(
            policy_id=0,
            policy_code="POL-LEAVE-001",
            title="Title",
            version="1.0",
        )
    assert "policy_id" in str(exc.value)


def test_policy_reference_rejects_extra_fields():
    with pytest.raises(ValidationError) as exc:
        PolicyReference(
            policy_id=1,
            policy_code="POL-LEAVE-001",
            title="Title",
            version="1.0",
            extra_key="not_allowed",
        )
    assert "extra_forbidden" in str(exc.value) or "extra_key" in str(exc.value)


# --- 3. PolicyAnswerResponse Tests ---


def test_valid_policy_answer_response():
    resp = PolicyAnswerResponse(
        employee_id="EMP-001",
        answer="According to company policy, you may carry forward up to 5 days of unused annual leave.",
        policy_references=[
            PolicyReference(
                policy_id=1,
                policy_code="POL-LEAVE-001",
                title="Annual Leave & Time Off Policy",
                version="1.0",
            )
        ],
        employee_facts_used=["Role: Senior Software Engineer", "Department: Engineering"],
    )
    assert resp.status == "success"
    assert resp.employee_id == "EMP-001"
    assert len(resp.policy_references) == 1
    assert len(resp.employee_facts_used) == 2
    assert isinstance(resp.created_at, datetime)
    assert resp.created_at.tzinfo == timezone.utc


def test_policy_answer_response_empty_references_rejected():
    with pytest.raises(ValidationError) as exc:
        PolicyAnswerResponse(
            employee_id="EMP-001",
            answer="Answer without any cited policy references.",
            policy_references=[],
        )
    assert "policy_references" in str(exc.value)


def test_policy_answer_response_rejects_extra_fields():
    with pytest.raises(ValidationError) as exc:
        PolicyAnswerResponse(
            employee_id="EMP-001",
            answer="Valid answer text.",
            policy_references=[
                PolicyReference(
                    policy_id=1,
                    policy_code="POL-LEAVE-001",
                    title="Title",
                    version="1.0",
                )
            ],
            unknown_property="forbidden",
        )
    assert "extra_forbidden" in str(exc.value) or "unknown_property" in str(exc.value)


# --- 4. PolicyFallbackResponse Tests ---


def test_valid_policy_fallback_response():
    resp = PolicyFallbackResponse(
        employee_id="EMP-001",
        message="I cannot answer this question because it is not covered by approved company policies.",
    )
    assert resp.status == "unsupported"
    assert resp.employee_id == "EMP-001"
    assert "not covered" in resp.message
    assert isinstance(resp.created_at, datetime)
    assert resp.created_at.tzinfo == timezone.utc


def test_policy_fallback_response_rejects_extra_fields():
    with pytest.raises(ValidationError) as exc:
        PolicyFallbackResponse(
            employee_id="EMP-001",
            message="Valid message text.",
            injected_field="not_allowed",
        )
    assert "extra_forbidden" in str(exc.value) or "injected_field" in str(exc.value)


# --- 5. Discriminated Union (PolicyAssistantResponse) Tests ---


def test_discriminated_union_success_branch():
    adapter = TypeAdapter(PolicyAssistantResponse)
    data = {
        "status": "success",
        "employee_id": "EMP-001",
        "answer": "Employees may work remotely up to 2 days per week upon supervisor approval.",
        "policy_references": [
            {
                "policy_id": 3,
                "policy_code": "POL-REMOTE-001",
                "title": "Hybrid & Remote Work Policy",
                "version": "1.0",
            }
        ],
        "employee_facts_used": [],
    }
    parsed = adapter.validate_python(data)
    assert isinstance(parsed, PolicyAnswerResponse)
    assert parsed.status == "success"
    assert parsed.policy_references[0].policy_code == "POL-REMOTE-001"


def test_discriminated_union_unsupported_branch():
    adapter = TypeAdapter(PolicyAssistantResponse)
    data = {
        "status": "unsupported",
        "employee_id": "EMP-001",
        "message": "This question is out of scope for HR policies.",
    }
    parsed = adapter.validate_python(data)
    assert isinstance(parsed, PolicyFallbackResponse)
    assert parsed.status == "unsupported"
    assert parsed.message == "This question is out of scope for HR policies."


def test_discriminated_union_invalid_status_rejected():
    adapter = TypeAdapter(PolicyAssistantResponse)
    data = {
        "status": "unknown_status",
        "employee_id": "EMP-001",
        "message": "Some message.",
    }
    with pytest.raises(ValidationError) as exc:
        adapter.validate_python(data)
    assert "status" in str(exc.value) or "discriminator" in str(exc.value).lower()

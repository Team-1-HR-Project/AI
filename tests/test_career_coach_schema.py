from datetime import datetime, timezone
import pytest
from pydantic import ValidationError

from app.schemas.career_coach import (
    CareerCoachSuccessResponse,
    CareerCoachInsufficientDataResponse,
    StrengthItem,
    DevelopmentAreaItem,
    DevelopmentPlanAction,
    FollowUp,
    PriorityLevel,
)

VALID_SUCCESS_DATA = {
    "status": "success",
    "employee_id": "EMP-001",
    "strengths": [
        {
            "title": "High-Efficiency Backend Development",
            "description": "Consistently delivers resilient microservice solutions with measurable optimization.",
            "evidence": ["95% task completion rate in 2026-Q3", "Reduced latency by 40%"]
        }
    ],
    "development_areas": [
        {
            "title": "Proactive Risk Escalation",
            "description": "Needs to alert technical leads earlier when dependencies stall.",
            "evidence": ["Evaluation theme highlighted delay in reporting upstream blockers"],
            "priority": "high"
        }
    ],
    "development_plan": [
        {
            "action": "Weekly Standup Blocker Summary",
            "reason": "Improves cross-functional visibility on critical project paths.",
            "measurable_target": "Log blockers in sprint dashboard every Monday and Thursday",
            "suggested_timeline": "Next 30 days"
        }
    ],
    "follow_up": {
        "checkpoint": "Mid-Quarter Check-in (2026-11-01)",
        "review_focus": "Review blocker logging compliance and team velocity metrics."
    },
    "created_at": datetime.now(timezone.utc).isoformat()
}

# 1. Valid Career Coach response passes validation
def test_valid_career_coach_response():
    model = CareerCoachSuccessResponse.model_validate(VALID_SUCCESS_DATA)
    assert model.status == "success"
    assert model.employee_id == "EMP-001"
    assert len(model.strengths) == 1
    assert model.strengths[0].evidence[0] == "95% task completion rate in 2026-Q3"
    assert model.development_areas[0].priority == PriorityLevel.HIGH
    assert model.development_plan[0].action == "Weekly Standup Blocker Summary"
    assert isinstance(model.created_at, datetime)

# 2. Invalid or missing required fields fail validation
def test_missing_required_fields():
    incomplete_data = VALID_SUCCESS_DATA.copy()
    del incomplete_data["strengths"]

    with pytest.raises(ValidationError) as exc_info:
        CareerCoachSuccessResponse.model_validate(incomplete_data)
    assert "strengths" in str(exc_info.value)

def test_empty_lists_fail_validation():
    data = VALID_SUCCESS_DATA.copy()
    data["strengths"] = []  # min_length=1

    with pytest.raises(ValidationError) as exc_info:
        CareerCoachSuccessResponse.model_validate(data)
    assert "strengths" in str(exc_info.value)

# 3. Development plan and priority field validation
def test_invalid_priority_value():
    data = VALID_SUCCESS_DATA.copy()
    data["development_areas"] = [
        {
            "title": "Test Area",
            "description": "Some description",
            "evidence": ["Some evidence"],
            "priority": "critical"  # invalid! Only high, medium, low allowed
        }
    ]
    with pytest.raises(ValidationError) as exc_info:
        CareerCoachSuccessResponse.model_validate(data)
    assert "priority" in str(exc_info.value)

def test_development_plan_missing_measurable_target():
    data = VALID_SUCCESS_DATA.copy()
    data["development_plan"] = [
        {
            "action": "Do something",
            "reason": "Good reason",
            # "measurable_target" missing
            "suggested_timeline": "2 weeks"
        }
    ]
    with pytest.raises(ValidationError) as exc_info:
        CareerCoachSuccessResponse.model_validate(data)
    assert "measurable_target" in str(exc_info.value)

# 4. Insufficient data fallback response works correctly
def test_insufficient_data_response():
    data = {
        "status": "insufficient_data",
        "employee_id": "EMP-002",
        "missing_categories": ["performance", "evaluation_themes"],
        "message": "Not enough approved employee data to generate a reliable career coaching plan."
    }
    model = CareerCoachInsufficientDataResponse.model_validate(data)
    assert model.status == "insufficient_data"
    assert model.employee_id == "EMP-002"
    assert "performance" in model.missing_categories
    assert isinstance(model.created_at, datetime)

def test_insufficient_data_empty_categories_fail():
    data = {
        "status": "insufficient_data",
        "employee_id": "EMP-002",
        "missing_categories": [],  # min_length=1
    }
    with pytest.raises(ValidationError):
        CareerCoachInsufficientDataResponse.model_validate(data)

# 5. Sensitive fields are not allowed in the schema (extra fields forbidden)
def test_sensitive_fields_rejected_by_extra_forbid():
    sensitive_data = VALID_SUCCESS_DATA.copy()
    sensitive_data["salary"] = 120000  # Extra sensitive field
    with pytest.raises(ValidationError) as exc_info:
        CareerCoachSuccessResponse.model_validate(sensitive_data)
    assert "Extra inputs are not permitted" in str(exc_info.value)

def test_sensitive_fields_in_nested_items_rejected():
    data = VALID_SUCCESS_DATA.copy()
    data["strengths"] = [
        {
            "title": "Good performance",
            "description": "Delivers well",
            "evidence": ["Solid delivery"],
            "bonus_amount": 5000  # Extra sensitive field
        }
    ]
    with pytest.raises(ValidationError) as exc_info:
        CareerCoachSuccessResponse.model_validate(data)
    assert "Extra inputs are not permitted" in str(exc_info.value)

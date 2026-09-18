"""Unit tests for Performance Insight schemas.

Covers:
- Valid request and response schemas
- Required field validation and empty value handling
- Invalid types, values, and boundary ranges (e.g., metric percentages)
- Unexpected field rejection (strict extra='forbid')
- Insufficient-data fallback schema and discriminated union parsing
- Safety constraint: prevention of representing speculative causes as verified facts
"""

from datetime import datetime, timezone

import pytest
from pydantic import TypeAdapter, ValidationError

from app.schemas.career_coach import PriorityLevel
from app.schemas.performance_insight import (
    AIInterpretation,
    CalculatedTrends,
    ContributingIndicator,
    MetricTrend,
    PerformanceDeclineItem,
    PerformanceImprovementItem,
    PerformanceInsightAIGeneration,
    PerformanceInsightInsufficientDataResponse,
    PerformanceInsightRequest,
    PerformanceInsightResponse,
    PerformanceInsightSuccessResponse,
    PerformancePeriodMetrics,
    ReviewActionItem,
    TrendDirection,
    VerifiedFacts,
)

VALID_METRICS_Q1 = {
    "period": "2026-Q1",
    "overall_score": 82.0,
    "task_completion_rate": 85.0,
    "goal_achievement_rate": 80.0,
    "attendance_rate": 96.0,
}

VALID_METRICS_Q2 = {
    "period": "2026-Q2",
    "overall_score": 88.0,
    "task_completion_rate": 92.0,
    "goal_achievement_rate": 86.0,
    "attendance_rate": 96.0,
}

VALID_SUCCESS_DATA = {
    "status": "success",
    "employee_id": "EMP-001",
    "verified_facts": {
        "target_period": "2026-Q2",
        "comparison_period": "2026-Q1",
        "metrics_by_period": [VALID_METRICS_Q1, VALID_METRICS_Q2],
    },
    "calculated_trends": {
        "from_period": "2026-Q1",
        "to_period": "2026-Q2",
        "metrics": {
            "overall_score": {
                "metric_name": "overall_score",
                "previous_value": 82.0,
                "current_value": 88.0,
                "delta": 6.0,
                "direction": "improved",
                "percent_change": 7.32,
            },
            "task_completion_rate": {
                "metric_name": "task_completion_rate",
                "previous_value": 85.0,
                "current_value": 92.0,
                "delta": 7.0,
                "direction": "improved",
                "percent_change": 8.24,
            },
            "goal_achievement_rate": {
                "metric_name": "goal_achievement_rate",
                "previous_value": 80.0,
                "current_value": 86.0,
                "delta": 6.0,
                "direction": "improved",
                "percent_change": 7.5,
            },
            "attendance_rate": {
                "metric_name": "attendance_rate",
                "previous_value": 96.0,
                "current_value": 96.0,
                "delta": 0.0,
                "direction": "stable",
                "percent_change": 0.0,
            },
        },
        "improved_metrics": ["overall_score", "task_completion_rate", "goal_achievement_rate"],
        "declined_metrics": [],
        "stable_metrics": ["attendance_rate"],
    },
    "ai_interpretation": {
        "summary": "Overall performance demonstrated measurable positive progression from Q1 to Q2, driven by higher task delivery velocity.",
        "improvements": [
            {
                "metric": "task_completion_rate",
                "summary": "Task completion increased from 85% to 92%, indicating stronger sprint throughput.",
                "contributing_indicators": [
                    {
                        "indicator_name": "Sprint backlog resolution",
                        "category": "workload",
                        "observation": "Observed reduction in rollover tasks across sprint cycles to review with team lead.",
                    }
                ],
            }
        ],
        "declines": [],
    },
    "suggested_review_actions": [
        {
            "priority": "medium",
            "focus_area": "Task Estimation Calibration",
            "recommended_action": "Conduct 1-on-1 review on quarterly sprint sizing to sustain delivery velocity.",
            "rationale": "High task throughput indicates potential capacity for expanded technical responsibility.",
        }
    ],
    "created_at": datetime.now(timezone.utc).isoformat(),
}

VALID_INSUFFICIENT_DATA = {
    "status": "insufficient_data",
    "employee_id": "EMP-002",
    "reason": "Only one approved performance period available; comparative trends require at least two periods.",
    "periods_found": ["2026-Q1"],
    "message": "Insufficient approved performance data to generate comparative performance insights.",
    "created_at": datetime.now(timezone.utc).isoformat(),
}


# 1. Request Schema Tests
def test_valid_request():
    req1 = PerformanceInsightRequest(employee_id="EMP-001")
    assert req1.employee_id == "EMP-001"
    assert req1.period is None

    req2 = PerformanceInsightRequest(employee_id="EMP-001", period="2026-Q3")
    assert req2.employee_id == "EMP-001"
    assert req2.period == "2026-Q3"


def test_request_missing_required_employee_id():
    with pytest.raises(ValidationError) as exc:
        PerformanceInsightRequest.model_validate({})
    assert "employee_id" in str(exc.value)


def test_request_empty_employee_id():
    with pytest.raises(ValidationError) as exc:
        PerformanceInsightRequest(employee_id="")
    assert "employee_id" in str(exc.value)


def test_request_unexpected_fields_forbidden():
    with pytest.raises(ValidationError) as exc:
        PerformanceInsightRequest.model_validate(
            {"employee_id": "EMP-001", "unauthorized_param": "malicious"}
        )
    assert "extra_forbidden" in str(exc.value)


# 2. Success Response Schema Tests
def test_valid_success_response():
    resp = PerformanceInsightSuccessResponse.model_validate(VALID_SUCCESS_DATA)
    assert resp.status == "success"
    assert resp.employee_id == "EMP-001"
    assert len(resp.verified_facts.metrics_by_period) == 2
    assert resp.calculated_trends.metrics["overall_score"].direction == TrendDirection.IMPROVED
    assert len(resp.ai_interpretation.improvements) == 1
    assert len(resp.suggested_review_actions) == 1
    assert resp.suggested_review_actions[0].priority == PriorityLevel.MEDIUM


def test_success_response_unexpected_fields_forbidden():
    data = dict(VALID_SUCCESS_DATA)
    data["extra_field"] = "not_allowed"
    with pytest.raises(ValidationError) as exc:
        PerformanceInsightSuccessResponse.model_validate(data)
    assert "extra_forbidden" in str(exc.value)


def test_missing_required_sections_in_response():
    # Missing verified_facts
    incomplete_data = dict(VALID_SUCCESS_DATA)
    del incomplete_data["verified_facts"]
    with pytest.raises(ValidationError) as exc:
        PerformanceInsightSuccessResponse.model_validate(incomplete_data)
    assert "verified_facts" in str(exc.value)

    # Missing calculated_trends
    incomplete_data2 = dict(VALID_SUCCESS_DATA)
    del incomplete_data2["calculated_trends"]
    with pytest.raises(ValidationError) as exc:
        PerformanceInsightSuccessResponse.model_validate(incomplete_data2)
    assert "calculated_trends" in str(exc.value)


# 3. Validation on Metric Ranges & Types
def test_metric_boundary_validation():
    # Overall score > 100
    invalid_metrics = dict(VALID_METRICS_Q1)
    invalid_metrics["overall_score"] = 105.0
    with pytest.raises(ValidationError) as exc:
        PerformancePeriodMetrics.model_validate(invalid_metrics)
    assert "less_than_equal" in str(exc.value)

    # Overall score < 0
    invalid_metrics["overall_score"] = -1.0
    with pytest.raises(ValidationError) as exc:
        PerformancePeriodMetrics.model_validate(invalid_metrics)
    assert "greater_than_equal" in str(exc.value)


def test_invalid_trend_direction():
    with pytest.raises(ValidationError) as exc:
        MetricTrend.model_validate(
            {
                "metric_name": "overall_score",
                "previous_value": 80.0,
                "current_value": 85.0,
                "delta": 5.0,
                "direction": "skyrocketed",  # Invalid direction
            }
        )
    assert "direction" in str(exc.value)


# 4. Insufficient Data Fallback Response Tests
def test_valid_insufficient_data_response():
    resp = PerformanceInsightInsufficientDataResponse.model_validate(VALID_INSUFFICIENT_DATA)
    assert resp.status == "insufficient_data"
    assert resp.employee_id == "EMP-002"
    assert resp.periods_found == ["2026-Q1"]
    assert "Only one approved performance period" in resp.reason


def test_insufficient_data_unexpected_fields_forbidden():
    data = dict(VALID_INSUFFICIENT_DATA)
    data["extra_data"] = "forbidden"
    with pytest.raises(ValidationError) as exc:
        PerformanceInsightInsufficientDataResponse.model_validate(data)
    assert "extra_forbidden" in str(exc.value)


# 5. Discriminated Union Tests
def test_discriminated_union_success():
    adapter = TypeAdapter(PerformanceInsightResponse)
    parsed = adapter.validate_python(VALID_SUCCESS_DATA)
    assert isinstance(parsed, PerformanceInsightSuccessResponse)
    assert parsed.status == "success"


def test_discriminated_union_insufficient_data():
    adapter = TypeAdapter(PerformanceInsightResponse)
    parsed = adapter.validate_python(VALID_INSUFFICIENT_DATA)
    assert isinstance(parsed, PerformanceInsightInsufficientDataResponse)
    assert parsed.status == "insufficient_data"


def test_discriminated_union_invalid_status():
    adapter = TypeAdapter(PerformanceInsightResponse)
    invalid = dict(VALID_SUCCESS_DATA)
    invalid["status"] = "in_progress"
    with pytest.raises(ValidationError) as exc:
        adapter.validate_python(invalid)
    assert "status" in str(exc.value)


# 6. Safety Constraint: Prevention of Representing Speculative Causes as Verified Facts
def test_verified_facts_forbids_causal_speculation():
    """Verified facts layer must reject attempts to inject subjective or causal claims."""
    # Attempt to inject cause into factual metrics
    with pytest.raises(ValidationError) as exc:
        PerformancePeriodMetrics.model_validate(
            {
                **VALID_METRICS_Q1,
                "cause": "Employee took extra training classes",  # Subjective cause injection
            }
        )
    assert "extra_forbidden" in str(exc.value)

    # Attempt to inject root cause into VerifiedFacts container
    with pytest.raises(ValidationError) as exc:
        VerifiedFacts.model_validate(
            {
                "target_period": "2026-Q2",
                "comparison_period": "2026-Q1",
                "metrics_by_period": [VALID_METRICS_Q1, VALID_METRICS_Q2],
                "proven_cause": "System downtime caused drop in Q1",
            }
        )
    assert "extra_forbidden" in str(exc.value)


def test_contributing_indicator_requires_safe_semantics():
    """Contributing indicators must be framed as observable indicators to review, not proven facts."""
    indicator = ContributingIndicator(
        indicator_name="Goal deadline shifts",
        category="goal_scope",
        observation="Two quarterly deliverables experienced deadline rescheduling.",
    )
    assert indicator.indicator_name == "Goal deadline shifts"
    assert indicator.category == "goal_scope"
    assert "rescheduling" in indicator.observation

    # Forbid attempting to define confirmed root cause fields
    with pytest.raises(ValidationError) as exc:
        ContributingIndicator.model_validate(
            {
                "indicator_name": "Goal shifts",
                "category": "goal_scope",
                "observation": "Deliverables rescheduled.",
                "root_cause": "Poor management oversight",  # Forbids ungrounded causal claims
            }
        )
    assert "extra_forbidden" in str(exc.value)


def test_improvements_and_declines_structure():
    """Validates that improvements and declines use contributing indicators as review context."""
    imp = PerformanceImprovementItem(
        metric="goal_achievement_rate",
        summary="Goal completion rate improved by 6%.",
        contributing_indicators=[
            ContributingIndicator(
                indicator_name="Milestone completion cadence",
                category="goals",
                observation="All key milestones completed on or before target sprint dates.",
            )
        ],
    )
    assert imp.metric == "goal_achievement_rate"
    assert len(imp.contributing_indicators) == 1

    dec = PerformanceDeclineItem(
        metric="attendance_rate",
        summary="Attendance rate dropped from 99% to 94%.",
        contributing_indicators=[
            ContributingIndicator(
                indicator_name="Recorded shift intervals",
                category="attendance",
                observation="Multiple partial-day absences logged in period.",
            )
        ],
    )
    assert dec.metric == "attendance_rate"
    assert len(dec.contributing_indicators) == 1


# 7. AI Raw Generation Schema Tests
def test_ai_generation_schema_ignores_extra():
    """The raw LLM output model should cleanly parse expected fields while ignoring extra fields."""
    raw_llm_output = {
        "summary": "Performance improved across key technical execution metrics during Q2.",
        "improvements": [
            {
                "metric": "overall_score",
                "summary": "Overall score increased by 6 points.",
                "contributing_indicators": [],
            }
        ],
        "declines": [],
        "suggested_review_actions": [
            {
                "priority": "high",
                "focus_area": "Technical Architecture",
                "recommended_action": "Review system designs for next quarter.",
                "rationale": "High score supports senior design ownership.",
            }
        ],
        # LLM hallucinated extra fields should be ignored safely
        "employee_id": "EMP-ATTEMPTED-OVERWRITE",
        "status": "tampered",
    }
    gen = PerformanceInsightAIGeneration.model_validate(raw_llm_output)
    assert gen.summary.startswith("Performance improved")
    assert len(gen.improvements) == 1
    assert len(gen.suggested_review_actions) == 1


def test_individual_submodel_instantiation():
    """Validates direct instantiation of submodels AIInterpretation, CalculatedTrends, and ReviewActionItem."""
    action = ReviewActionItem(
        priority=PriorityLevel.HIGH,
        focus_area="Delivery",
        recommended_action="Action to take",
        rationale="Operational reason",
    )
    assert action.priority == PriorityLevel.HIGH

    trends = CalculatedTrends(
        from_period="2026-Q1",
        to_period="2026-Q2",
        metrics={
            "overall_score": MetricTrend(
                metric_name="overall_score",
                previous_value=80.0,
                current_value=85.0,
                delta=5.0,
                direction=TrendDirection.IMPROVED,
            )
        },
        improved_metrics=["overall_score"],
    )
    assert trends.from_period == "2026-Q1"

    interp = AIInterpretation(
        summary="Clear summary of overall improvements.",
        improvements=[],
        declines=[],
    )
    assert interp.summary.startswith("Clear summary")


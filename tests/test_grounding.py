"""Focused unit tests for structured, field-aware numeric grounding.

Requirements covered:
1. 'Q2' is ignored as a metric.
2. 'Q3 2026' is ignored as metrics.
3. '2026-Q3' is ignored as metrics.
4. 'PerformanceRecord #2' is ignored as a metric.
5. Employee IDs like 'EMP-001' or 'EMP-ENG-ALICE' are ignored as metrics.
6. Skill/task/goal IDs like 'Goal #5', 'Task #12', 'Skill #3' are ignored as metrics.
7. Genuine percentage metrics still require grounding (e.g., 95.0% vs unsupported 77.7%).
8. Genuine score/count metrics still require grounding (e.g., score 90.0, 3 delayed goals).
9. Unsupported/fabricated numeric business metrics are still rejected.
10. Mixed response containing IDs + periods + real metrics is validated correctly.
11. Structured facts correctly represent source type, source ID, field name, and numeric value.
12. Integration with all 7 AI services' extractors and numeric grounding checks.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.models import Goal, PerformanceRecord, Skill
from app.schemas.career_coach import (
    CareerCoachModelOutput,
    DevelopmentAreaItem,
    DevelopmentPlanAction,
    EvidenceItem,
    FollowUp,
    PriorityLevel,
    StrengthItem,
)
from app.services.attention_signal_ai import (
    _extract_context_numbers as attention_extract_context_numbers,
    _extract_numbers_from_text as attention_extract_numbers,
)
from app.services.career_coach_ai import (
    CareerCoachAIService,
    CareerCoachAIServiceError,
    _extract_numbers_from_text as career_coach_extract_numbers,
    _extract_source_numbers as career_coach_extract_source_numbers,
)
from app.services.evaluation_draft_ai import (
    EvaluationDraftAIService,
    _extract_numbers_from_record as eval_draft_extract_record_numbers,
    _extract_numbers_from_text as eval_draft_extract_numbers,
)
from app.services.grounding import (
    GroundedNumericFact,
    clean_text_for_numeric_extraction,
    extract_business_metrics_from_text,
    extract_facts_from_record,
    extract_grounded_facts_from_context,
    get_allowed_numeric_set,
    validate_numeric_grounding,
)
from app.services.performance_insight_ai import (
    _extract_context_numbers as perf_insight_extract_context_numbers,
    _extract_numbers_from_text as perf_insight_extract_numbers,
)
from app.services.policy_ai import (
    _extract_numbers_from_text as policy_extract_numbers,
    _extract_policy_numbers as policy_extract_policy_numbers,
)
from app.services.skill_gap_ai import (
    _extract_numbers_from_text as skill_gap_extract_numbers,
    _extract_source_numbers as skill_gap_extract_source_numbers,
)
from app.services.team_insight_ai import (
    _extract_context_numbers as team_insight_extract_context_numbers,
    _extract_numbers_from_text as team_insight_extract_numbers,
)


# ==============================================================================
# 1. Period Tokens Ignored as Metrics
# ==============================================================================

def test_q2_is_ignored_as_a_metric():
    """Q2 or Q1-Q4 should not be extracted as numeric metric 2."""
    text = "In Q2 performance was evaluated."
    numbers = extract_business_metrics_from_text(text)
    assert numbers == []
    assert 2.0 not in numbers

    text_with_metric = "In Q2 the team completed 85.0% of their objectives."
    numbers_metric = extract_business_metrics_from_text(text_with_metric)
    assert numbers_metric == [85.0]
    assert 2.0 not in numbers_metric


def test_q3_2026_is_ignored_as_metrics():
    """'Q3 2026' or 'Quarter 3, 2026' should not be extracted as 3 or 2026."""
    text = "The review conducted in Q3 2026 showed steady improvement."
    numbers = extract_business_metrics_from_text(text)
    assert numbers == []
    assert 3.0 not in numbers
    assert 2026.0 not in numbers

    text_with_metric = "During Quarter 3 2026, employee achieved a score of 91.5."
    numbers_metric = extract_business_metrics_from_text(text_with_metric)
    assert numbers_metric == [91.5]
    assert 3.0 not in numbers_metric
    assert 2026.0 not in numbers_metric


def test_2026_q3_is_ignored_as_metrics():
    """'2026-Q3' or '2025-Q1' should not extract 2026 or 3 as business metrics."""
    text = "Cycle 2026-Q3 performance review."
    numbers = extract_business_metrics_from_text(text)
    assert numbers == []
    assert 2026.0 not in numbers
    assert 3.0 not in numbers

    text_with_metric = "In cycle 2026-Q3, 4 goals were delivered."
    numbers_metric = extract_business_metrics_from_text(text_with_metric)
    assert numbers_metric == [4.0]
    assert 2026.0 not in numbers_metric
    assert 3.0 not in numbers_metric


def test_calendar_years_and_dates_are_ignored():
    """Dates like 2026-08-20 and standalone years like FY2026 should be ignored."""
    text = "On 2026-08-20 for FY2026, employee logged 40 hours."
    numbers = extract_business_metrics_from_text(text)
    assert numbers == [40.0]
    assert 2026.0 not in numbers
    assert 8.0 not in numbers
    assert 20.0 not in numbers


def test_clean_text_for_numeric_extraction():
    """clean_text_for_numeric_extraction replaces periods and IDs with spaces."""
    raw = "In 2026-Q3 EMP-001 achieved 95.0% on Goal #5."
    cleaned = clean_text_for_numeric_extraction(raw)
    assert "2026-Q3" not in cleaned
    assert "EMP-001" not in cleaned
    assert "Goal #5" not in cleaned
    assert "95.0" in cleaned


# ==============================================================================
# 2. Entity and Database Identifiers Ignored as Metrics
# ==============================================================================

def test_performancerecord_id_is_ignored_as_a_metric():
    """'PerformanceRecord #2' or 'Record 2' must not be extracted as numeric metric 2."""
    text = "Referencing PerformanceRecord #2 with a rating of 4.5."
    numbers = extract_business_metrics_from_text(text)
    assert numbers == [4.5]
    assert 2.0 not in numbers

    text_plain = "Based on Record #14 and PerformanceRecord 7."
    numbers_plain = extract_business_metrics_from_text(text_plain)
    assert numbers_plain == []


def test_employee_ids_are_ignored_as_metrics():
    """Employee IDs like EMP-001, EMP-ENG-ALICE, EMP-1002 must not extract digits as metrics."""
    text = "Employee EMP-001 achieved 92.5% on their evaluation."
    numbers = extract_business_metrics_from_text(text)
    assert numbers == [92.5]
    assert 1.0 not in numbers
    assert 0.0 not in numbers

    text_multi = "Review for EMP-ENG-ALICE and EMP-1002 showed 5 overdue tasks."
    numbers_multi = extract_business_metrics_from_text(text_multi)
    assert numbers_multi == [5.0]
    assert 1002.0 not in numbers_multi


def test_skill_task_goal_policy_ids_are_ignored_as_metrics():
    """IDs like Goal #5, Task #12, Skill #3, Policy #1 must not be extracted as metrics."""
    text = "Completed Goal #5 and Task #12 in Skill #3, referenced Policy #1."
    numbers = extract_business_metrics_from_text(text)
    assert numbers == []
    assert 5.0 not in numbers
    assert 12.0 not in numbers
    assert 3.0 not in numbers
    assert 1.0 not in numbers

    text_with_metric = "Completed Goal #5 with 3 days remaining and a score of 88.0."
    numbers_with_metric = extract_business_metrics_from_text(text_with_metric)
    assert numbers_with_metric == [3.0, 88.0]
    assert 5.0 not in numbers_with_metric


# ==============================================================================
# 3. Genuine Business Metrics Require Grounding
# ==============================================================================

def test_genuine_percentage_metrics_require_grounding():
    """Percentages like 95.0% are extracted and require grounding; unsupported ones fail."""
    allowed = {95.0, 80.0}

    # Supported percentage
    valid, ungrounded = validate_numeric_grounding([95.0], allowed)
    assert valid is True
    assert ungrounded == []

    # Unsupported percentage
    valid_bad, ungrounded_bad = validate_numeric_grounding([77.7], allowed)
    assert valid_bad is False
    assert 77.7 in ungrounded_bad


def test_genuine_score_and_count_metrics_require_grounding():
    """Scores like 90.0 and counts like 3 delayed goals are extracted and require grounding."""
    text = "The overall score was 90.0 and there were 3 delayed goals."
    extracted = extract_business_metrics_from_text(text)
    assert extracted == [90.0, 3.0]

    allowed = {90.0, 3.0, 10.0}
    valid, ungrounded = validate_numeric_grounding(extracted, allowed)
    assert valid is True
    assert ungrounded == []


def test_unsupported_fabricated_numeric_business_metrics_rejected():
    """Fabricated metrics not in the allowed ground truth are rejected."""
    allowed = {85.0, 4.0}
    fabricated_extracted = [85.0, 99.9]

    valid, ungrounded = validate_numeric_grounding(fabricated_extracted, allowed)
    assert valid is False
    assert 99.9 in ungrounded


# ==============================================================================
# 4. Mixed Response Containing IDs + Periods + Real Metrics
# ==============================================================================

def test_mixed_response_validated_correctly():
    """A response with periods, IDs, and real metrics validates real metrics while ignoring metadata tokens."""
    context_allowed = {95.0, 3.0}

    # Response mentions Q3 2026, 2026-Q3, EMP-001, Task #12, Goal #5, plus real metrics 95.0% and 3
    response_text = (
        "In 2026-Q3 (reviewed during Q3 2026), employee EMP-001 completed Task #12 and Goal #5. "
        "The overall score reached 95.0% with 3 completed milestones."
    )
    extracted = extract_business_metrics_from_text(response_text)
    assert extracted == [95.0, 3.0]

    valid, ungrounded = validate_numeric_grounding(extracted, context_allowed)
    assert valid is True
    assert ungrounded == []

    # If the response also includes a fabricated metric (e.g. 99.0%), it must fail
    bad_response = response_text + " Efficiency improved by 99.0%."
    bad_extracted = extract_business_metrics_from_text(bad_response)
    assert 99.0 in bad_extracted

    valid_bad, ungrounded_bad = validate_numeric_grounding(bad_extracted, context_allowed)
    assert valid_bad is False
    assert 99.0 in ungrounded_bad


# ==============================================================================
# 5. Structured Facts Field Representation
# ==============================================================================

def test_structured_facts_representation():
    """GroundedNumericFact properly captures source type, source ID, field name, value, period, and unit."""
    fact = GroundedNumericFact(
        source_type="PerformanceRecord",
        source_id=42,
        field_name="rating",
        value=4.8,
        period="2026-Q3",
        unit="score",
    )
    assert fact.source_type == "PerformanceRecord"
    assert fact.source_id == 42
    assert fact.field_name == "rating"
    assert fact.value == 4.8
    assert fact.period == "2026-Q3"
    assert fact.unit == "score"


def test_extract_facts_from_record_ignores_id_and_extracts_metrics():
    """extract_facts_from_record extracts score, completion_rate from records while ignoring record id."""
    record = PerformanceRecord(
        id=7,
        employee_id="EMP-001",
        period="2026-Q3",
        overall_score=88.5,
        task_completion_rate=95.0,
        goal_achievement_rate=90.0,
        attendance_rate=99.0,
    )
    facts = extract_facts_from_record(record, source_type="PerformanceRecord")
    fact_values = {f.value for f in facts}

    # Record ID (7) and period (2026, 3) must NOT be extracted as numeric facts
    assert 7.0 not in fact_values
    assert 2026.0 not in fact_values
    assert 3.0 not in fact_values

    # Real metrics: overall_score (88.5), task_completion_rate (95.0), goal_achievement_rate (90.0)
    assert 88.5 in fact_values
    assert 95.0 in fact_values
    assert 90.0 in fact_values
    assert 99.0 in fact_values


def test_extract_facts_from_goal_ignores_id_and_extracts_metrics():
    """extract_facts_from_record on Goal extracts progress while ignoring goal ID."""
    goal = Goal(
        id=15,
        employee_id="EMP-001",
        title="Deliver Sprint 12",
        progress=85.0,
        status="in_progress",
        period="2026-Q2",
    )
    facts = extract_facts_from_record(goal, source_type="Goal")
    allowed_set = get_allowed_numeric_set(facts)

    # Goal ID 15 and 2026-Q2 must NOT be in allowed set
    assert 15.0 not in allowed_set
    assert 2026.0 not in allowed_set

    # Real metric value: progress 85.0%
    assert 85.0 in allowed_set


def test_extract_grounded_facts_from_context():
    """extract_grounded_facts_from_context processes heterogeneous lists of records."""
    skill = Skill(
        id=3,
        employee_id="EMP-001",
        name="Python",
        level="Advanced",
        evidence="Completed 10 tasks with 95.0% accuracy.",
    )
    records = [
        PerformanceRecord(
            id=1,
            employee_id="EMP-001",
            period="2026-Q3",
            overall_score=92.0,
            task_completion_rate=90.0,
            goal_achievement_rate=88.0,
            attendance_rate=99.0,
        ),
        Goal(
            id=2,
            employee_id="EMP-001",
            title="Sprint Goal",
            progress=75.0,
            status="completed",
        ),
        skill,
    ]
    facts = extract_grounded_facts_from_context(records)
    allowed = get_allowed_numeric_set(facts)

    # Metrics present: 92.0, 90.0, 88.0, 99.0, 75.0, 10.0, 95.0
    assert 92.0 in allowed
    assert 75.0 in allowed
    assert 10.0 in allowed
    assert 95.0 in allowed

    # Record and Employee IDs 1, 2, 3 must NOT be in allowed numeric set
    assert 1.0 not in allowed
    assert 2.0 not in allowed
    assert 3.0 not in allowed


# ==============================================================================
# 6. Service-Level Numeric Grounding Integration Tests
# ==============================================================================

def test_career_coach_service_integration():
    """Career coach extractor and _validate_evidence_grounding method tests."""
    text = "In Q3 2026, employee EMP-001 completed Goal #5 with score 88.0."
    nums = career_coach_extract_numbers(text)
    assert nums == [88.0]
    assert 3.0 not in nums
    assert 2026.0 not in nums
    assert 1.0 not in nums
    assert 5.0 not in nums

    service = CareerCoachAIService(api_key="mock-key", client=MagicMock())

    approved_sources = {
        ("performance", 1): {
            "source_type": "performance",
            "source_id": 1,
            "rating": 4.5,
            "score": 88.0,
            "summary": "Achieved 88.0 score in Q3 2026.",
        }
    }

    # Verify source numbers extraction
    source_nums = career_coach_extract_source_numbers(approved_sources[("performance", 1)])
    assert 4.5 in source_nums
    assert 88.0 in source_nums
    assert 1.0 not in source_nums

    # Valid output referencing score 88.0 and Q3 2026
    valid_output = CareerCoachModelOutput(
        status="success",
        strengths=[
            StrengthItem(
                title="Consistent delivery",
                description="Consistently achieves targets",
                evidence=[
                    EvidenceItem(
                        source_type="performance",
                        source_id=1,
                        claim="Achieved overall score of 88.0 in Q3 2026.",
                    )
                ],
            )
        ],
        development_areas=[
            DevelopmentAreaItem(
                title="System Design",
                description="Expand system architecture knowledge",
                evidence=[
                    EvidenceItem(
                        source_type="performance",
                        source_id=1,
                        claim="Rating was 4.5.",
                    )
                ],
                priority=PriorityLevel.MEDIUM,
            )
        ],
        development_plan=[
            DevelopmentPlanAction(
                action="Attend architecture workshops",
                reason="Deepen architecture background",
                measurable_target="Complete 1 workshop",
                suggested_timeline="30 days",
            )
        ],
        follow_up=FollowUp(
            checkpoint="Q4 2026",
            review_focus="Evaluate design review participation",
        ),
    )

    # Valid grounding should not raise
    service._validate_evidence_grounding(valid_output, approved_sources)

    # Fabricated metric (99.9) in evidence claim must raise CareerCoachAIServiceError
    bad_output = CareerCoachModelOutput(
        status="success",
        strengths=[
            StrengthItem(
                title="Consistent delivery",
                description="Consistently achieves targets",
                evidence=[
                    EvidenceItem(
                        source_type="performance",
                        source_id=1,
                        claim="Achieved fabricated score of 99.9 in Q3 2026.",
                    )
                ],
            )
        ],
        development_areas=valid_output.development_areas,
        development_plan=valid_output.development_plan,
        follow_up=valid_output.follow_up,
    )

    with pytest.raises(CareerCoachAIServiceError) as exc_info:
        service._validate_evidence_grounding(bad_output, approved_sources)
    assert "99.9" in str(exc_info.value)


def test_skill_gap_service_integration():
    """Skill gap extractor ignores IDs and validates evidence."""
    text = "Skill #4 proficiency is 3.5 out of 5.0."
    nums = skill_gap_extract_numbers(text)
    assert 4.0 not in nums
    assert 3.5 in nums
    assert 5.0 in nums

    source_data = {
        "source_type": "skill",
        "source_id": 4,
        "name": "Python",
        "description": "Proficiency 3.5 out of 5.0",
    }
    extracted_source_nums = skill_gap_extract_source_numbers(source_data)
    assert 3.5 in extracted_source_nums
    assert 5.0 in extracted_source_nums
    assert 4.0 not in extracted_source_nums


def test_evaluation_draft_service_integration():
    """Evaluation draft extractor ignores period tokens and IDs, and validates evidence."""
    text = "Cycle 2026-Q3 performance: PerformanceRecord #2 rating 4.0."
    nums = eval_draft_extract_numbers(text)
    assert nums == [4.0]
    assert 2026.0 not in nums
    assert 3.0 not in nums
    assert 2.0 not in nums

    service = EvaluationDraftAIService(client=MagicMock())
    approved_sources = {
        ("performance", 2): {
            "source_type": "performance",
            "source_id": 2,
            "rating": 4.0,
        }
    }

    # Verify record extractor
    rec_nums = eval_draft_extract_record_numbers(approved_sources[("performance", 2)])
    assert rec_nums == {4.0}

    # Valid evidence
    service._validate_evidence_items(
        [
            EvidenceItem(
                source_type="performance",
                source_id=2,
                claim="Rating was 4.0 in 2026-Q3.",
            )
        ],
        approved_sources,
    )

    # Fabricated metric raises ValueError
    with pytest.raises(ValueError) as exc_info:
        service._validate_evidence_items(
            [
                EvidenceItem(
                    source_type="performance",
                    source_id=2,
                    claim="Rating was 4.9 in 2026-Q3.",
                )
            ],
            approved_sources,
        )
    assert "4.9" in str(exc_info.value)


def test_performance_insight_service_integration():
    """Performance insight extractor ignores IDs and periods, validates grounding."""
    text = "Employee EMP-ENG-ALICE in Q2 achieved 90.0% completion."
    nums = perf_insight_extract_numbers(text)
    assert nums == [90.0]
    assert 2.0 not in nums

    context = {
        "facts": {
            "periods": ["2026-Q2"],
            "records": [
                {
                    "id": 1,
                    "score": 90.0,
                    "rating": 4.5,
                }
            ],
        },
        "calculated_trends": {},
    }
    context_numbers = perf_insight_extract_context_numbers(context)
    assert 90.0 in context_numbers
    assert 4.5 in context_numbers
    # ID 1 must NOT be in context_numbers
    assert 1.0 not in context_numbers


def test_policy_ai_extract_numbers():
    """Policy AI extractor ignores policy codes and article/section IDs while keeping numbers."""
    text = "Under Policy POL-HR-01, Section #4, employees receive 25 days of annual leave."
    nums = policy_extract_numbers(text)
    assert nums == [25.0]
    assert 1.0 not in nums
    assert 4.0 not in nums

    policy_meta = {
        "title": "Annual Leave Policy POL-HR-01",
        "summary": "Covers 25 days of annual leave for 2026.",
        "content": "Section #4: Up to 25 days may be taken.",
    }
    extracted_policy_nums = policy_extract_policy_numbers(policy_meta)
    assert extracted_policy_nums == {25.0}
    assert 1.0 not in extracted_policy_nums
    assert 4.0 not in extracted_policy_nums
    assert 2026.0 not in extracted_policy_nums


def test_attention_signal_and_team_insight_extractors():
    """Attention signal and team insight extractors ignore IDs and periods."""
    text = "Team ENG-01 reviewed 2026-Q3 with 6 overdue tasks and 75.0% velocity."
    assert attention_extract_numbers(text) == [6.0, 75.0]
    assert team_insight_extract_numbers(text) == [6.0, 75.0]

    ctx = {
        "records": [
            {"id": 99, "score": 85.0, "period": "2026-Q3"},
        ]
    }
    assert 85.0 in attention_extract_context_numbers(ctx)
    assert 85.0 in team_insight_extract_context_numbers(ctx)
    assert 99.0 not in attention_extract_context_numbers(ctx)
    assert 2026.0 not in attention_extract_context_numbers(ctx)


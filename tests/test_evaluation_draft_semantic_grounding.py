"""Focused tests for semantic and field-level evidence validation in Evaluation Draft Assistant.

Covers all required scenarios:
1. Valid paraphrase of an evidence field -> accepted.
2. Valid numeric claim tied to the correct field -> accepted.
3. Valid qualitative claim directly supported by evidence -> accepted.
4. Valid source_id but fabricated achievement -> rejected.
5. Fabricated award -> rejected.
6. Fabricated leadership claim -> rejected.
7. Fabricated certification/project/responsibility -> rejected.
8. Claim supported by another employee's evidence -> rejected.
9. Claim based on an unapproved record -> rejected.
10. Multiple narrative sentences where one sentence is unsupported -> entire response rejected.
11. Existing valid Evaluation Draft responses continue to pass.
12. Existing numeric grounding cases with Q2/Q3/IDs continue to pass correctly.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from app.schemas.career_coach import EvidenceItem, PriorityLevel
from app.schemas.evaluation_draft import (
    EvaluationDraftModelOutput,
    EvaluationDraftSuccessResponse,
    EvaluationImprovementItem,
    EvaluationStrengthItem,
)
from app.services.evaluation_draft_ai import (
    EvaluationDraftAIService,
    EvaluationEvidenceGroundingError,
)
from app.services.grounding import (
    GroundedEvidenceFact,
    extract_all_grounded_evidence_facts,
    extract_evidence_facts_from_record,
    validate_evidence_claim_grounding,
    validate_narrative_grounding,
)


@pytest.fixture
def sample_context():
    return {
        "employee": {
            "id": "EMP-EVAL-01",
            "first_name": "Alice",
            "last_name": "Smith",
            "role_title": "Senior Software Engineer",
            "department": "Engineering",
        },
        "period": "2026-Q3",
        "has_sufficient_data": True,
        "performance": [
            {
                "id": 10,
                "period": "2026-Q3",
                "overall_score": 90.0,
                "task_completion_rate": 95.0,
                "goal_achievement_rate": 88.0,
                "attendance_rate": 99.0,
            }
        ],
        "goals": [
            {
                "id": 20,
                "title": "Migrate core monolith",
                "progress": 85.0,
                "status": "in_progress",
                "period": "2026-Q3",
            }
        ],
        "skills": [
            {
                "id": 30,
                "name": "Python & FastAPI",
                "level": "Expert",
                "evidence": "Implemented high-throughput event processing pipelines",
            }
        ],
        "task_outcomes": [
            {
                "id": 40,
                "title": "API Gateway Optimization",
                "status": "completed",
                "outcome": "Reduced latency to 45ms",
                "period": "2026-Q3",
            }
        ],
        "evaluation_themes": [
            {
                "id": 50,
                "theme": "Technical Mentorship",
                "sentiment": "positive",
                "evidence": "Mentored junior peers effectively on software design",
                "period": "2026-Q3",
            }
        ],
        "approved_sources": {
            ("performance", 10): {
                "id": 10,
                "overall_score": 90.0,
                "task_completion_rate": 95.0,
                "goal_achievement_rate": 88.0,
                "attendance_rate": 99.0,
                "period": "2026-Q3",
            },
            ("goal", 20): {
                "id": 20,
                "title": "Migrate core monolith",
                "progress": 85.0,
                "status": "in_progress",
                "period": "2026-Q3",
            },
            ("skill", 30): {
                "id": 30,
                "name": "Python & FastAPI",
                "level": "Expert",
                "evidence": "Implemented high-throughput event processing pipelines",
            },
            ("task_outcome", 40): {
                "id": 40,
                "title": "API Gateway Optimization",
                "status": "completed",
                "outcome": "Reduced latency to 45ms",
                "period": "2026-Q3",
            },
            ("evaluation_theme", 50): {
                "id": 50,
                "theme": "Technical Mentorship",
                "sentiment": "positive",
                "evidence": "Mentored junior peers effectively on software design",
                "period": "2026-Q3",
            },
        },
    }


# ==============================================================================
# 1. Field-Level Evidence Representation
# ==============================================================================

def test_grounded_evidence_fact_representation():
    """GroundedEvidenceFact preserves source_type, source_id, field_name, value, period, and unit."""
    fact = GroundedEvidenceFact(
        source_type="goal",
        source_id=20,
        field_name="title",
        value="Migrate core monolith",
        period="2026-Q3",
        unit="title",
    )
    assert fact.source_type == "goal"
    assert fact.source_id == 20
    assert fact.field_name == "title"
    assert fact.value == "Migrate core monolith"
    assert fact.period == "2026-Q3"
    assert fact.unit == "title"


def test_extract_evidence_facts_from_record():
    """extract_evidence_facts_from_record extracts all fields with semantic units."""
    rec = {
        "id": 10,
        "overall_score": 90.0,
        "task_completion_rate": 95.0,
        "period": "2026-Q3",
    }
    facts = extract_evidence_facts_from_record("performance", rec)
    fact_fields = {f.field_name: (f.value, f.unit) for f in facts}

    assert "id" not in fact_fields
    assert fact_fields["overall_score"] == (90.0, "score")
    assert fact_fields["task_completion_rate"] == (95.0, "%")


def test_extract_all_grounded_evidence_facts(sample_context):
    """extract_all_grounded_evidence_facts indexes all approved records."""
    facts = extract_all_grounded_evidence_facts(sample_context["approved_sources"], sample_context)
    assert len(facts) > 0
    source_types = {f.source_type for f in facts}
    assert "performance" in source_types
    assert "goal" in source_types
    assert "skill" in source_types
    assert "task_outcome" in source_types
    assert "evaluation_theme" in source_types
    assert "employee" in source_types


# ==============================================================================
# 2. Semantic Claim & Narrative Validation
# ==============================================================================

def test_valid_paraphrase_of_evidence_field_accepted(sample_context):
    """Valid narrative paraphrasing underlying evidence is accepted."""
    # Source has overall_score=90.0 and goal_achievement=88.0
    narrative = (
        "The employee demonstrated strong overall performance throughout the evaluation period, "
        "with an overall score of 90 and goal achievement of 88."
    )
    is_valid, reason = validate_narrative_grounding(
        narrative, sample_context["approved_sources"], sample_context
    )
    assert is_valid is True
    assert reason is None


def test_valid_numeric_claim_tied_to_correct_field(sample_context):
    """Evidence claim referencing 95.0% completion for performance record is accepted."""
    source_data = sample_context["approved_sources"][("performance", 10)]
    claim = "Achieved 95.0% task completion rate during the quarter."
    is_valid, reason = validate_evidence_claim_grounding(claim, "performance", source_data, sample_context)
    assert is_valid is True
    assert reason is None


def test_valid_qualitative_claim_supported_by_evidence(sample_context):
    """Qualitative claim regarding latency reduction is supported by task outcome #40."""
    source_data = sample_context["approved_sources"][("task_outcome", 40)]
    claim = "Successfully optimized the API Gateway, reducing latency to 45ms."
    is_valid, reason = validate_evidence_claim_grounding(claim, "task_outcome", source_data, sample_context)
    assert is_valid is True
    assert reason is None


def test_valid_source_id_but_fabricated_achievement_rejected(sample_context):
    """Citing valid PerformanceRecord #10 but claiming unrelated global leadership is rejected."""
    source_data = sample_context["approved_sources"][("performance", 10)]
    claim = "Won an international award and led a global initiative."
    is_valid, reason = validate_evidence_claim_grounding(claim, "performance", source_data, sample_context)
    assert is_valid is False
    assert "award" in reason.lower() or "leadership" in reason.lower() or "scope" in reason.lower()


def test_fabricated_award_in_narrative_rejected(sample_context):
    """Narrative claiming an unapproved award is rejected."""
    narrative = (
        "Alice demonstrated strong overall performance with overall score 90. "
        "She won the prestigious International Innovator of the Year Award."
    )
    is_valid, reason = validate_narrative_grounding(
        narrative, sample_context["approved_sources"], sample_context
    )
    assert is_valid is False
    assert "award" in reason.lower()


def test_fabricated_leadership_claim_rejected():
    """Leadership claims without leadership evidence or manager role are rejected."""
    # Context without any leadership records or titles
    context_no_leader = {
        "employee": {
            "id": "EMP-DEV-01",
            "first_name": "Bob",
            "last_name": "Jones",
            "role_title": "Junior Developer",
            "department": "Engineering",
        },
        "approved_sources": {
            ("performance", 1): {
                "id": 1,
                "overall_score": 85.0,
            }
        },
    }
    narrative = (
        "Bob maintained solid performance with score 85. "
        "He spearheaded and led the enterprise cloud migration team."
    )
    is_valid, reason = validate_narrative_grounding(
        narrative, context_no_leader["approved_sources"], context_no_leader
    )
    assert is_valid is False
    assert "leadership" in reason.lower() or "unsupported" in reason.lower()


def test_fabricated_certification_or_project_rejected(sample_context):
    """Fabricated certification and unapproved project names are rejected."""
    narrative = (
        "Alice achieved a 90 overall score. "
        "She earned her AWS Certified Security Specialist credential and delivered the Kubernetes deployment."
    )
    is_valid, reason = validate_narrative_grounding(
        narrative, sample_context["approved_sources"], sample_context
    )
    assert is_valid is False
    assert "certification" in reason.lower() or "unsupported" in reason.lower()


def test_claim_supported_by_another_employee_rejected(sample_context):
    """Evidence claiming records belonging to another employee fails source verification."""
    service = EvaluationDraftAIService(client=MagicMock())
    # Source ID 999 belongs to another employee and is not in approved_sources
    bad_output = EvaluationDraftModelOutput(
        evaluation_narrative="Alice demonstrated solid progress with 90 overall score.",
        strengths=[
            EvaluationStrengthItem(
                title="Performance",
                description="High delivery",
                evidence=[
                    EvidenceItem(
                        source_type="performance",
                        source_id=999,  # Another employee's record
                        claim="Score was 90.0",
                    )
                ],
            )
        ],
        improvement_areas=[
            EvaluationImprovementItem(
                title="Area",
                description="Description",
                evidence=[
                    EvidenceItem(
                        source_type="goal",
                        source_id=20,
                        claim="Migration reached 85% progress",
                    )
                ],
                priority=PriorityLevel.LOW,
            )
        ],
    )

    with pytest.raises(EvaluationEvidenceGroundingError) as exc_info:
        service._validate_model_output(bad_output, sample_context["approved_sources"], sample_context)
    assert "Hallucinated evidence reference" in str(exc_info.value)


def test_claim_based_on_unapproved_record_rejected(sample_context):
    """Records where is_approved=False never enter approved_sources and are rejected."""
    service = EvaluationDraftAIService(client=MagicMock())
    unapproved_id = 9999
    assert ("performance", unapproved_id) not in sample_context["approved_sources"]

    bad_output = EvaluationDraftModelOutput(
        evaluation_narrative="Alice delivered strong results with 90 overall score.",
        strengths=[
            EvaluationStrengthItem(
                title="Performance",
                description="High delivery",
                evidence=[
                    EvidenceItem(
                        source_type="performance",
                        source_id=unapproved_id,
                        claim="Score was 90.0",
                    )
                ],
            )
        ],
        improvement_areas=[
            EvaluationImprovementItem(
                title="Area",
                description="Description",
                evidence=[
                    EvidenceItem(
                        source_type="goal",
                        source_id=20,
                        claim="Migration reached 85% progress",
                    )
                ],
                priority=PriorityLevel.LOW,
            )
        ],
    )

    with pytest.raises(EvaluationEvidenceGroundingError) as exc_info:
        service._validate_model_output(bad_output, sample_context["approved_sources"], sample_context)
    assert "does not exist in approved context" in str(exc_info.value)


def test_multiple_narrative_sentences_one_unsupported_rejects_entire_response(sample_context):
    """If one sentence in the narrative is unsupported, the whole response is rejected."""
    service = EvaluationDraftAIService(client=MagicMock())
    bad_narrative = (
        "Alice delivered strong performance across Q3 2026, achieving a 90 overall score. "
        "She led key technical migrations and demonstrated consistent mentorship. "
        "Additionally, she won an international award and led a global initiative."
    )
    bad_output = EvaluationDraftModelOutput(
        evaluation_narrative=bad_narrative,
        strengths=[
            EvaluationStrengthItem(
                title="High Execution",
                description="Demonstrated high score",
                evidence=[
                    EvidenceItem(
                        source_type="performance",
                        source_id=10,
                        claim="Achieved 90.0 overall score",
                    )
                ],
            )
        ],
        improvement_areas=[
            EvaluationImprovementItem(
                title="Growth",
                description="Description of growth area",
                evidence=[
                    EvidenceItem(
                        source_type="goal",
                        source_id=20,
                        claim="Migration reached 85% progress",
                    )
                ],
                priority=PriorityLevel.LOW,
            )
        ],
    )

    with pytest.raises(EvaluationEvidenceGroundingError) as exc_info:
        service._validate_model_output(bad_output, sample_context["approved_sources"], sample_context)
    assert "award" in str(exc_info.value).lower() or "unsupported" in str(exc_info.value).lower()


def test_existing_valid_evaluation_draft_response_passes(sample_context):
    """Valid draft response passing all checks without errors."""
    service = EvaluationDraftAIService(client=MagicMock())
    valid_output = EvaluationDraftModelOutput(
        evaluation_narrative=(
            "Alice delivered strong performance across Q3 2026, achieving a 90 overall score. "
            "She led key technical migrations on the monolith and demonstrated consistent mentorship."
        ),
        strengths=[
            EvaluationStrengthItem(
                title="High Delivery Quality",
                description="Exceeded targets with top execution scores.",
                evidence=[
                    EvidenceItem(
                        source_type="performance",
                        source_id=10,
                        claim="Achieved overall score of 90.0 and 95% task completion",
                    )
                ],
            )
        ],
        improvement_areas=[
            EvaluationImprovementItem(
                title="Monolith Migration Completion",
                description="Maintain momentum on core monolith migration.",
                evidence=[
                    EvidenceItem(
                        source_type="goal",
                        source_id=20,
                        claim="Monolith migration reached 85% progress",
                    )
                ],
                priority=PriorityLevel.MEDIUM,
            )
        ],
    )

    # Should validate without raising any error
    service._validate_model_output(valid_output, sample_context["approved_sources"], sample_context)


def test_existing_numeric_grounding_with_q2_q3_and_ids_passes(sample_context):
    """Numeric grounding ignores Q2, 2026-Q3, PerformanceRecord #10 while verifying real metrics."""
    service = EvaluationDraftAIService(client=MagicMock())
    valid_output = EvaluationDraftModelOutput(
        evaluation_narrative=(
            "In period 2026-Q3 (reviewed during Q3 2026), employee Alice on PerformanceRecord #10 "
            "achieved a 90.0 overall score and 95.0% task completion rate."
        ),
        strengths=[
            EvaluationStrengthItem(
                title="Strong Delivery",
                description="Excellent score in 2026-Q3",
                evidence=[
                    EvidenceItem(
                        source_type="performance",
                        source_id=10,
                        claim="Overall score was 90.0 in 2026-Q3 on Record #10",
                    )
                ],
            )
        ],
        improvement_areas=[
            EvaluationImprovementItem(
                title="Goal Progress",
                description="Continue progress in 2026-Q3",
                evidence=[
                    EvidenceItem(
                        source_type="goal",
                        source_id=20,
                        claim="Goal #20 migration reached 85.0% progress in Q3",
                    )
                ],
                priority=PriorityLevel.LOW,
            )
        ],
    )

    service._validate_model_output(valid_output, sample_context["approved_sources"], sample_context)


def test_generate_draft_end_to_end_rejects_fabricated_claim_with_opaque_error(sample_context):
    """generate_draft fails closed and returns opaque Reference ID when LLM repeatedly returns fabricated claims."""
    mock_client = MagicMock()
    mock_choice = MagicMock()
    fabricated_payload = {
        "evaluation_narrative": "Won an international award and led a global initiative.",
        "strengths": [
            {
                "title": "Award Strength",
                "description": "Exceeded all expectations with global award.",
                "evidence": [
                    {
                        "source_type": "performance",
                        "source_id": 10,
                        "claim": "Won an international award and led a global initiative.",
                    }
                ],
            }
        ],
        "improvement_areas": [
            {
                "title": "Area",
                "description": "Desc",
                "evidence": [
                    {
                        "source_type": "goal",
                        "source_id": 20,
                        "claim": "Migration reached 85% progress",
                    }
                ],
                "priority": "low",
            }
        ],
    }
    mock_choice.message.content = json.dumps(fabricated_payload)
    mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

    service = EvaluationDraftAIService(client=mock_client)
    with pytest.raises(RuntimeError) as exc_info:
        service.generate_draft(context=sample_context, period="2026-Q3")

    assert "AI service temporarily unavailable. Reference ID:" in str(exc_info.value)

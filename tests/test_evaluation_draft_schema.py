"""Unit tests for Evaluation Draft Assistant Pydantic schemas."""

from datetime import datetime, timezone

import pytest
from pydantic import TypeAdapter, ValidationError

from app.schemas.career_coach import EvidenceItem, PriorityLevel
from app.schemas.evaluation_draft import (
    EvaluationDraftInsufficientDataResponse,
    EvaluationDraftRequest,
    EvaluationDraftResponse,
    EvaluationDraftSuccessResponse,
    EvaluationImprovementItem,
    EvaluationStrengthItem,
)


def test_valid_evaluation_draft_request():
    req = EvaluationDraftRequest(
        employee_id="EMP-001",
        period="2026-Q3",
        evaluation_scores={"overall_score": 88.5, "teamwork": 90.0},
        manager_notes="Strong leadership on platform initiative.",
    )
    assert req.employee_id == "EMP-001"
    assert req.period == "2026-Q3"
    assert req.evaluation_scores["overall_score"] == 88.5
    assert req.manager_notes == "Strong leadership on platform initiative."


def test_minimal_evaluation_draft_request():
    req = EvaluationDraftRequest(
        employee_id="EMP-001",
        period="2026-Q3",
    )
    assert req.employee_id == "EMP-001"
    assert req.period == "2026-Q3"
    assert req.evaluation_scores is None
    assert req.manager_notes is None


def test_invalid_evaluation_draft_request_missing_fields():
    with pytest.raises(ValidationError):
        EvaluationDraftRequest(employee_id="EMP-001")  # missing period

    with pytest.raises(ValidationError):
        EvaluationDraftRequest(period="2026-Q3")  # missing employee_id


def test_invalid_evaluation_scores_out_of_range():
    with pytest.raises(ValidationError) as exc:
        EvaluationDraftRequest(
            employee_id="EMP-001",
            period="2026-Q3",
            evaluation_scores={"overall_score": 105.0},
        )
    assert "Score for 'overall_score' must be between 0.0 and 100.0" in str(exc.value)

    with pytest.raises(ValidationError):
        EvaluationDraftRequest(
            employee_id="EMP-001",
            period="2026-Q3",
            evaluation_scores={"overall_score": -5.0},
        )


def test_extra_fields_forbidden():
    with pytest.raises(ValidationError):
        EvaluationDraftRequest(
            employee_id="EMP-001",
            period="2026-Q3",
            unexpected_field="disallowed",
        )


def test_evaluation_draft_success_response():
    resp = EvaluationDraftSuccessResponse(
        employee_id="EMP-001",
        period="2026-Q3",
        evaluation_narrative="Overall excellent performance during Q3 2026.",
        strengths=[
            EvaluationStrengthItem(
                title="Technical Leadership",
                description="Led the platform re-architecture effectively.",
                evidence=[
                    EvidenceItem(
                        source_type="task_outcome",
                        source_id=1,
                        claim="Optimized p99 latency to 45ms",
                    )
                ],
            )
        ],
        improvement_areas=[
            EvaluationImprovementItem(
                title="Documentation",
                description="Document microservice runbooks more consistently.",
                evidence=[
                    EvidenceItem(
                        source_type="evaluation_theme",
                        source_id=2,
                        claim="Peer review highlighted opportunity for more runbooks",
                    )
                ],
                priority=PriorityLevel.MEDIUM,
            )
        ],
        entered_scores={"overall_score": 90.0},
        created_at=datetime.now(timezone.utc),
    )

    assert resp.status == "success"
    assert resp.human_review_required is True
    assert "draft" in resp.review_disclaimer.lower()
    assert len(resp.strengths) == 1
    assert len(resp.improvement_areas) == 1


def test_evaluation_draft_insufficient_data_response():
    resp = EvaluationDraftInsufficientDataResponse(
        employee_id="EMP-001",
        period="2026-Q3",
        missing_categories=["performance", "goals"],
        created_at=datetime.now(timezone.utc),
    )

    assert resp.status == "insufficient_data"
    assert resp.human_review_required is False
    assert "performance" in resp.missing_categories


def test_discriminated_union_parsing():
    adapter = TypeAdapter(EvaluationDraftResponse)

    success_data = {
        "status": "success",
        "employee_id": "EMP-001",
        "period": "2026-Q3",
        "evaluation_narrative": "Solid quarter with delivery across targets.",
        "strengths": [
            {
                "title": "Reliable Delivery",
                "description": "Met all project deadlines.",
                "evidence": [
                    {"source_type": "goal", "source_id": 1, "claim": "Completed migration"}
                ],
            }
        ],
        "improvement_areas": [
            {
                "title": "Cross-team communication",
                "description": "Increase cross-team updates.",
                "evidence": [
                    {"source_type": "evaluation_theme", "source_id": 2, "claim": "Theme on communication"}
                ],
                "priority": "low",
            }
        ],
        "human_review_required": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    parsed_success = adapter.validate_python(success_data)
    assert isinstance(parsed_success, EvaluationDraftSuccessResponse)

    insufficient_data = {
        "status": "insufficient_data",
        "employee_id": "EMP-001",
        "period": "2026-Q3",
        "missing_categories": ["performance"],
        "message": "Not enough data",
        "human_review_required": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    parsed_insufficient = adapter.validate_python(insufficient_data)
    assert isinstance(parsed_insufficient, EvaluationDraftInsufficientDataResponse)


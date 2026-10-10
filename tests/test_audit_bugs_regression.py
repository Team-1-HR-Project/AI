"""Deterministic regression tests for 3 known audit bugs:

1. Skill Gap grounding must not return/reproduce a 502 because of record/source IDs.
   Example: "Skill #3", "PerformanceRecord #2", or employee IDs must not be treated as business metrics.

2. Team Insight grounding must not return/reproduce a 502 because of period tokens.
   Example: "Q2", "Q3 2026", and "2026-Q3" must not be treated as numeric business metrics.

3. Evaluation Draft must reject a fabricated claim when the source_id is valid but the claim is unsupported by the evidence.
   Example: valid PerformanceRecord #2 + fabricated "won an international award" => reject.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.schemas.career_coach import EvidenceItem, PriorityLevel
from app.schemas.evaluation_draft import (
    EvaluationDraftModelOutput,
    EvaluationImprovementItem,
    EvaluationStrengthItem,
)
from app.schemas.skill_gap import (
    GapSeverity,
    LearningType,
    SkillGapItem,
    SkillGapModelOutput,
    SkillRecommendationItem,
)
from app.schemas.team_insight import (
    RecommendedManagementAction,
)
from app.services.evaluation_draft_ai import (
    EvaluationDraftAIService,
    EvaluationEvidenceGroundingError,
)
from app.services.skill_gap_ai import (
    SkillGapAIService,
)
from app.services.skill_gap_ai import (
    _extract_numbers_from_text as skill_gap_extract_numbers,
)
from app.services.team_insight_ai import (
    TeamInsightAIService,
    TeamInsightModelOutput,
)
from app.services.team_insight_ai import (
    _extract_numbers_from_text as team_insight_extract_numbers,
)

# ==============================================================================
# Bug 1: Skill Gap Record & Source IDs
# ==============================================================================


def test_skill_gap_grounding_ignores_record_ids():
    """Skill Gap grounding must not treat record/source IDs like 'Skill #3' or

    'PerformanceRecord #2' as business metrics, preventing false 502 / grounding failures.
    """
    # 1. Verify text extractor ignores record IDs and employee IDs
    text = (
        "Evidence from Skill #3 and PerformanceRecord #2 for employee EMP-101 "
        "confirms level 3 proficiency with 88.0 overall score."
    )
    extracted = skill_gap_extract_numbers(text)
    assert extracted == [3.0, 88.0]
    assert 2.0 not in extracted  # From PerformanceRecord #2
    assert 101.0 not in extracted  # From EMP-101

    # 2. Verify evidence grounding validation in SkillGapAIService succeeds
    approved_sources = {
        ("skill", 3): {
            "id": 3,
            "source_type": "skill",
            "skill_name": "Python",
            "current_level": 3,
            "verified": True,
        },
        ("performance", 2): {
            "id": 2,
            "source_type": "performance",
            "overall_score": 88.0,
            "task_completion_rate": 92.0,
        },
    }
    output = SkillGapModelOutput(
        skill_gaps=[
            SkillGapItem(
                skill_name="Python",
                current_level="Intermediate",
                desired_level="Advanced",
                gap_severity=GapSeverity.MEDIUM,
                rationale="Skill #3 shows current level is 3, while role requires advanced proficiency.",
                evidence=[
                    EvidenceItem(
                        source_type="skill",
                        source_id=3,
                        claim="Skill #3 confirms current verified skill proficiency level is 3.",
                    ),
                    EvidenceItem(
                        source_type="performance",
                        source_id=2,
                        claim="PerformanceRecord #2 demonstrates an overall score of 88.0 and 92.0% completion.",
                    ),
                ],
            )
        ],
        recommendations=[
            SkillRecommendationItem(
                title="Advanced Python Patterns",
                learning_type=LearningType.TRAINING_COURSE,
                focus_skill="Python",
                description="Complete deep-dive modules on concurrency and design patterns.",
                expected_outcome="Demonstrate advanced async patterns in production.",
                measurable_target="Complete 3 capstone projects and pass assessment with score > 85.",
                timeline="6 weeks",
                priority=PriorityLevel.HIGH,
            )
        ],
    )
    service = SkillGapAIService(client=MagicMock())
    # Must succeed without raising SkillGapAIServiceError
    service._validate_evidence_grounding(output, approved_sources)


# ==============================================================================
# Bug 2: Team Insight Period Tokens
# ==============================================================================


def test_team_insight_grounding_ignores_period_tokens():
    """Team Insight grounding must not treat period tokens like 'Q2', 'Q3 2026',

    and '2026-Q3' as numeric business metrics, preventing false 502 / grounding failures.
    """
    # 1. Verify text extractor ignores period tokens
    text = "In Q2, Q3 2026, and 2026-Q3, the team maintained an average task completion rate of 95.0%."
    extracted = team_insight_extract_numbers(text)
    assert extracted == [95.0]
    assert 2.0 not in extracted  # From Q2
    assert 3.0 not in extracted  # From Q3 2026
    assert 2026.0 not in extracted  # From Q3 2026 / 2026-Q3

    # 2. Verify model output validation in TeamInsightAIService succeeds
    context = {
        "department": "Engineering",
        "period": "2026-Q3",
        "has_sufficient_data": True,
        "completion_trends": {
            "direction": "improved",
            "team_avg_task_completion": 95.0,
        },
        "overdue_workload": {
            "overdue_count": 0,
        },
        "skill_patterns": {
            "top_common_skills": ["Python"],
        },
        "evaluation_theme_patterns": {
            "top_positive_themes": ["Mentorship"],
            "top_needs_improvement_themes": ["Testing"],
        },
        "grounding_registry": {
            "skill_frequencies": {"Python": 5},
            "positive_theme_frequencies": {"Mentorship": 4},
            "needs_improvement_theme_frequencies": {"Testing": 2},
        },
        "drill_down_factors": [],
    }
    output = TeamInsightModelOutput(
        executive_summary="During Q2, Q3 2026, and 2026-Q3, the team maintained a 95.0% task completion rate with improved progress.",
        overdue_workload_summary="There are no overdue tasks currently blocking team delivery.",
        completion_trends_summary="Completion trends improved across Q2 and 2026-Q3 with 95.0% velocity.",
        skill_gap_summary="The team shows high proficiency in Python across all sprints.",
        top_common_gaps=["Python"],
        evaluation_theme_summary="Recurring positive evaluation themes highlight Mentorship across the department.",
        top_positive_themes=["Mentorship"],
        top_needs_improvement_themes=["Testing"],
        drill_down_factors=[],
        recommended_management_actions=[
            RecommendedManagementAction(
                action_title="Maintain Sprint Velocity",
                description="Continue weekly review checkpoints and unblock dependency bottlenecks.",
                priority=PriorityLevel.MEDIUM,
            )
        ],
    )
    service = TeamInsightAIService(client=MagicMock())
    # Must succeed without raising TeamInsightAIServiceError
    service._validate_model_output(output, context)


# ==============================================================================
# Bug 3: Evaluation Draft Unsupported Fabricated Claim
# ==============================================================================


def test_evaluation_draft_rejects_unsupported_fabricated_claim_with_valid_source_id():
    """Evaluation Draft must reject a fabricated claim even when citing a valid source_id."""
    # Context contains valid PerformanceRecord #2 with overall_score=90.0, task_completion_rate=95.0
    context = {
        "employee": {
            "id": "EMP-001",
            "first_name": "Alice",
            "last_name": "Smith",
            "role_title": "Software Engineer",
            "department": "Engineering",
        },
        "period": "2026-Q3",
        "has_sufficient_data": True,
        "performance": [
            {
                "id": 2,
                "overall_score": 90.0,
                "task_completion_rate": 95.0,
                "period": "2026-Q3",
            }
        ],
        "approved_sources": {
            ("performance", 2): {
                "id": 2,
                "source_type": "performance",
                "overall_score": 90.0,
                "task_completion_rate": 95.0,
                "period": "2026-Q3",
            }
        },
    }
    # Model output cites valid source ("performance", 2), but introduces a fabricated claim ("won an international award")
    output = EvaluationDraftModelOutput(
        evaluation_narrative="Alice demonstrated solid technical performance with an overall score of 90. She won an international award and led global initiatives.",
        strengths=[
            EvaluationStrengthItem(
                title="Outstanding Achievement",
                description="Recognized internationally for exceptional technical contributions.",
                evidence=[
                    EvidenceItem(
                        source_type="performance",
                        source_id=2,
                        claim="PerformanceRecord #2 demonstrates Alice won an international award and led global initiatives.",
                    )
                ],
            )
        ],
        improvement_areas=[
            EvaluationImprovementItem(
                title="Documentation Cadence",
                description="Ensure design documents are submitted earlier in the sprint cycle.",
                evidence=[
                    EvidenceItem(
                        source_type="performance",
                        source_id=2,
                        claim="PerformanceRecord #2 records a task completion rate of 95.0%.",
                    )
                ],
                priority=PriorityLevel.LOW,
            )
        ],
    )
    service = EvaluationDraftAIService(client=MagicMock())
    with pytest.raises(EvaluationEvidenceGroundingError) as exc_info:
        service._validate_model_output(
            output,
            approved_sources=context["approved_sources"],
            context=context,
        )

    err_msg = str(exc_info.value).lower()
    assert "award" in err_msg or "unsupported" in err_msg or "grounding" in err_msg


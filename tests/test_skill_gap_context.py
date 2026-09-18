"""Unit and safety tests for AI #5: Skill-Gap & Development Recommendations.

Covers:
- Approved-only context filtering and employee isolation
- Context character budgets and pruning behavior
- Skill gap recommendation validation
- measurable_target validation (concrete vs generic target rejection)
- Grounding: ungrounded numbers, source type mismatches, missing sources
- Prohibited employment decisions (hiring, firing, promotion, salary, PIP)
- Insufficient-data behavior (pre-LLM fail-closed)
- Prompt injection defense (sanitization and delimiter escape prevention)
"""

import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.session import Base
from app.models import (
    Employee,
    EvaluationTheme,
    Goal,
    PerformanceRecord,
    Skill,
    TaskOutcome,
)
from app.schemas.skill_gap import (
    SkillGapModelOutput,
    SkillRecommendationItem,
)
from app.services.skill_gap_ai import (
    SkillGapAIService,
    SkillGapAIServiceError,
    _extract_numbers_from_text,
    _sanitize_untrusted_text,
)
from app.services.skill_gap_context import (
    SkillGapContextBuilder,
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


@pytest.fixture
def seed_data(db_session):
    """Seed Alice with approved and unapproved records, plus Bob for isolation."""
    alice = Employee(
        id="EMP-ALICE",
        first_name="Alice",
        last_name="Architect",
        role_title="Backend Engineer",
        department="Engineering",
    )
    bob = Employee(
        id="EMP-BOB",
        first_name="Bob",
        last_name="Sales",
        role_title="Sales Rep",
        department="Sales",
    )
    db_session.add_all([alice, bob])
    db_session.commit()

    # Approved records for Alice
    s1 = Skill(
        employee_id="EMP-ALICE",
        name="Python Backend",
        level="Intermediate",
        evidence="Builds microservices in FastAPI",
        is_approved=True,
    )
    p1 = PerformanceRecord(
        employee_id="EMP-ALICE",
        period="2026-Q3",
        overall_score=88.0,
        task_completion_rate=92.0,
        goal_achievement_rate=85.0,
        attendance_rate=98.0,
        is_approved=True,
    )
    g1 = Goal(
        employee_id="EMP-ALICE",
        title="Migrate Cache Infrastructure",
        progress=60.0,
        status="in_progress",
        period="2026-Q3",
        is_approved=True,
    )
    t1 = TaskOutcome(
        employee_id="EMP-ALICE",
        title="Redis cluster cutover",
        status="completed",
        outcome="Latency reduced by 40%",
        period="2026-Q3",
        is_approved=True,
    )
    th1 = EvaluationTheme(
        employee_id="EMP-ALICE",
        theme="System Architecture",
        sentiment="needs_improvement",
        evidence="Needs deeper knowledge of distributed transactions and event sourcing",
        period="2026-Q3",
        is_approved=True,
    )

    # Unapproved records for Alice (must be completely ignored)
    s_unapproved = Skill(
        employee_id="EMP-ALICE",
        name="Unapproved Deep Learning",
        level="Expert",
        evidence="Self-reported unapproved claim",
        is_approved=False,
    )
    p_unapproved = PerformanceRecord(
        employee_id="EMP-ALICE",
        period="2026-Q4",
        overall_score=99.0,
        task_completion_rate=100.0,
        goal_achievement_rate=100.0,
        attendance_rate=100.0,
        is_approved=False,
    )

    # Bob records (for cross-employee isolation)
    s_bob = Skill(
        employee_id="EMP-BOB",
        name="Enterprise Sales",
        level="Expert",
        evidence="Closing deals",
        is_approved=True,
    )

    db_session.add_all([s1, p1, g1, t1, th1, s_unapproved, p_unapproved, s_bob])
    db_session.commit()
    return alice, bob


# -----------------------------------------------------------------------------
# 1. Context Builder Unit Tests: Approved-only filtering & Employee Isolation
# -----------------------------------------------------------------------------

def test_skill_gap_context_approved_only_filtering(db_session, seed_data):
    """Context builder must exclude unapproved records and enforce strict employee boundary."""
    res = SkillGapContextBuilder.build_context(db_session, employee_id="EMP-ALICE", period="2026-Q3")
    assert res["has_sufficient_data"] is True
    context = res["context"]

    # Only approved skills
    skill_names = [s["name"] for s in context["skills"]]
    assert "Python Backend" in skill_names
    assert "Unapproved Deep Learning" not in skill_names
    assert "Enterprise Sales" not in skill_names

    # Approved sources dictionary check
    approved_sources = res["approved_sources"]
    assert ("skill", 1) in approved_sources
    # Unapproved records are not in approved_sources
    for source_key in approved_sources:
        assert source_key != ("skill", 2)

    # No cross-employee data
    context_str = json.dumps(context)
    assert "EMP-BOB" not in context_str
    assert "Enterprise Sales" not in context_str


def test_skill_gap_context_insufficient_data_when_no_skills(db_session):
    """If an employee has no approved skills, context builder returns has_sufficient_data=False."""
    emp = Employee(
        id="EMP-NO-SKILLS",
        first_name="No",
        last_name="Skills",
        role_title="Intern",
        department="Engineering",
    )
    db_session.add(emp)
    db_session.commit()

    res = SkillGapContextBuilder.build_context(db_session, employee_id="EMP-NO-SKILLS")
    assert res["has_sufficient_data"] is False
    assert "skills" in res["missing_categories"]


def test_skill_gap_context_budget_enforcement(db_session, seed_data):
    """enforce_total_context_limit trims older supporting items while preserving skills."""
    res = SkillGapContextBuilder.build_context(db_session, employee_id="EMP-ALICE", period="2026-Q3")
    context = res["context"]
    # Enforce a budget smaller than original context length
    orig_len = len(json.dumps(context))
    trimmed, _, _ = SkillGapContextBuilder.enforce_total_context_limit(context, max_chars=orig_len - 100)
    # Skills must be preserved
    assert len(trimmed["skills"]) >= 1


# -----------------------------------------------------------------------------
# 2. AI Service Unit Tests: measurable_target Validation
# -----------------------------------------------------------------------------

@pytest.mark.parametrize(
    "generic_target",
    [
        "Improve your skills",
        "Enhance their performance",
        "Develop abilities.",
        "Grow knowledge",
        "Strengthen competencies!",
        "Become better at distributed architecture",
        "Learn more about event sourcing",
        "Work harder.",
        "Do better.",
        "Keep learning",
        "Try harder",
    ],
)
def test_skill_gap_rejects_generic_measurable_target(generic_target):
    """Safety policy must reject vague or generic targets that lack concrete deliverables."""
    service = SkillGapAIService(api_key="mock_key")
    output = SkillGapModelOutput(
        status="success",
        skill_gaps=[],
        recommendations=[
            SkillRecommendationItem(
                title="Study Distributed Systems",
                learning_type="self_paced_study",
                focus_skill="Distributed Systems",
                description="Read through architecture guides.",
                expected_outcome="Better system design knowledge.",
                measurable_target=generic_target,
                timeline="4 weeks",
                priority="high",
            )
        ],
    )

    with pytest.raises(SkillGapAIServiceError) as exc_info:
        service._validate_safety_policy(output)

    assert "too generic" in str(exc_info.value)


def test_skill_gap_accepts_concrete_measurable_target():
    """Safety policy must accept concrete measurable targets with deliverables, numbers, or deadlines."""
    service = SkillGapAIService(api_key="mock_key")
    output = SkillGapModelOutput(
        status="success",
        skill_gaps=[],
        recommendations=[
            SkillRecommendationItem(
                title="Distributed Systems Masterclass",
                learning_type="training_course",
                focus_skill="Distributed Systems",
                description="Study saga pattern implementation.",
                expected_outcome="Complete transaction engine.",
                measurable_target="Complete course and deliver a working 2-phase commit prototype by sprint 4",
                timeline="6 weeks",
                priority="high",
            )
        ],
    )
    # Should not raise any error
    service._validate_safety_policy(output)


# -----------------------------------------------------------------------------
# 3. AI Service Unit Tests: Prohibited Employment Decisions
# -----------------------------------------------------------------------------

@pytest.mark.parametrize(
    "prohibited_term",
    [
        "Recommend immediate promotion to Staff Architect",
        "Terminate employment due to lack of skills",
        "Fire employee if goals are not met",
        "Give 15% salary increase upon completion",
        "Place employee on a formal disciplinary PIP",
        "Issue suspension pending review",
        "Offer equity grant and cash bonus",
    ],
)
def test_skill_gap_rejects_prohibited_employment_decisions(prohibited_term):
    """Service strictly rejects binding employment, compensation, or disciplinary decisions."""
    service = SkillGapAIService(api_key="mock_key")
    output = SkillGapModelOutput(
        status="success",
        skill_gaps=[],
        recommendations=[
            SkillRecommendationItem(
                title="Career Action",
                learning_type="mentorship",
                focus_skill="General",
                description=prohibited_term,
                expected_outcome="Outcome",
                measurable_target="Complete 3 mentoring sessions by week 4",
                timeline="4 weeks",
                priority="medium",
            )
        ],
    )

    with pytest.raises(SkillGapAIServiceError) as exc_info:
        service._validate_safety_policy(output)

    assert "Output safety policy violation" in str(exc_info.value)


# -----------------------------------------------------------------------------
# 4. Grounding Validation Unit Tests: Evidence, Numbers, Sources
# -----------------------------------------------------------------------------

def test_skill_gap_grounding_rejects_missing_source():
    """Grounding fails if model cites a source not present in approved_sources."""
    service = SkillGapAIService(api_key="mock_key")
    approved_sources = {
        ("skill", 1): {"source_type": "skill", "id": 1, "name": "Python", "level": "Intermediate"}
    }
    raw_json = {
        "status": "success",
        "skill_gaps": [
            {
                "skill_name": "Distributed Systems",
                "current_level": "Intermediate",
                "desired_level": "Advanced",
                "gap_severity": "high",
                "rationale": "Needs improvement",
                "evidence": [
                    {
                        "source_type": "performance",
                        "source_id": 999,  # Missing source
                        "claim": "Score was 88.0",
                    }
                ],
            }
        ],
        "recommendations": [],
    }
    output = SkillGapModelOutput.model_validate(raw_json)

    with pytest.raises(SkillGapAIServiceError) as exc_info:
        service._validate_evidence_grounding(output, approved_sources)

    assert "does not exist in the approved context" in str(exc_info.value)


def test_skill_gap_grounding_rejects_invented_number():
    """Grounding fails if evidence claim includes a number not in the referenced source."""
    service = SkillGapAIService(api_key="mock_key")
    approved_sources = {
        ("performance", 1): {
            "source_type": "performance",
            "id": 1,
            "overall_score": 88.0,
            "task_completion_rate": 92.0,
        }
    }
    raw_json = {
        "status": "success",
        "skill_gaps": [
            {
                "skill_name": "FastAPI",
                "current_level": "Intermediate",
                "desired_level": "Advanced",
                "gap_severity": "low",
                "rationale": "Good progress",
                "evidence": [
                    {
                        "source_type": "performance",
                        "source_id": 1,
                        "claim": "Achieved overall score of 99.5 in 2026-Q3",  # 99.5 is invented!
                    }
                ],
            }
        ],
        "recommendations": [],
    }
    output = SkillGapModelOutput.model_validate(raw_json)

    with pytest.raises(SkillGapAIServiceError) as exc_info:
        service._validate_evidence_grounding(output, approved_sources)

    assert "numeric value '99.5'" in str(exc_info.value)
    assert "does not match source record" in str(exc_info.value)


def test_skill_gap_grounding_rejects_source_type_mismatch():
    """Grounding fails if model cites a skill record with source_type='performance'."""
    service = SkillGapAIService(api_key="mock_key")
    approved_sources = {
        ("performance", 1): {
            "source_type": "performance",
            "id": 1,
            "overall_score": 88.0,
        }
    }
    raw_json = {
        "status": "success",
        "skill_gaps": [
            {
                "skill_name": "FastAPI",
                "current_level": "Intermediate",
                "desired_level": "Advanced",
                "gap_severity": "low",
                "rationale": "Good progress",
                "evidence": [
                    {
                        "source_type": "skill",  # Mismatch: record is performance
                        "source_id": 1,
                        "claim": "Overall score of 88.0",
                    }
                ],
            }
        ],
        "recommendations": [],
    }
    output = SkillGapModelOutput.model_validate(raw_json)

    with pytest.raises(SkillGapAIServiceError) as exc_info:
        service._validate_evidence_grounding(output, approved_sources)

    assert "does not exist in the approved context" in str(exc_info.value)


# -----------------------------------------------------------------------------
# 5. Prompt Injection & Sanitization Unit Tests
# -----------------------------------------------------------------------------

def test_sanitize_untrusted_text():
    """Sanitizer replaces angle brackets with brackets and removes non-printable control chars."""
    raw = "<script>alert('xss');</script>\x00\x08Hello"
    sanitized = _sanitize_untrusted_text(raw)
    assert "<" not in sanitized
    assert ">" not in sanitized
    assert "[script]" in sanitized
    assert "\x00" not in sanitized
    assert "Hello" in sanitized


def test_extract_numbers_from_text():
    """_extract_numbers_from_text extracts numbers while ignoring dates and quarters."""
    assert _extract_numbers_from_text("Q3 2026") == []
    assert _extract_numbers_from_text("2026-Q3") == []
    assert _extract_numbers_from_text("2026-09-18") == []
    assert _extract_numbers_from_text("Achieved 88.0% score in Q3 2026") == [88.0]

"""Unit tests for TeamInsightContextBuilder (Feature #7: Team Insight Summary - Manager).

Verifies:
1. Department filtering
2. Team isolation (other departments completely excluded)
3. Approved-only filtering (unapproved records excluded from metrics, counts, and summaries)
4. Team size (count of approved members in the department)
5. Blocked task count
6. Delayed goal count
7. Affected member count (unique members with >= 1 blocked task or delayed goal)
8. Team performance averages (exact arithmetic means)
9. Missing metrics are not treated as zero
10. Previous-period trend calculation
11. No previous period handling
12. Common skill aggregation
13. Positive evaluation themes
14. Needs-improvement evaluation themes
15. Drill-down factors
16. No employee PII leakage in team context
17. Unknown department
18. No approved records
19. Explicit period
20. Latest-period selection when period is omitted
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
from app.schemas.performance_insight import TrendDirection
from app.services.team_insight_context import TeamInsightContextBuilder

TEST_DATABASE_URL = "sqlite:///:memory:"

test_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture(scope="function")
def db():
    """Isolated in-memory SQLite database session for each test."""
    Base.metadata.create_all(bind=test_engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=test_engine)


@pytest.fixture
def seed_team_data(db):
    """Seed employees and approved/unapproved records across Engineering and Sales."""
    # Engineering team members
    alice = Employee(
        id="EMP-ENG-ALICE",
        first_name="Alice",
        last_name="Architect",
        role_title="Backend Lead",
        department="Engineering",
    )
    bob = Employee(
        id="EMP-ENG-BOB",
        first_name="Bob",
        last_name="Builder",
        role_title="DevOps Engineer",
        department="Engineering",
    )
    charlie = Employee(
        id="EMP-ENG-CHARLIE",
        first_name="Charlie",
        last_name="Coder",
        role_title="Junior Developer",
        department="Engineering",
    )
    # Sales team member (for team isolation tests)
    sarah = Employee(
        id="EMP-SALES-SARAH",
        first_name="Sarah",
        last_name="Seller",
        role_title="Account Executive",
        department="Sales",
    )
    db.add_all([alice, bob, charlie, sarah])
    db.commit()

    # 1. Performance Records (Engineering: 2026-Q2 and 2026-Q3)
    # Alice Q2 & Q3
    p_alice_q2 = PerformanceRecord(
        employee_id="EMP-ENG-ALICE",
        period="2026-Q2",
        overall_score=85.0,
        task_completion_rate=90.0,
        goal_achievement_rate=80.0,
        attendance_rate=95.0,
        is_approved=True,
    )
    p_alice_q3 = PerformanceRecord(
        employee_id="EMP-ENG-ALICE",
        period="2026-Q3",
        overall_score=95.0,
        task_completion_rate=98.0,
        goal_achievement_rate=92.0,
        attendance_rate=98.0,
        is_approved=True,
    )

    # Bob Q2 & Q3
    p_bob_q2 = PerformanceRecord(
        employee_id="EMP-ENG-BOB",
        period="2026-Q2",
        overall_score=80.0,
        task_completion_rate=80.0,
        goal_achievement_rate=70.0,
        attendance_rate=90.0,
        is_approved=True,
    )
    p_bob_q3 = PerformanceRecord(
        employee_id="EMP-ENG-BOB",
        period="2026-Q3",
        overall_score=85.0,
        task_completion_rate=92.0,
        goal_achievement_rate=88.0,
        attendance_rate=92.0,
        is_approved=True,
    )

    # Charlie Q3
    p_charlie_q3 = PerformanceRecord(
        employee_id="EMP-ENG-CHARLIE",
        period="2026-Q3",
        overall_score=75.0,
        task_completion_rate=80.0,
        goal_achievement_rate=90.0,
        attendance_rate=90.0,
        is_approved=True,
    )

    # Charlie Unapproved Q3 record (must be excluded)
    p_charlie_unapproved = PerformanceRecord(
        employee_id="EMP-ENG-CHARLIE",
        period="2026-Q3",
        overall_score=10.0,
        task_completion_rate=10.0,
        goal_achievement_rate=10.0,
        attendance_rate=10.0,
        is_approved=False,
    )

    # Sarah Sales Q3 Record (Sales: must be isolated out)
    p_sarah_q3 = PerformanceRecord(
        employee_id="EMP-SALES-SARAH",
        period="2026-Q3",
        overall_score=60.0,
        task_completion_rate=50.0,
        goal_achievement_rate=50.0,
        attendance_rate=80.0,
        is_approved=True,
    )

    # 2. Tasks (Approved and Unapproved)
    t_alice_blocked = TaskOutcome(
        employee_id="EMP-ENG-ALICE",
        title="Async Queue Cutover",
        status="blocked",
        outcome="Blocked pending IAM security approval",
        period="2026-Q3",
        is_approved=True,
    )
    t_bob_completed = TaskOutcome(
        employee_id="EMP-ENG-BOB",
        title="Deploy CI/CD runner",
        status="completed",
        outcome="Runner pipeline operational",
        period="2026-Q3",
        is_approved=True,
    )
    t_bob_blocked = TaskOutcome(
        employee_id="EMP-ENG-BOB",
        title="Database failover test",
        status="blocked",
        outcome="Blocked on storage provisioning",
        period="2026-Q3",
        is_approved=True,
    )
    t_unapproved_blocked = TaskOutcome(
        employee_id="EMP-ENG-CHARLIE",
        title="Secret project cutover",
        status="blocked",
        outcome="Unapproved blocker",
        period="2026-Q3",
        is_approved=False,
    )
    t_sales_blocked = TaskOutcome(
        employee_id="EMP-SALES-SARAH",
        title="Sales contract signature",
        status="blocked",
        outcome="Blocked on legal",
        period="2026-Q3",
        is_approved=True,
    )

    # 3. Goals (Approved and Unapproved)
    g_bob_delayed = Goal(
        employee_id="EMP-ENG-BOB",
        title="Upgrade Kubernetes clusters",
        status="delayed",
        progress=45.0,
        period="2026-Q3",
        is_approved=True,
    )
    g_charlie_delayed = Goal(
        employee_id="EMP-ENG-CHARLIE",
        title="Automate integration tests",
        status="delayed",
        progress=30.0,
        period="2026-Q3",
        is_approved=True,
    )
    g_unapproved_delayed = Goal(
        employee_id="EMP-ENG-ALICE",
        title="Unapproved Draft Goal",
        status="delayed",
        progress=10.0,
        period="2026-Q3",
        is_approved=False,
    )
    g_sales_delayed = Goal(
        employee_id="EMP-SALES-SARAH",
        title="Close Q3 enterprise deals",
        status="delayed",
        progress=20.0,
        period="2026-Q3",
        is_approved=True,
    )

    # 4. Skills
    s_alice1 = Skill(
        employee_id="EMP-ENG-ALICE",
        name="Python Backend",
        level="Expert",
        is_approved=True,
    )
    s_alice2 = Skill(
        employee_id="EMP-ENG-ALICE",
        name="Distributed Systems",
        level="Advanced",
        is_approved=True,
    )
    s_bob1 = Skill(
        employee_id="EMP-ENG-BOB",
        name="Python Backend",
        level="Intermediate",
        is_approved=True,
    )
    s_bob2 = Skill(
        employee_id="EMP-ENG-BOB",
        name="Kubernetes",
        level="Expert",
        is_approved=True,
    )
    s_charlie1 = Skill(
        employee_id="EMP-ENG-CHARLIE",
        name="Python Backend",
        level="Beginner",
        is_approved=True,
    )
    s_unapproved = Skill(
        employee_id="EMP-ENG-ALICE",
        name="Quantum Computing",
        level="Expert",
        is_approved=False,
    )
    s_sales = Skill(
        employee_id="EMP-SALES-SARAH",
        name="Enterprise Sales",
        level="Expert",
        is_approved=True,
    )

    # 5. Evaluation Themes
    th_alice_pos = EvaluationTheme(
        employee_id="EMP-ENG-ALICE",
        theme="Technical Leadership",
        sentiment="positive",
        evidence="Exemplary leadership on architecture.",
        period="2026-Q3",
        is_approved=True,
    )
    th_bob_pos = EvaluationTheme(
        employee_id="EMP-ENG-BOB",
        theme="Technical Leadership",
        sentiment="positive",
        evidence="Guided team during production incidents.",
        period="2026-Q3",
        is_approved=True,
    )
    th_bob_needs_imp = EvaluationTheme(
        employee_id="EMP-ENG-BOB",
        theme="Documentation",
        sentiment="needs_improvement",
        evidence="Runbooks lack sequence diagrams.",
        period="2026-Q3",
        is_approved=True,
    )
    th_charlie_needs_imp = EvaluationTheme(
        employee_id="EMP-ENG-CHARLIE",
        theme="Documentation",
        sentiment="needs_improvement",
        evidence="Needs to write cleaner API docs.",
        period="2026-Q3",
        is_approved=True,
    )
    th_unapproved = EvaluationTheme(
        employee_id="EMP-ENG-ALICE",
        theme="Fake Theme",
        sentiment="needs_improvement",
        evidence="Unapproved note",
        period="2026-Q3",
        is_approved=False,
    )
    th_sales = EvaluationTheme(
        employee_id="EMP-SALES-SARAH",
        theme="Cold Calling",
        sentiment="positive",
        evidence="Strong sales outreach",
        period="2026-Q3",
        is_approved=True,
    )

    db.add_all([
        p_alice_q2, p_alice_q3,
        p_bob_q2, p_bob_q3,
        p_charlie_q3, p_charlie_unapproved,
        p_sarah_q3,
        t_alice_blocked, t_bob_completed, t_bob_blocked, t_unapproved_blocked, t_sales_blocked,
        g_bob_delayed, g_charlie_delayed, g_unapproved_delayed, g_sales_delayed,
        s_alice1, s_alice2, s_bob1, s_bob2, s_charlie1, s_unapproved, s_sales,
        th_alice_pos, th_bob_pos, th_bob_needs_imp, th_charlie_needs_imp, th_unapproved, th_sales,
    ])
    db.commit()


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

def test_team_insight_context_team_size(db, seed_team_data):
    """Team size must equal exactly the count of employees in the requested department."""
    res = TeamInsightContextBuilder.build_context(db, department="Engineering", period="2026-Q3")
    assert res["has_sufficient_data"] is True
    # Alice, Bob, Charlie = 3
    assert res["team_size"] == 3
    assert res["grounding_registry"]["team_size"] == 3


def test_team_insight_context_department_isolation(db, seed_team_data):
    """Data from other departments (Sales) must never leak into Engineering context."""
    res = TeamInsightContextBuilder.build_context(db, department="Engineering", period="2026-Q3")
    context_str = json.dumps(res)

    # Verify Sarah's records / skills / themes do not appear
    assert "Sales" not in res["grounding_registry"]["department"]
    assert "Enterprise Sales" not in context_str
    assert "Cold Calling" not in context_str
    assert "Sales contract signature" not in context_str


def test_team_insight_context_approved_only_filtering(db, seed_team_data):
    """Unapproved records must be completely excluded from metrics, counts, and themes."""
    res = TeamInsightContextBuilder.build_context(db, department="Engineering", period="2026-Q3")

    # Unapproved skill 'Quantum Computing' must be absent
    assert "Quantum Computing" not in res["skill_patterns"]["top_common_skills"]
    assert "Quantum Computing" not in res["grounding_registry"]["skill_frequencies"]

    # Unapproved theme 'Fake Theme' must be absent
    assert "Fake Theme" not in res["evaluation_theme_patterns"]["top_needs_improvement_themes"]

    # Unapproved blocked task 'Secret project cutover' must not be counted
    blocked_titles = [t["title"] for t in res["workload_patterns"]["blocked_task_samples"]]
    assert "Secret project cutover" not in blocked_titles


def test_team_insight_context_blocked_task_count(db, seed_team_data):
    """Blocked tasks count only approved blocked tasks for department members."""
    res = TeamInsightContextBuilder.build_context(db, department="Engineering", period="2026-Q3")
    # Alice has 1 blocked task, Bob has 1 blocked task -> 2 total
    assert res["workload_patterns"]["total_blocked_tasks"] == 2
    assert res["grounding_registry"]["total_blocked_tasks"] == 2


def test_team_insight_context_delayed_goal_count(db, seed_team_data):
    """Delayed goals count only approved delayed goals for department members."""
    res = TeamInsightContextBuilder.build_context(db, department="Engineering", period="2026-Q3")
    # Bob has 1 delayed goal, Charlie has 1 delayed goal -> 2 total
    assert res["workload_patterns"]["total_delayed_goals"] == 2
    assert res["grounding_registry"]["total_delayed_goals"] == 2


def test_team_insight_context_affected_member_count(db, seed_team_data):
    """Affected member count equals unique employees with >=1 blocked task or delayed goal."""
    res = TeamInsightContextBuilder.build_context(db, department="Engineering", period="2026-Q3")
    # Alice (1 blocked task), Bob (1 blocked task + 1 delayed goal), Charlie (1 delayed goal)
    # Unique members = 3
    assert res["workload_patterns"]["affected_member_count"] == 3
    assert res["grounding_registry"]["affected_member_count"] == 3


def test_team_insight_context_performance_averages(db, seed_team_data):
    """Arithmetic means are computed accurately across approved records in target period."""
    res = TeamInsightContextBuilder.build_context(db, department="Engineering", period="2026-Q3")
    # Alice:   95.0 overall, 98.0 task, 92.0 goal
    # Bob:     85.0 overall, 92.0 task, 88.0 goal
    # Charlie: 75.0 overall, 80.0 task, 90.0 goal
    # Overall avg: (95 + 85 + 75) / 3 = 85.0
    # Task avg:    (98 + 92 + 80) / 3 = 90.0
    # Goal avg:    (92 + 88 + 90) / 3 = 90.0
    assert res["completion_trends"]["team_avg_overall_score"] == 85.0
    assert res["completion_trends"]["team_avg_task_completion"] == 90.0
    assert res["completion_trends"]["team_avg_goal_achievement"] == 90.0


def test_team_insight_context_missing_metrics_not_treated_as_zero(db, seed_team_data):
    """When a performance metric is None, it is excluded from denominator rather than treated as 0."""
    from unittest.mock import MagicMock

    mock_records = [
        MagicMock(task_completion_rate=90.0, goal_achievement_rate=90.0, overall_score=85.0),
        MagicMock(task_completion_rate=80.0, goal_achievement_rate=None, overall_score=75.0),
    ]
    averages = TeamInsightContextBuilder.compute_metric_averages(mock_records)
    # If None was treated as 0: (90 + 0) / 2 = 45.0.
    # Excluded from denominator: 90 / 1 = 90.0.
    assert averages["team_avg_goal_achievement"] == 90.0
    assert averages["team_avg_goal_achievement"] != 45.0
    assert averages["team_avg_task_completion"] == 85.0
    assert averages["team_avg_overall_score"] == 80.0


def test_team_insight_context_period_over_period_trend(db, seed_team_data):
    """When comparison period exists (2026-Q2), trend direction is deterministically calculated."""
    res = TeamInsightContextBuilder.build_context(db, department="Engineering", period="2026-Q3")
    assert res["comparison_period"] == "2026-Q2"
    # Q2 overall: Alice (85.0) + Bob (80.0) -> avg 82.5
    # Q3 overall: Alice (95.0) + Bob (85.0) + Charlie (75.0) -> avg 85.0
    # Direction: 85.0 > 82.5 -> improved
    assert res["completion_trends"]["direction"] == TrendDirection.IMPROVED.value
    assert res["grounding_registry"]["direction"] == "improved"


def test_team_insight_context_no_previous_period(db, seed_team_data):
    """When evaluating oldest period (2026-Q2), comparison_period is None and direction is stable."""
    res = TeamInsightContextBuilder.build_context(db, department="Engineering", period="2026-Q2")
    assert res["comparison_period"] is None
    assert res["completion_trends"]["direction"] == TrendDirection.STABLE.value
    assert res["grounding_registry"]["comparison_averages"] is None


def test_team_insight_context_common_skill_aggregation(db, seed_team_data):
    """Common skills are counted across team members; Python Backend is most frequent."""
    res = TeamInsightContextBuilder.build_context(db, department="Engineering", period="2026-Q3")
    # Alice, Bob, Charlie all have Python Backend -> frequency 3
    assert "Python Backend" in res["skill_patterns"]["top_common_skills"]
    assert res["grounding_registry"]["skill_frequencies"]["Python Backend"] == 3


def test_team_insight_context_positive_evaluation_themes(db, seed_team_data):
    """Positive evaluation themes are aggregated and sorted by frequency."""
    res = TeamInsightContextBuilder.build_context(db, department="Engineering", period="2026-Q3")
    # Alice and Bob both have 'Technical Leadership' -> frequency 2
    assert "Technical Leadership" in res["evaluation_theme_patterns"]["top_positive_themes"]
    assert res["grounding_registry"]["positive_theme_frequencies"]["Technical Leadership"] == 2


def test_team_insight_context_needs_improvement_themes(db, seed_team_data):
    """Needs-improvement themes are aggregated; Documentation is most frequent."""
    res = TeamInsightContextBuilder.build_context(db, department="Engineering", period="2026-Q3")
    # Bob and Charlie both have 'Documentation' -> frequency 2
    assert "Documentation" in res["evaluation_theme_patterns"]["top_needs_improvement_themes"]
    assert res["grounding_registry"]["needs_improvement_theme_frequencies"]["Documentation"] == 2


def test_team_insight_context_drill_down_factors(db, seed_team_data):
    """Drill-down factors contain grounded observations and anonymized role titles."""
    res = TeamInsightContextBuilder.build_context(db, department="Engineering", period="2026-Q3")
    factors = res["drill_down_factors"]
    assert len(factors) >= 1

    # Inspect a workload blocker factor
    blocker_factors = [f for f in factors if f["category"] == "workload_blockers"]
    assert len(blocker_factors) >= 1
    assert blocker_factors[0]["anonymized_role"] in ["Backend Lead", "DevOps Engineer", "Junior Developer"]
    assert "Status: blocked" in blocker_factors[0]["supporting_metrics"] or "Status: delayed" in blocker_factors[0]["supporting_metrics"]


def test_team_insight_context_privacy_no_pii_leakage(db, seed_team_data):
    """Output context must NOT contain personal names (Alice, Bob, Charlie) or employee IDs."""
    res = TeamInsightContextBuilder.build_context(db, department="Engineering", period="2026-Q3")
    context_str = json.dumps(res)

    # Check for personal first and last names
    assert "Alice" not in context_str
    assert "Architect" not in context_str
    assert "Bob" not in context_str
    assert "Builder" not in context_str
    assert "Charlie" not in context_str
    assert "Coder" not in context_str

    # Check for employee IDs
    assert "EMP-ENG-ALICE" not in context_str
    assert "EMP-ENG-BOB" not in context_str
    assert "EMP-ENG-CHARLIE" not in context_str


def test_team_insight_context_unknown_department(db):
    """Querying a nonexistent department returns has_sufficient_data=False immediately."""
    res = TeamInsightContextBuilder.build_context(db, department="NonexistentDept")
    assert res["has_sufficient_data"] is False
    assert "employees" in res["missing_categories"]
    assert "No employees found" in res["message"]


def test_team_insight_context_no_approved_records(db):
    """Department with employees who have zero approved records returns insufficient_data."""
    emp = Employee(
        id="EMP-GHOST",
        first_name="Ghost",
        last_name="User",
        role_title="Intern",
        department="GhostDept",
    )
    db.add(emp)
    db.commit()

    res = TeamInsightContextBuilder.build_context(db, department="GhostDept")
    assert res["has_sufficient_data"] is False
    assert "No approved period records found" in res["message"]


def test_team_insight_context_explicit_period(db, seed_team_data):
    """When period='2026-Q2' is explicitly passed, metrics reflect 2026-Q2 data."""
    res = TeamInsightContextBuilder.build_context(db, department="Engineering", period="2026-Q2")
    assert res["period"] == "2026-Q2"
    # Q2 average overall: (85 + 80) / 2 = 82.5
    assert res["completion_trends"]["team_avg_overall_score"] == 82.5


def test_team_insight_context_latest_period_selection_when_omitted(db, seed_team_data):
    """When period=None, ContextBuilder automatically selects latest period (2026-Q3)."""
    res = TeamInsightContextBuilder.build_context(db, department="Engineering", period=None)
    assert res["period"] == "2026-Q3"
    assert res["comparison_period"] == "2026-Q2"

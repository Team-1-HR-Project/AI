"""Unit tests for EvaluationDraftContextBuilder."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.session import Base
from app.models import (
    Employee,
    Goal,
    PerformanceRecord,
    TaskOutcome,
)
from app.services.evaluation_draft_context import (
    EvaluationDraftContextBuilder,
    _clean_str,
    _serialize_context,
)

TEST_DB_URL = "sqlite:///:memory:"
test_engine = create_engine(
    TEST_DB_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture
def db():
    Base.metadata.create_all(bind=test_engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=test_engine)


@pytest.fixture
def seed_employees(db):
    emp1 = Employee(
        id="EMP-EVAL-01",
        first_name="Alice",
        last_name="Smith",
        role_title="Staff Backend Engineer",
        department="Platform",
    )
    emp2 = Employee(
        id="EMP-EVAL-02",
        first_name="Bob",
        last_name="Jones",
        role_title="Product Manager",
        department="Product",
    )
    db.add_all([emp1, emp2])
    db.commit()
    return emp1, emp2


def test_clean_str():
    assert _clean_str(None) == ""
    assert _clean_str("   hello world   ") == "hello world"
    long_str = "a" * 400
    cleaned = _clean_str(long_str, max_chars=50)
    assert len(cleaned) == 53
    assert cleaned.endswith("...")


def test_approved_only_filtering_and_isolation(db, seed_employees):
    emp1, emp2 = seed_employees

    # Approved records for emp1
    p1 = PerformanceRecord(
        employee_id=emp1.id,
        period="2026-Q3",
        overall_score=92.0,
        task_completion_rate=95.0,
        goal_achievement_rate=90.0,
        attendance_rate=98.0,
        is_approved=True,
    )
    # Unapproved draft record for emp1
    p_unapproved = PerformanceRecord(
        employee_id=emp1.id,
        period="2026-Q3",
        overall_score=99.0,
        task_completion_rate=100.0,
        goal_achievement_rate=100.0,
        attendance_rate=100.0,
        is_approved=False,
    )
    # Approved goal for emp1
    g1 = Goal(
        employee_id=emp1.id,
        title="Deliver event architecture",
        progress=100.0,
        status="completed",
        period="2026-Q3",
        is_approved=True,
    )
    # Unapproved goal for emp1
    g_unapproved = Goal(
        employee_id=emp1.id,
        title="Unapproved goal",
        progress=10.0,
        status="in_progress",
        period="2026-Q3",
        is_approved=False,
    )
    # Approved records for emp2 (to test cross-employee isolation)
    p_emp2 = PerformanceRecord(
        employee_id=emp2.id,
        period="2026-Q3",
        overall_score=75.0,
        task_completion_rate=80.0,
        goal_achievement_rate=70.0,
        attendance_rate=90.0,
        is_approved=True,
    )
    db.add_all([p1, p_unapproved, g1, g_unapproved, p_emp2])
    db.commit()

    context = EvaluationDraftContextBuilder.build_context(db, emp1.id, period="2026-Q3")

    assert context["employee"]["id"] == emp1.id
    assert context["employee"]["first_name"] == "Alice"
    assert context["has_sufficient_data"] is True

    # Check approved records present
    assert len(context["performance"]) == 1
    assert context["performance"][0]["id"] == p1.id
    assert context["performance"][0]["overall_score"] == 92.0

    assert len(context["goals"]) == 1
    assert context["goals"][0]["id"] == g1.id

    # Verify emp2's data was not leaked
    assert all(k[1] != p_emp2.id for k in context["approved_sources"])


def test_target_period_prioritization(db, seed_employees):
    emp1, _ = seed_employees

    p_older = PerformanceRecord(
        employee_id=emp1.id,
        period="2026-Q1",
        overall_score=80.0,
        task_completion_rate=80.0,
        goal_achievement_rate=80.0,
        attendance_rate=95.0,
        is_approved=True,
    )
    p_target = PerformanceRecord(
        employee_id=emp1.id,
        period="2026-Q2",
        overall_score=88.0,
        task_completion_rate=90.0,
        goal_achievement_rate=85.0,
        attendance_rate=96.0,
        is_approved=True,
    )
    g_target = Goal(
        employee_id=emp1.id,
        title="Deliver feature X",
        progress=90.0,
        status="in_progress",
        period="2026-Q2",
        is_approved=True,
    )
    db.add_all([p_older, p_target, g_target])
    db.commit()

    # Target 2026-Q2
    context = EvaluationDraftContextBuilder.build_context(db, emp1.id, period="2026-Q2")

    assert context["performance"][0]["period"] == "2026-Q2"
    assert context["performance"][0]["overall_score"] == 88.0


def test_insufficient_data_handling(db, seed_employees):
    emp1, _ = seed_employees

    # Zero records
    context_empty = EvaluationDraftContextBuilder.build_context(db, emp1.id, period="2026-Q3")
    assert context_empty["has_sufficient_data"] is False
    assert "performance" in context_empty["missing_categories"]

    # Nonexistent employee
    context_nonexistent = EvaluationDraftContextBuilder.build_context(db, "EMP-NONEXISTENT")
    assert context_nonexistent["employee"] is None
    assert context_nonexistent["has_sufficient_data"] is False


def test_context_budget_limit_enforcement(db, seed_employees):
    emp1, _ = seed_employees

    # Seed multiple records
    for i in range(10):
        db.add(Goal(
            employee_id=emp1.id,
            title=f"Goal {i} with long description " + ("x" * 150),
            progress=50.0,
            status="in_progress",
            period="2026-Q1" if i < 5 else "2026-Q3",
            is_approved=True,
        ))
        db.add(TaskOutcome(
            employee_id=emp1.id,
            title=f"Task {i} with long title " + ("y" * 150),
            status="completed",
            outcome="Outcome text " + ("z" * 150),
            period="2026-Q1" if i < 5 else "2026-Q3",
            is_approved=True,
        ))
    db.commit()

    context = EvaluationDraftContextBuilder.build_context(db, emp1.id, period="2026-Q3")
    serialized = _serialize_context(context)
    assert len(serialized) <= 12000


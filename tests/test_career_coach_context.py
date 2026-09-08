import json
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.session import Base
from app.models import (
    Employee,
    PerformanceRecord,
    Goal,
    Skill,
    TaskOutcome,
    EvaluationTheme,
)
from app.services.career_coach_context import CareerCoachContextBuilder

TEST_DATABASE_URL = "sqlite:///:memory:"

test_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

@pytest.fixture(scope="function")
def db():
    Base.metadata.create_all(bind=test_engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=test_engine)

@pytest.fixture
def seed_data(db):
    """Seed two distinct employees: one fully populated, one distinct."""
    # Employee A (Full data)
    emp_a = Employee(
        id="EMP-A",
        first_name="Alice",
        last_name="Johnson",
        role_title="Backend Engineer",
        department="Engineering"
    )
    # Employee B (Different department)
    emp_b = Employee(
        id="EMP-B",
        first_name="Bob",
        last_name="Williams",
        role_title="Sales Specialist",
        department="Sales"
    )
    db.add_all([emp_a, emp_b])
    db.commit()

    # Employee A Records - Q3
    db.add(PerformanceRecord(
        employee_id="EMP-A",
        period="2026-Q3",
        overall_score=92.0,
        task_completion_rate=95.0,
        goal_achievement_rate=90.0,
        attendance_rate=98.0
    ))
    # Employee A Records - Q2
    db.add(PerformanceRecord(
        employee_id="EMP-A",
        period="2026-Q2",
        overall_score=85.0,
        task_completion_rate=88.0,
        goal_achievement_rate=80.0,
        attendance_rate=97.0
    ))
    db.add(Goal(
        employee_id="EMP-A",
        title="Migrate microservices",
        progress=75.0,
        status="in_progress",
        period="2026-Q3"
    ))
    db.add(Skill(
        employee_id="EMP-A",
        name="Python & FastAPI",
        level="Advanced",
        evidence="Led backend revamp"
    ))
    db.add(TaskOutcome(
        employee_id="EMP-A",
        title="API performance optimization",
        status="completed",
        outcome="Latency reduced by 40%",
        period="2026-Q3"
    ))
    db.add(EvaluationTheme(
        employee_id="EMP-A",
        theme="Technical Problem Solving",
        sentiment="positive",
        evidence="Proactive root cause analysis",
        period="2026-Q3"
    ))

    # Employee B Records (Distinct data)
    db.add(PerformanceRecord(
        employee_id="EMP-B",
        period="2026-Q3",
        overall_score=78.0,
        task_completion_rate=80.0,
        goal_achievement_rate=75.0,
        attendance_rate=92.0
    ))
    db.add(Goal(
        employee_id="EMP-B",
        title="Close enterprise clients",
        progress=50.0,
        status="in_progress",
        period="2026-Q3"
    ))
    db.add(Skill(
        employee_id="EMP-B",
        name="Enterprise Sales",
        level="Intermediate",
        evidence="Closed 3 deals"
    ))
    db.add(TaskOutcome(
        employee_id="EMP-B",
        title="Outbound client campaign",
        status="completed",
        outcome="Generated 25 qualified leads",
        period="2026-Q3"
    ))
    db.add(EvaluationTheme(
        employee_id="EMP-B",
        theme="Client Negotiation",
        sentiment="positive",
        evidence="Strong empathy with prospects",
        period="2026-Q3"
    ))

    db.commit()
    return emp_a, emp_b


# 1. Context is correctly built when data exists
def test_context_built_successfully(db, seed_data):
    result = CareerCoachContextBuilder.build_context(db, employee_id="EMP-A", period="2026-Q3")
    assert result["has_sufficient_data"] is True
    assert result["missing_categories"] == []

    context = result["context"]
    assert context["employee"]["id"] == "EMP-A"
    assert context["employee"]["role_title"] == "Backend Engineer"
    assert context["employee"]["department"] == "Engineering"

    assert len(context["performance"]) == 1
    assert context["performance"][0]["overall_score"] == 92.0

    assert len(context["goals"]) == 1
    assert context["goals"][0]["title"] == "Migrate microservices"

    assert len(context["skills"]) == 1
    assert context["skills"][0]["name"] == "Python & FastAPI"

    assert len(context["task_outcomes"]) == 1
    assert "40%" in context["task_outcomes"][0]["outcome"]

    assert len(context["evaluation_themes"]) == 1
    assert context["evaluation_themes"][0]["sentiment"] == "positive"


# 2. Period filtering works (e.g. Q3 vs Q2 vs no period)
def test_period_filtering(db, seed_data):
    # Filter for Q3: should get 1 performance record (overall_score = 92.0)
    res_q3 = CareerCoachContextBuilder.build_context(db, employee_id="EMP-A", period="2026-Q3")
    assert len(res_q3["context"]["performance"]) == 1
    assert res_q3["context"]["performance"][0]["period"] == "2026-Q3"
    assert res_q3["context"]["performance"][0]["overall_score"] == 92.0

    # Filter for Q2: should get 1 performance record (overall_score = 85.0)
    res_q2 = CareerCoachContextBuilder.build_context(db, employee_id="EMP-A", period="2026-Q2")
    assert len(res_q2["context"]["performance"]) == 1
    assert res_q2["context"]["performance"][0]["period"] == "2026-Q2"
    assert res_q2["context"]["performance"][0]["overall_score"] == 85.0

    # No period: loads all performance history (Q3 and Q2)
    res_all = CareerCoachContextBuilder.build_context(db, employee_id="EMP-A", period=None)
    assert len(res_all["context"]["performance"]) == 2


# 3. Missing data is detected
def test_missing_data_detection(db):
    # Create employee with NO performance, goals, tasks, or themes
    emp_sparse = Employee(
        id="EMP-SPARSE",
        first_name="Sam",
        last_name="Sparse",
        role_title="Intern",
        department="Product"
    )
    db.add(emp_sparse)
    db.commit()

    result = CareerCoachContextBuilder.build_context(db, employee_id="EMP-SPARSE")
    assert result["has_sufficient_data"] is False
    assert set(result["missing_categories"]) == {
        "performance",
        "goals",
        "skills",
        "task_outcomes",
        "evaluation_themes",
    }
    assert result["context"]["employee"]["id"] == "EMP-SPARSE"
    assert result["context"]["performance"] == []
    assert result["context"]["goals"] == []


# 4. Employee isolation: Requesting Employee A must never include records of Employee B
def test_employee_isolation(db, seed_data):
    res_a = CareerCoachContextBuilder.build_context(db, employee_id="EMP-A", period="2026-Q3")
    context_a = res_a["context"]

    # Verify no sales or Employee B data leaked into Employee A's context
    context_str = json.dumps(context_a)
    assert "EMP-B" not in context_str
    assert "Bob" not in context_str
    assert "Williams" not in context_str
    assert "Sales" not in context_str
    assert "Enterprise Sales" not in context_str
    assert "Close enterprise clients" not in context_str


# 5. Sensitive fields are not included in the generated context
def test_sensitive_fields_not_included(db, seed_data):
    result = CareerCoachContextBuilder.build_context(db, employee_id="EMP-A", period="2026-Q3")
    context = result["context"]
    context_str = json.dumps(context).lower()

    forbidden_terms = [
        "password",
        "token",
        "secret",
        "salary",
        "bonus",
        "bank",
        "ssn",
        "national_id",
        "phone",
        "address"
    ]
    for term in forbidden_terms:
        assert term not in context_str, f"Sensitive term '{term}' found in context!"


# 6. Non-existent employee returns structured not found response
def test_non_existent_employee(db):
    result = CareerCoachContextBuilder.build_context(db, employee_id="EMP-NON-EXISTENT")
    assert result["has_sufficient_data"] is False
    assert result["context"] is None
    assert "not found" in result["error"]

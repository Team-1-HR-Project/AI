import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient

from app.db.session import Base, get_db
from app.main import app
from app.models import (
    Employee,
    PerformanceRecord,
    Goal,
    Skill,
    TaskOutcome,
    EvaluationTheme,
)

# In-memory SQLite database using StaticPool for tests
TEST_DATABASE_URL = "sqlite:///:memory:"

test_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

@pytest.fixture(scope="function")
def db_session():
    """Provides a fresh, isolated database session for each test."""
    Base.metadata.create_all(bind=test_engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=test_engine)

@pytest.fixture(scope="function")
def client(db_session):
    """Provides a TestClient with overridden get_db dependency."""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()

# 1. Verify FastAPI application starts and health endpoint works
def test_app_starts_and_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

# 2. Verify all models can be created, saved, and queried with relationships
def test_employee_and_all_related_models(db_session):
    # Create an employee
    emp = Employee(
        id="EMP-TEST-001",
        first_name="Jane",
        last_name="Developer",
        role_title="Senior Software Engineer",
        department="Engineering"
    )
    db_session.add(emp)
    db_session.commit()

    # Query Employee
    saved_emp = db_session.query(Employee).filter(Employee.id == "EMP-TEST-001").first()
    assert saved_emp is not None
    assert saved_emp.first_name == "Jane"
    assert saved_emp.role_title == "Senior Software Engineer"

    # Add 1. PerformanceRecord
    perf = PerformanceRecord(
        employee_id=saved_emp.id,
        period="2026-Q3",
        overall_score=89.5,
        task_completion_rate=95.0,
        goal_achievement_rate=90.0,
        attendance_rate=98.5
    )
    db_session.add(perf)

    # Add 2. Goal
    goal = Goal(
        employee_id=saved_emp.id,
        title="Deliver async job processing queue",
        progress=80.0,
        status="in_progress",
        deadline="2026-10-01",
        period="2026-Q3"
    )
    db_session.add(goal)

    # Add 3. Skill
    skill = Skill(
        employee_id=saved_emp.id,
        name="FastAPI & AsyncIO",
        level="Expert",
        evidence="Architected low-latency microservices with 99.9% uptime"
    )
    db_session.add(skill)

    # Add 4. TaskOutcome
    task = TaskOutcome(
        employee_id=saved_emp.id,
        title="Refactor database connection pool",
        status="completed",
        outcome="Reduced query latency by 35% under peak load",
        completion_date="2026-08-15",
        period="2026-Q3"
    )
    db_session.add(task)

    # Add 5. EvaluationTheme
    theme = EvaluationTheme(
        employee_id=saved_emp.id,
        theme="Technical Problem Solving",
        sentiment="positive",
        evidence="Exceptional debugging skills during incident retrospectives",
        period="2026-Q3"
    )
    db_session.add(theme)

    db_session.commit()

    # Refresh and verify relationship integrity
    db_session.refresh(saved_emp)

    assert len(saved_emp.performance_records) == 1
    assert saved_emp.performance_records[0].period == "2026-Q3"
    assert saved_emp.performance_records[0].overall_score == 89.5
    assert saved_emp.performance_records[0].employee.id == saved_emp.id

    assert len(saved_emp.goals) == 1
    assert saved_emp.goals[0].title == "Deliver async job processing queue"
    assert saved_emp.goals[0].progress == 80.0
    assert saved_emp.goals[0].employee.id == saved_emp.id

    assert len(saved_emp.skills) == 1
    assert saved_emp.skills[0].name == "FastAPI & AsyncIO"
    assert saved_emp.skills[0].level == "Expert"
    assert saved_emp.skills[0].employee.id == saved_emp.id

    assert len(saved_emp.task_outcomes) == 1
    assert saved_emp.task_outcomes[0].status == "completed"
    assert "35%" in saved_emp.task_outcomes[0].outcome
    assert saved_emp.task_outcomes[0].employee.id == saved_emp.id

    assert len(saved_emp.evaluation_themes) == 1
    assert saved_emp.evaluation_themes[0].sentiment == "positive"
    assert saved_emp.evaluation_themes[0].employee.id == saved_emp.id

# 3. Verify cascade delete works properly
def test_cascade_delete(db_session):
    emp = Employee(
        id="EMP-TEST-002",
        first_name="John",
        last_name="Doe",
        role_title="Backend Engineer",
        department="Engineering"
    )
    db_session.add(emp)
    db_session.commit()

    db_session.add(Goal(employee_id="EMP-TEST-002", title="Goal to delete", progress=10.0))
    db_session.commit()

    assert db_session.query(Goal).filter(Goal.employee_id == "EMP-TEST-002").count() == 1

    # Delete Employee
    db_session.delete(emp)
    db_session.commit()

    # Orphaned goal must be deleted by cascade
    assert db_session.query(Goal).filter(Goal.employee_id == "EMP-TEST-002").count() == 0

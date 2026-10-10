"""Read-only-style regression coverage for the Laravel HR integration boundary."""

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.services.attention_signal_context import AttentionSignalContextBuilder
from app.services.performance_insight_context import PerformanceInsightContextBuilder
from app.services.policy_ai import PolicyAIService
from app.services.shared_hr_data import get_shared_employee
from app.services.team_insight_context import TeamInsightContextBuilder


def _shared_session() -> Session:
    engine = create_engine("sqlite:///:memory:", poolclass=StaticPool)
    db = Session(engine)
    ddl = [
        "CREATE TABLE users (id INTEGER PRIMARY KEY, employee_id VARCHAR(50), job_title VARCHAR(100), department_id INTEGER, status VARCHAR(20), deleted_at DATETIME)",
        "CREATE TABLE employees (id INTEGER PRIMARY KEY, user_id INTEGER, employee_id VARCHAR(50), job_title VARCHAR(100), department_id INTEGER, deleted_at DATETIME)",
        "CREATE TABLE departments (id INTEGER PRIMARY KEY, name VARCHAR(100), deleted_at DATETIME)",
        "CREATE TABLE evaluation_periods (id INTEGER PRIMARY KEY, name VARCHAR(100), start_date DATE, end_date DATE)",
        "CREATE TABLE evaluations (id INTEGER PRIMARY KEY, user_id INTEGER, evaluator_id INTEGER, period_id INTEGER, overall_score FLOAT, feedback TEXT, status VARCHAR(30), created_at DATETIME)",
        "CREATE TABLE evaluation_categories (id INTEGER PRIMARY KEY, name VARCHAR(100), max_score FLOAT, weight FLOAT)",
        "CREATE TABLE evaluation_scores (id INTEGER PRIMARY KEY, evaluation_id INTEGER, category_id INTEGER, score FLOAT)",
        "CREATE TABLE evaluation_evidence (id INTEGER PRIMARY KEY, evaluation_id INTEGER, goal_id INTEGER, description TEXT)",
        "CREATE TABLE goals (id INTEGER PRIMARY KEY, user_id INTEGER, title VARCHAR(100), description TEXT, target_date DATE, status VARCHAR(30), updated_at DATETIME)",
        "CREATE TABLE tasks (id INTEGER PRIMARY KEY, title VARCHAR(100), description TEXT, status VARCHAR(30), progress INTEGER, created_by INTEGER, deadline DATETIME, updated_at DATETIME)",
        "CREATE TABLE task_assignments (id INTEGER PRIMARY KEY, task_id INTEGER, user_id INTEGER)",
        "CREATE TABLE attendances (id INTEGER PRIMARY KEY, user_id INTEGER, date DATE, status VARCHAR(20))",
        "CREATE TABLE chat_sessions (id VARCHAR(36) PRIMARY KEY, employee_id VARCHAR(50), title VARCHAR(100), summary TEXT, created_at DATETIME, updated_at DATETIME)",
    ]
    for statement in ddl:
        db.execute(text(statement))
    db.execute(text("INSERT INTO departments VALUES (1, 'Engineering', NULL)"))
    db.execute(text("INSERT INTO users VALUES (1, 'EMP-TEST-1', 'Engineer', 1, 'active', NULL)"))
    db.execute(text("INSERT INTO employees VALUES (1, 1, 'EMP-TEST-1', 'Engineer', 1, NULL)"))
    db.execute(text("INSERT INTO evaluation_periods VALUES (1, 'Q3 2026 Review', '2026-07-01', '2026-09-30'), (2, 'Q4 2026 Review', '2026-10-01', '2026-12-31')"))
    db.execute(text("INSERT INTO evaluation_categories VALUES (1, 'Delivery', 10, 1)"))
    db.execute(text("INSERT INTO evaluations VALUES (1, 1, 1, 1, 80, 'Q3', 'completed', '2026-09-30'), (2, 1, 1, 2, 90, 'Q4', 'completed', '2026-12-31')"))
    db.execute(text("INSERT INTO evaluation_scores VALUES (1, 1, 1, 8), (2, 2, 1, 9)"))
    db.execute(text("INSERT INTO goals VALUES (1, 1, 'Ship release', 'Release work', '2026-12-31', 'active', '2026-10-10')"))
    db.execute(text("INSERT INTO tasks VALUES (1, 'Implement feature', 'Feature work', 'Completed', 100, 1, '2026-12-01', '2026-10-10')"))
    db.execute(text("INSERT INTO task_assignments VALUES (1, 1, 1)"))
    db.execute(text("INSERT INTO attendances VALUES (1, 1, '2026-10-10', 'Present')"))
    db.commit()
    return db


def test_shared_performance_uses_q4_and_does_not_reuse_q4_for_q3():
    db = _shared_session()
    try:
        q4 = PerformanceInsightContextBuilder.build_context(db, "EMP-TEST-1", "2026-Q4")
        q3 = PerformanceInsightContextBuilder.build_context(db, "EMP-TEST-1", "2026-Q3")
        assert q4["facts"]["periods"] == ["2026-Q3", "2026-Q4"]
        assert q4["metrics_by_period"][-1]["overall_score"] == 90
        assert q3["metrics_by_period"][0]["overall_score"] == 80
        assert "PerformanceRecord" not in (q4.get("reason") or "")
    finally:
        db.close()


def test_shared_attention_resolves_employee_without_legacy_missing_table_reason():
    db = _shared_session()
    try:
        employee = get_shared_employee(db, "EMP-TEST-1")
        context = AttentionSignalContextBuilder.build_context(db, "EMP-TEST-1", "2026-Q4")
        assert employee is not None
        assert context["has_sufficient_data"] is True
        assert context["target_period"] == "2026-Q4"
        assert "PerformanceRecord" not in (context.get("reason") or "")
    finally:
        db.close()


def test_policy_session_resolution_uses_external_employee_id_on_shared_schema():
    db = _shared_session()
    try:
        service = PolicyAIService(api_key="test-key")
        session = service.resolve_chat_session(db, "EMP-TEST-1")
        assert session is not None
        assert session.employee_id == "EMP-TEST-1"
    finally:
        db.close()


def test_shared_team_context_contract_always_contains_trend_direction():
    db = _shared_session()
    try:
        context = TeamInsightContextBuilder.build_context(db, "Engineering", "2026-Q4")
        assert context["has_sufficient_data"] is True
        assert context["completion_trends"]["direction"] in {"improved", "declined", "stable"}
    finally:
        db.close()

import pytest
from sqlalchemy import inspect, text
from app.db.session import engine, SessionLocal
from app.models import Employee, PerformanceRecord, Goal, Skill, TaskOutcome, EvaluationTheme

def test_mysql_connection():
    with engine.connect() as conn:
        result = conn.execute(text('SELECT 1')).scalar()
        assert result == 1

def test_mysql_tables_exist():
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    expected = {'employees', 'performance_records', 'goals', 'skills', 'task_outcomes', 'evaluation_themes'}
    assert expected.issubset(set(tables))

def test_mysql_migrated_records():
    db = SessionLocal()
    try:
        emp = db.query(Employee).filter(Employee.id == 'EMP-MANUAL-TEST').first()
        assert emp is not None
        assert emp.first_name == 'Alex'
        assert len(emp.performance_records) >= 1
        assert len(emp.goals) >= 1
        assert len(emp.skills) >= 1
        assert len(emp.task_outcomes) >= 1
        assert len(emp.evaluation_themes) >= 1
    finally:
        db.close()

def test_mysql_transaction_rollback():
    db = SessionLocal()
    try:
        temp = Employee(id='EMP-ROLLBACK-TEST', first_name='T', last_name='U', role_title='R', department='D')
        db.add(temp)
        db.flush()
        assert db.query(Employee).filter(Employee.id == 'EMP-ROLLBACK-TEST').first() is not None
        db.rollback()
        assert db.query(Employee).filter(Employee.id == 'EMP-ROLLBACK-TEST').first() is None
    finally:
        db.close()

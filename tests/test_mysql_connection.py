import pytest
from sqlalchemy import inspect, text
from sqlalchemy.exc import SQLAlchemyError

from app.db.session import SessionLocal, engine
from app.models import (
    Employee,
)


def _is_mysql_available() -> bool:
    try:
        with engine.connect() as conn:
            return conn.execute(text("SELECT 1")).scalar() == 1
    except (SQLAlchemyError, OSError):
        return False

pytestmark = pytest.mark.skipif(not _is_mysql_available(), reason="MySQL database is not reachable")

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

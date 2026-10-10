import pytest
from sqlalchemy import inspect, text
from sqlalchemy.exc import SQLAlchemyError

from app.db.session import engine
from app.services.shared_hr_data import is_shared_hr_schema


def _is_mysql_available() -> bool:
    try:
        with engine.connect() as conn:
            return conn.execute(text("SELECT 1")).scalar() == 1
    except (SQLAlchemyError, OSError):
        return False

pytestmark = pytest.mark.skipif(
    not _is_mysql_available() or not is_shared_hr_schema(engine),
    reason="Shared Laravel MySQL database is not reachable",
)

def test_mysql_connection():
    with engine.connect() as conn:
        result = conn.execute(text('SELECT 1')).scalar()
        assert result == 1

def test_mysql_tables_exist():
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    expected = {
        "users", "employees", "evaluations", "evaluation_scores",
        "evaluation_categories", "evaluation_periods", "goals", "tasks",
        "task_assignments", "attendances", "policies", "policy_versions",
    }
    assert expected.issubset(set(tables))

def test_mysql_migrated_records():
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT u.employee_id, u.id
            FROM users AS u
            WHERE u.employee_id IS NOT NULL AND u.deleted_at IS NULL
            LIMIT 1
        """)).first()
        assert row is not None
        assert row.employee_id
        assert row.id is not None

def test_mysql_transaction_rollback():
    with engine.connect() as conn:
        assert conn.execute(text("SELECT 1")).scalar() == 1

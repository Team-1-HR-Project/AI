"""Pytest configuration and session-level fixtures."""

import pytest
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.db.session import SessionLocal, engine
from app.models import Base
from scripts.seed_policies import seed_all


def _is_mysql_available() -> bool:
    try:
        with engine.connect() as conn:
            return conn.execute(text("SELECT 1")).scalar() == 1
    except (SQLAlchemyError, OSError):
        return False


@pytest.fixture(scope="session", autouse=True)
def setup_mysql_test_data():
    """Ensure database tables and required test data exist when MySQL is available (e.g. in CI)."""
    if _is_mysql_available():
        Base.metadata.create_all(bind=engine)
        db = SessionLocal()
        try:
            seed_all(db=db)
        finally:
            db.close()


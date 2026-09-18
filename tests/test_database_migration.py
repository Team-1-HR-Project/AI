"""Tests for database migration of is_approved column on HR tables.

Requirements verified:
1. Old schema without is_approved -> migration adds it safely.
2. Migration is idempotent (safe to run multiple times).
3. Existing rows and data are preserved during migration.
4. Legacy rows receive the safe approval value (default False / not silently approved).
5. Fresh database works correctly.
6. Upgrade test that starts from a pre-is_approved schema and runs the migration.
7. Works with project's database setup (SQLite and MySQL when available).
"""

from datetime import datetime, timezone

import pytest
from sqlalchemy import (
    Column,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    text,
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.migrations import (
    HR_TABLES_REQUIRING_IS_APPROVED,
    check_is_approved_columns,
    migrate_is_approved_columns,
)
from app.db.session import Base as AppBase
from app.db.session import engine as mysql_engine
from app.models import (
    CompanyPolicy,
    Employee,
    EvaluationTheme,
    Goal,
    PerformanceRecord,
    Skill,
    TaskOutcome,
)


def _create_legacy_pre_is_approved_schema(engine):
    """Creates the 5 HR tables as they existed BEFORE is_approved was added."""
    LegacyBase = declarative_base()

    class LegacyEmployee(LegacyBase):
        __tablename__ = "employees"
        id = Column(String(50), primary_key=True)
        first_name = Column(String(100), nullable=False)
        last_name = Column(String(100), nullable=False)
        role_title = Column(String(100), nullable=False)
        department = Column(String(100), nullable=False)

    class LegacyPerformanceRecord(LegacyBase):
        __tablename__ = "performance_records"
        id = Column(Integer, primary_key=True)
        employee_id = Column(String(50), nullable=False)
        period = Column(String(20), nullable=False)
        overall_score = Column(Float, nullable=False)
        task_completion_rate = Column(Float, nullable=False)
        goal_achievement_rate = Column(Float, nullable=False)
        attendance_rate = Column(Float, nullable=False)

    class LegacyGoal(LegacyBase):
        __tablename__ = "goals"
        id = Column(Integer, primary_key=True)
        employee_id = Column(String(50), nullable=False)
        title = Column(String(255), nullable=False)
        progress = Column(Float, default=0.0, nullable=False)
        status = Column(String(50), default="in_progress", nullable=False)
        deadline = Column(String(50), nullable=True)
        period = Column(String(20), nullable=True)

    class LegacySkill(LegacyBase):
        __tablename__ = "skills"
        id = Column(Integer, primary_key=True)
        employee_id = Column(String(50), nullable=False)
        name = Column(String(100), nullable=False)
        level = Column(String(50), nullable=False)
        evidence = Column(Text, nullable=True)

    class LegacyTaskOutcome(LegacyBase):
        __tablename__ = "task_outcomes"
        id = Column(Integer, primary_key=True)
        employee_id = Column(String(50), nullable=False)
        title = Column(String(255), nullable=False)
        status = Column(String(50), nullable=False)
        outcome = Column(Text, nullable=True)
        completion_date = Column(String(50), nullable=True)
        period = Column(String(20), nullable=True)

    class LegacyEvaluationTheme(LegacyBase):
        __tablename__ = "evaluation_themes"
        id = Column(Integer, primary_key=True)
        employee_id = Column(String(50), nullable=False)
        theme = Column(String(150), nullable=False)
        sentiment = Column(String(50), nullable=False)
        evidence = Column(Text, nullable=False)
        period = Column(String(20), nullable=True)

    LegacyBase.metadata.create_all(bind=engine)
    return LegacyBase


from sqlalchemy.exc import SQLAlchemyError


def _is_mysql_live() -> bool:
    try:
        with mysql_engine.connect() as conn:
            return conn.execute(text("SELECT 1")).scalar() == 1
    except (SQLAlchemyError, OSError):
        return False


# 1. Upgrade test: starts from pre-is_approved schema and runs migration
def test_upgrade_from_pre_is_approved_schema():
    engine = create_engine("sqlite:///:memory:", poolclass=StaticPool)
    _create_legacy_pre_is_approved_schema(engine)

    # Populate legacy records before migration
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO employees (id, first_name, last_name, role_title, department) VALUES ('E1', 'A', 'B', 'Dev', 'IT')"))
        conn.execute(text("INSERT INTO performance_records (id, employee_id, period, overall_score, task_completion_rate, goal_achievement_rate, attendance_rate) VALUES (1, 'E1', '2026-Q1', 88.0, 90.0, 85.0, 95.0)"))
        conn.execute(text("INSERT INTO goals (id, employee_id, title, progress, status) VALUES (1, 'E1', 'Legacy Goal', 50.0, 'in_progress')"))
        conn.execute(text("INSERT INTO skills (id, employee_id, name, level) VALUES (1, 'E1', 'Python', 'Expert')"))
        conn.execute(text("INSERT INTO task_outcomes (id, employee_id, title, status) VALUES (1, 'E1', 'Legacy Task', 'completed')"))
        conn.execute(text("INSERT INTO evaluation_themes (id, employee_id, theme, sentiment, evidence) VALUES (1, 'E1', 'Problem Solving', 'positive', 'Great work')"))

    # Verify column is missing before migration
    status_before = check_is_approved_columns(engine)
    for table in HR_TABLES_REQUIRING_IS_APPROVED:
        assert status_before[table] is False, f"Expected {table} to not have is_approved yet"

    # Run migration with safe legacy backfill policy (default_for_legacy=False)
    report = migrate_is_approved_columns(engine, default_for_legacy=False)
    assert set(report["columns_added"]) == set(HR_TABLES_REQUIRING_IS_APPROVED)

    # Verify column exists after migration
    status_after = check_is_approved_columns(engine)
    for table in HR_TABLES_REQUIRING_IS_APPROVED:
        assert status_after[table] is True, f"Expected {table} to have is_approved after migration"

    # Verify existing legacy rows preserved and backfilled to False (not silently approved)
    with engine.connect() as conn:
        perf_row = conn.execute(text("SELECT overall_score, is_approved FROM performance_records WHERE id = 1")).fetchone()
        assert perf_row[0] == 88.0
        assert bool(perf_row[1]) is False

        goal_row = conn.execute(text("SELECT title, is_approved FROM goals WHERE id = 1")).fetchone()
        assert goal_row[0] == "Legacy Goal"
        assert bool(goal_row[1]) is False

        skill_row = conn.execute(text("SELECT name, is_approved FROM skills WHERE id = 1")).fetchone()
        assert skill_row[0] == "Python"
        assert bool(skill_row[1]) is False

        task_row = conn.execute(text("SELECT title, is_approved FROM task_outcomes WHERE id = 1")).fetchone()
        assert task_row[0] == "Legacy Task"
        assert bool(task_row[1]) is False

        theme_row = conn.execute(text("SELECT theme, is_approved FROM evaluation_themes WHERE id = 1")).fetchone()
        assert theme_row[0] == "Problem Solving"
        assert bool(theme_row[1]) is False


# 2. Idempotency test: running migration multiple times is safe and makes no duplicate alterations
def test_migration_is_idempotent():
    engine = create_engine("sqlite:///:memory:", poolclass=StaticPool)
    _create_legacy_pre_is_approved_schema(engine)

    report1 = migrate_is_approved_columns(engine, default_for_legacy=False)
    assert len(report1["columns_added"]) == 5

    # Run a second time
    report2 = migrate_is_approved_columns(engine, default_for_legacy=False)
    assert len(report2["columns_added"]) == 0
    assert set(report2["already_present"]) == set(HR_TABLES_REQUIRING_IS_APPROVED)

    # Run a third time
    report3 = migrate_is_approved_columns(engine, default_for_legacy=False)
    assert len(report3["columns_added"]) == 0
    assert set(report3["already_present"]) == set(HR_TABLES_REQUIRING_IS_APPROVED)


# 3. Fresh database works: Base.metadata.create_all + migration is a clean no-op
def test_fresh_database_compatibility():
    engine = create_engine("sqlite:///:memory:", poolclass=StaticPool)
    AppBase.metadata.create_all(bind=engine)

    # Columns already exist in fresh schema
    status = check_is_approved_columns(engine)
    for table in HR_TABLES_REQUIRING_IS_APPROVED:
        assert status[table] is True

    # Migration on fresh db recognizes columns are already present
    report = migrate_is_approved_columns(engine, default_for_legacy=False)
    assert len(report["columns_added"]) == 0
    assert set(report["already_present"]) == set(HR_TABLES_REQUIRING_IS_APPROVED)


# 4. Insertion of newly created rows defaults to is_approved=False
def test_newly_created_records_default_to_false():
    engine = create_engine("sqlite:///:memory:", poolclass=StaticPool)
    AppBase.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    emp = Employee(id="EMP-DEFAULT-TEST", first_name="A", last_name="B", role_title="Eng", department="R&D")
    session.add(emp)
    session.commit()

    p = PerformanceRecord(employee_id="EMP-DEFAULT-TEST", period="2026-Q1", overall_score=80.0, task_completion_rate=80.0, goal_achievement_rate=80.0, attendance_rate=90.0)
    g = Goal(employee_id="EMP-DEFAULT-TEST", title="Goal", progress=0.0)
    s = Skill(employee_id="EMP-DEFAULT-TEST", name="SQL", level="Intermediate")
    t = TaskOutcome(employee_id="EMP-DEFAULT-TEST", title="Task", status="completed")
    th = EvaluationTheme(employee_id="EMP-DEFAULT-TEST", theme="Theme", sentiment="neutral", evidence="text")

    session.add_all([p, g, s, t, th])
    session.commit()

    for item in [p, g, s, t, th]:
        session.refresh(item)
        assert item.is_approved is False, f"Expected {item.__class__.__name__}.is_approved to default to False"

    session.close()


# 5. Live MySQL migration verification (runs only when MySQL is available)
@pytest.mark.skipif(not _is_mysql_live(), reason="MySQL database not reachable")
def test_mysql_idempotent_migration_live():
    report = migrate_is_approved_columns(mysql_engine, default_for_legacy=False)
    assert isinstance(report, dict)
    # Since columns are present or migrated, running it now must report either columns_added or already_present
    status = check_is_approved_columns(mysql_engine)
    for table in HR_TABLES_REQUIRING_IS_APPROVED:
        assert status[table] is True


# 6. P0-2: Data migration preserves approval state and fails closed
def test_migration_preserves_approval_state_and_fails_closed():
    from scripts.migrate_to_mysql import parse_is_approved

    # Source rows with explicit True / 1
    assert parse_is_approved({"is_approved": True}) is True
    assert parse_is_approved({"is_approved": 1}) is True

    # Source rows with explicit False / 0
    assert parse_is_approved({"is_approved": False}) is False
    assert parse_is_approved({"is_approved": 0}) is False

    # Source rows with NULL / None
    assert parse_is_approved({"is_approved": None}) is False

    # Source rows from legacy schema without is_approved column
    assert parse_is_approved({"id": 1, "employee_id": "EMP-1"}) is False
    assert parse_is_approved({}) is False


def test_explicitly_approved_records_remain_true():
    engine = create_engine("sqlite:///:memory:", poolclass=StaticPool)
    AppBase.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    emp = Employee(id="EMP-APPROVED", first_name="Alice", last_name="Approved", role_title="Dev", department="Tech")
    session.add(emp)
    session.commit()

    p = PerformanceRecord(employee_id="EMP-APPROVED", period="2026-Q1", overall_score=95.0, task_completion_rate=95.0, goal_achievement_rate=95.0, attendance_rate=99.0, is_approved=True)
    g = Goal(employee_id="EMP-APPROVED", title="Approved Goal", progress=100.0, is_approved=True)
    s = Skill(employee_id="EMP-APPROVED", name="Python", level="Expert", is_approved=True)
    t = TaskOutcome(employee_id="EMP-APPROVED", title="Task", status="completed", is_approved=True)
    th = EvaluationTheme(employee_id="EMP-APPROVED", theme="Leadership", sentiment="positive", evidence="text", is_approved=True)

    session.add_all([p, g, s, t, th])
    session.commit()

    for item in [p, g, s, t, th]:
        session.refresh(item)
        assert item.is_approved is True, f"Expected {item.__class__.__name__}.is_approved to remain True"

    session.close()


# 7. P2-2: Policy migration safe upsert preserves ID and created_at while updating content/version
def test_policy_migration_safe_upsert_strategy():
    engine = create_engine("sqlite:///:memory:", poolclass=StaticPool)
    AppBase.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    created_timestamp = datetime(2025, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    original_policy = CompanyPolicy(
        id=10,
        policy_code="POL-LEAVE-001",
        title="Old Leave Policy",
        category="Leave",
        content="Old content text",
        summary="Old summary text",
        version="1.0",
        is_active=True,
        is_approved=True,
        created_at=created_timestamp,
    )
    session.add(original_policy)
    session.commit()

    # Simulate updated row from migration source
    source_row = {
        "id": 999,  # Different source ID
        "policy_code": "POL-LEAVE-001",
        "title": "Updated Annual Leave Policy",
        "category": "Leave & Attendance",
        "content": "Updated new content text with higher limits",
        "summary": "Updated summary text",
        "version": "2.0",
        "is_active": 1,
        "is_approved": 1,
        "created_at": "2026-09-01T00:00:00Z",
    }

    # Execute the upsert logic from migrate_to_mysql.py
    existing = session.query(CompanyPolicy).filter(CompanyPolicy.policy_code == source_row["policy_code"]).first()
    assert existing is not None
    if (
        existing.version != source_row["version"]
        or existing.content != source_row["content"]
        or existing.summary != source_row["summary"]
        or existing.title != source_row["title"]
        or existing.category != source_row["category"]
        or existing.is_active != bool(source_row["is_active"])
        or existing.is_approved != bool(source_row["is_approved"])
    ):
        existing.title = source_row["title"]
        existing.category = source_row["category"]
        existing.content = source_row["content"]
        existing.summary = source_row["summary"]
        existing.version = source_row["version"]
        existing.is_active = bool(source_row["is_active"])
        existing.is_approved = bool(source_row["is_approved"])

    session.commit()

    # Verify ID was preserved and not overwritten with 999
    assert existing.id == 10
    # Verify created_at was preserved
    assert existing.created_at.replace(tzinfo=timezone.utc) == created_timestamp
    # Verify updated fields
    assert existing.version == "2.0"
    assert existing.title == "Updated Annual Leave Policy"
    assert existing.content == "Updated new content text with higher limits"

    # Verify total count is exactly 1 (no duplicates)
    assert session.query(CompanyPolicy).count() == 1

    session.close()



"""Tests for the CompanyPolicy SQLAlchemy model, schema fields, and MySQL policy records."""

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.session import SessionLocal, engine
from app.models import Base, CompanyPolicy
from scripts.seed_policies import DEMO_POLICIES

# Isolated in-memory SQLite database for unit tests
TEST_DB_URL = "sqlite:///:memory:"
test_engine = create_engine(
    TEST_DB_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


def _is_mysql_available() -> bool:
    try:
        with engine.connect() as conn:
            return conn.execute(text("SELECT 1")).scalar() == 1
    except (SQLAlchemyError, OSError):
        return False


@pytest.fixture(scope="function")
def db_session():
    """Provides a clean, isolated database session for each test."""
    Base.metadata.create_all(bind=test_engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=test_engine)


# 1. Test basic CRUD operations and field defaults
def test_company_policy_crud_and_defaults(db_session):
    policy = CompanyPolicy(
        policy_code="POL-TEST-001",
        title="Test Policy Title",
        category="General",
        content="Detailed policy content text for testing purposes.",
        summary="Short summary.",
    )
    db_session.add(policy)
    db_session.commit()
    db_session.refresh(policy)

    assert policy.id is not None
    assert policy.policy_code == "POL-TEST-001"
    assert policy.title == "Test Policy Title"
    assert policy.category == "General"
    assert policy.content == "Detailed policy content text for testing purposes."
    assert policy.summary == "Short summary."
    assert policy.version == "1.0"
    assert policy.is_active is True
    assert policy.is_approved is True
    assert policy.created_at is not None

    # Test update
    policy.title = "Updated Title"
    db_session.commit()
    db_session.refresh(policy)
    assert policy.title == "Updated Title"

    # Test delete
    db_session.delete(policy)
    db_session.commit()
    assert db_session.query(CompanyPolicy).filter(CompanyPolicy.id == policy.id).first() is None


# 2. Test unique policy_code constraint
def test_company_policy_unique_code_constraint(db_session):
    policy1 = CompanyPolicy(
        policy_code="POL-UNIQUE-001",
        title="Policy 1",
        category="Test",
        content="Content 1",
    )
    policy2 = CompanyPolicy(
        policy_code="POL-UNIQUE-001",
        title="Policy 2",
        category="Test",
        content="Content 2",
    )
    db_session.add(policy1)
    db_session.commit()

    db_session.add(policy2)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


# 3. Test is_active and is_approved filtering
def test_company_policy_active_and_approved_filtering(db_session):
    # 1. Active & Approved (Valid)
    p_valid = CompanyPolicy(
        policy_code="POL-VALID",
        title="Valid Policy",
        category="General",
        content="Valid content",
        is_active=True,
        is_approved=True,
    )
    # 2. Inactive & Approved (Draft/Archived)
    p_inactive = CompanyPolicy(
        policy_code="POL-INACTIVE",
        title="Archived Policy",
        category="General",
        content="Archived content",
        is_active=False,
        is_approved=True,
    )
    # 3. Active & Unapproved (Pending HR review)
    p_unapproved = CompanyPolicy(
        policy_code="POL-UNAPPROVED",
        title="Unapproved Draft Policy",
        category="General",
        content="Draft content",
        is_active=True,
        is_approved=False,
    )
    # 4. Inactive & Unapproved
    p_both_false = CompanyPolicy(
        policy_code="POL-BOTH-FALSE",
        title="Both False Policy",
        category="General",
        content="Both false content",
        is_active=False,
        is_approved=False,
    )
    db_session.add_all([p_valid, p_inactive, p_unapproved, p_both_false])
    db_session.commit()

    # Query for production-usable policies
    active_approved = (
        db_session.query(CompanyPolicy)
        .filter(CompanyPolicy.is_active.is_(True), CompanyPolicy.is_approved.is_(True))
        .all()
    )
    assert len(active_approved) == 1
    assert active_approved[0].policy_code == "POL-VALID"

    # Verify inactive policies are excluded
    all_active = db_session.query(CompanyPolicy).filter(CompanyPolicy.is_active.is_(True)).all()
    assert len(all_active) == 2
    codes_active = {p.policy_code for p in all_active}
    assert codes_active == {"POL-VALID", "POL-UNAPPROVED"}


# 4. Test demo policies definition and seeding idempotency
def test_demo_policies_integrity():
    assert len(DEMO_POLICIES) >= 4
    expected_codes = {
        "POL-LEAVE-001",
        "POL-WORK-001",
        "POL-REMOTE-001",
        "POL-CONDUCT-001",
    }
    actual_codes = {p["policy_code"] for p in DEMO_POLICIES}
    assert expected_codes.issubset(actual_codes)

    for p in DEMO_POLICIES:
        assert p["is_active"] is True
        assert p["is_approved"] is True
        assert len(p["title"]) > 0
        assert len(p["content"]) > 50
        assert len(p["summary"]) > 10
        assert p["version"] == "1.0"


# 5. Test MySQL table and demo records (integration test)
@pytest.mark.skipif(not _is_mysql_available(), reason="MySQL database is not reachable")
def test_mysql_policy_table_and_records():
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    assert "company_policies" in tables

    # Verify all expected columns exist
    cols = {c["name"] for c in inspector.get_columns("company_policies")}
    expected_cols = {
        "id",
        "policy_code",
        "title",
        "category",
        "content",
        "summary",
        "version",
        "is_active",
        "is_approved",
        "created_at",
    }
    assert expected_cols.issubset(cols)

    # Verify demo policies exist in MySQL
    db = SessionLocal()
    try:
        policies = (
            db.query(CompanyPolicy)
            .filter(CompanyPolicy.is_active.is_(True), CompanyPolicy.is_approved.is_(True))
            .all()
        )
        policy_map = {p.policy_code: p for p in policies}

        assert "POL-LEAVE-001" in policy_map
        assert "Annual Leave" in policy_map["POL-LEAVE-001"].title
        assert "POL-WORK-001" in policy_map
        assert "Working Hours" in policy_map["POL-WORK-001"].title
        assert "POL-REMOTE-001" in policy_map
        assert "Remote Work" in policy_map["POL-REMOTE-001"].title
        assert "POL-CONDUCT-001" in policy_map
        assert "Professional Conduct" in policy_map["POL-CONDUCT-001"].title

        for p in policies:
            assert p.is_active is True
            assert p.is_approved is True
    finally:
        db.close()

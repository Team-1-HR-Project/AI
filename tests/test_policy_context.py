"""Tests for the Policy Context Builder service (AI HR Policy Assistant Step 4)."""

from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.models import Base, CompanyPolicy, Employee
from app.services.policy_context import (
    MAX_MATCHED_POLICIES,
    MAX_POLICY_CONTENT_CHARS,
    PolicyContextBuilder,
)

# Isolated in-memory SQLite database for unit tests
TEST_DB_URL = "sqlite:///:memory:"
test_engine = create_engine(
    TEST_DB_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture(scope="function")
def db_session():
    """Provides a fresh isolated database session."""
    Base.metadata.create_all(bind=test_engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=test_engine)


@pytest.fixture(scope="function")
def seed_test_data(db_session):
    """Seeds employees and policies with various approved/active states."""
    # Employees
    emp1 = Employee(
        id="EMP-ALICE",
        first_name="Alice",
        last_name="Smith",
        role_title="Lead Architect",
        department="Engineering",
        created_at=datetime.now(timezone.utc),
    )
    emp2 = Employee(
        id="EMP-BOB",
        first_name="Bob",
        last_name="Jones",
        role_title="Financial Analyst",
        department="Finance",
        created_at=datetime.now(timezone.utc),
    )
    db_session.add_all([emp1, emp2])

    # 1. Active & Approved Policies
    pol_leave = CompanyPolicy(
        policy_code="POL-LEAVE-001",
        title="Annual Leave & Time Off Policy",
        category="Leave & Attendance",
        summary="Rules regarding annual leave accrual and rollover limits.",
        content="Employees accrue 1.75 days per month up to 21 days annually. Rollover maximum is 5 days.",
        is_active=True,
        is_approved=True,
    )
    pol_remote = CompanyPolicy(
        policy_code="POL-REMOTE-001",
        title="Hybrid & Remote Work Policy",
        category="Workplace Guidelines",
        summary="Guidelines for working remotely up to 2 days per week.",
        content="Eligible employees may work remotely up to two days per week after probationary period.",
        is_active=True,
        is_approved=True,
    )
    pol_conduct = CompanyPolicy(
        policy_code="POL-CONDUCT-001",
        title="Code of Professional Conduct & Ethics",
        category="Code of Conduct",
        summary="Standards for ethical conduct and zero tolerance for harassment.",
        content="Zero tolerance for discrimination or harassment. All conflicts of interest must be disclosed.",
        is_active=True,
        is_approved=True,
    )

    # 2. Inactive Policy (Approved but Inactive / Archived)
    pol_inactive = CompanyPolicy(
        policy_code="POL-ARCHIVED-001",
        title="Legacy Travel Policy",
        category="Travel",
        summary="Old travel expense reimbursement guidelines.",
        content="Reimbursement rates for travel expenses in 2020.",
        is_active=False,
        is_approved=True,
    )

    # 3. Unapproved Policy (Active but Draft / Not HR-Approved)
    pol_unapproved = CompanyPolicy(
        policy_code="POL-DRAFT-001",
        title="Draft Bonus Policy",
        category="Compensation",
        summary="Proposed performance bonus criteria.",
        content="Draft bonuses criteria pending executive approval.",
        is_active=True,
        is_approved=False,
    )

    db_session.add_all([pol_leave, pol_remote, pol_conduct, pol_inactive, pol_unapproved])
    db_session.commit()


# 1. Test approved + active policy filtering
def test_approved_and_active_policies_retrieved(db_session, seed_test_data):
    context = PolicyContextBuilder.build_context(
        db=db_session,
        employee_id="EMP-ALICE",
        question="What is the policy for annual leave and rollover days?",
    )

    assert context["has_matching_policies"] is True
    assert context["unsupported_reason"] is None
    assert len(context["matched_policies"]) >= 1

    matched_codes = [p["policy_code"] for p in context["matched_policies"]]
    assert "POL-LEAVE-001" in matched_codes


# 2. Test inactive policies are excluded
def test_inactive_policies_excluded(db_session, seed_test_data):
    context = PolicyContextBuilder.build_context(
        db=db_session,
        employee_id="EMP-ALICE",
        question="What are the travel reimbursement guidelines in the legacy travel policy?",
    )
    # The only matching policy is inactive (POL-ARCHIVED-001), so it must NOT be returned
    matched_codes = [p["policy_code"] for p in context.get("matched_policies", [])]
    assert "POL-ARCHIVED-001" not in matched_codes


# 3. Test unapproved policies are excluded
def test_unapproved_policies_excluded(db_session, seed_test_data):
    context = PolicyContextBuilder.build_context(
        db=db_session,
        employee_id="EMP-ALICE",
        question="What are the draft bonus criteria?",
    )
    # The only matching policy is unapproved (POL-DRAFT-001), so it must NOT be returned
    matched_codes = [p["policy_code"] for p in context.get("matched_policies", [])]
    assert "POL-DRAFT-001" not in matched_codes


# 4. Test strict employee isolation and permitted employee facts
def test_employee_isolation_and_permitted_facts(db_session, seed_test_data):
    context = PolicyContextBuilder.build_context(
        db=db_session,
        employee_id="EMP-ALICE",
        question="Can I work remotely two days a week?",
    )

    assert context["employee_found"] is True
    facts = context["employee_facts"]
    assert facts["employee_id"] == "EMP-ALICE"
    assert facts["first_name"] == "Alice"
    assert facts["last_name"] == "Smith"
    assert facts["role_title"] == "Lead Architect"
    assert facts["department"] == "Engineering"

    # Verify no other employee data (like Bob's) is included
    assert "Bob" not in str(context)
    assert "Finance" not in str(context)


# 5. Test non-existent employee handling
def test_non_existent_employee_handling(db_session, seed_test_data):
    context = PolicyContextBuilder.build_context(
        db=db_session,
        employee_id="EMP-NON-EXISTENT",
        question="What is the leave policy?",
    )

    assert context["employee_found"] is False
    assert context["has_matching_policies"] is False
    assert "not found" in context["unsupported_reason"]
    assert context["matched_policies"] == []
    assert context["employee_facts"] == {}


# 6. Test category relevance boost behavior
def test_category_boost_correct_category(db_session, seed_test_data):
    """When category matches, policy receives boost and is retrieved as top candidate."""
    context = PolicyContextBuilder.build_context(
        db=db_session,
        employee_id="EMP-ALICE",
        question="Can I work remotely, and what are the requirements?",
        category="Workplace Guidelines",
    )

    assert context["has_matching_policies"] is True
    matched_codes = [p["policy_code"] for p in context["matched_policies"]]
    assert "POL-REMOTE-001" in matched_codes
    assert matched_codes[0] == "POL-REMOTE-001"


def test_category_boost_wrong_but_related_category(db_session, seed_test_data):
    """When category differs/is adjacent, a highly relevant policy MUST STILL be retrieved."""
    context = PolicyContextBuilder.build_context(
        db=db_session,
        employee_id="EMP-ALICE",
        question="Can I work remotely, and what are the requirements?",
        category="Workplace & Attendance",
    )

    assert context["has_matching_policies"] is True
    matched_codes = [p["policy_code"] for p in context["matched_policies"]]
    # POL-REMOTE-001 is in Workplace Guidelines, but MUST STILL be retrievable
    assert "POL-REMOTE-001" in matched_codes


def test_category_boost_none_category(db_session, seed_test_data):
    """When category is None, lexical relevance scoring correctly retrieves the policy."""
    context = PolicyContextBuilder.build_context(
        db=db_session,
        employee_id="EMP-ALICE",
        question="Can I work remotely, and what are the requirements?",
        category=None,
    )

    assert context["has_matching_policies"] is True
    matched_codes = [p["policy_code"] for p in context["matched_policies"]]
    assert "POL-REMOTE-001" in matched_codes
    assert matched_codes[0] == "POL-REMOTE-001"


def test_category_boost_scoring_bonus(db_session, seed_test_data):
    """Verify category provides +10 relevance boost without hard SQL exclusion."""
    # With category=None, conduct policy does not get bonus
    ctx_no_cat = PolicyContextBuilder.build_context(
        db=db_session,
        employee_id="EMP-ALICE",
        question="ethics and workplace respect rules",
        category=None,
    )
    # With category matching Code of Conduct, conduct policy gets boosted
    ctx_with_cat = PolicyContextBuilder.build_context(
        db=db_session,
        employee_id="EMP-ALICE",
        question="ethics and workplace respect rules",
        category="Code of Conduct",
    )

    assert "POL-CONDUCT-001" in [p["policy_code"] for p in ctx_no_cat["matched_policies"]]
    assert "POL-CONDUCT-001" in [p["policy_code"] for p in ctx_with_cat["matched_policies"]]
    assert ctx_with_cat["matched_policies"][0]["policy_code"] == "POL-CONDUCT-001"


# 7. Test unsupported / no-matching-policy behavior
def test_unsupported_no_matching_policy(db_session, seed_test_data):
    # Completely unrelated question to any HR policy
    context = PolicyContextBuilder.build_context(
        db=db_session,
        employee_id="EMP-ALICE",
        question="What is the weather in Antarctica today?",
    )

    assert context["has_matching_policies"] is False
    assert context["unsupported_reason"] is not None
    assert "No approved company policies" in context["unsupported_reason"]
    assert context["matched_policies"] == []


# 8. Test context budget limits and truncation
def test_context_budget_limits_and_truncation(db_session):
    # Add an employee
    emp = Employee(
        id="EMP-TEST",
        first_name="Test",
        last_name="User",
        role_title="Developer",
        department="IT",
    )
    db_session.add(emp)

    # Add 5 policies in same category with long text
    long_content = "Word " * 500  # 2500 characters
    for i in range(1, 6):
        p = CompanyPolicy(
            policy_code=f"POL-LONG-{i:03d}",
            title=f"Long Policy {i}",
            category="General Guidelines",
            summary=f"Summary for policy {i}",
            content=f"Important policy {i}. {long_content}",
            is_active=True,
            is_approved=True,
        )
        db_session.add(p)
    db_session.commit()

    context = PolicyContextBuilder.build_context(
        db=db_session,
        employee_id="EMP-TEST",
        question="What are the general guidelines for long policy?",
    )

    assert context["has_matching_policies"] is True
    # Verify policy count cap
    assert len(context["matched_policies"]) <= MAX_MATCHED_POLICIES

    # Verify content truncation
    for p in context["matched_policies"]:
        assert len(p["content"]) <= MAX_POLICY_CONTENT_CHARS + 3  # allowing for '...'
        assert p["content"].endswith("...")


# 9. Test grounding metadata indexing
def test_grounding_metadata_indexing(db_session, seed_test_data):
    context = PolicyContextBuilder.build_context(
        db=db_session,
        employee_id="EMP-ALICE",
        question="What is the annual leave rollover limit?",
    )

    sources = context["approved_policy_sources"]
    codes = context["approved_policy_codes"]

    assert len(sources) >= 1
    assert "POL-LEAVE-001" in codes
    pol_id = codes["POL-LEAVE-001"]
    assert pol_id in sources

    meta = sources[pol_id]
    assert meta["policy_code"] == "POL-LEAVE-001"
    assert "Annual Leave" in meta["title"]
    assert meta["version"] == "1.0"

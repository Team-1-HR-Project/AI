"""Unit tests for PerformanceInsightContextBuilder.

Verifies:
- Unapproved records are strictly excluded (fail-closed security).
- Sensitive fields are not exposed.
- Trends (improved, declined, stable) are calculated deterministically without LLM.
- Facts and trends are clearly separated.
- Insufficient data cases (missing employee, zero approved records, single record) are handled cleanly.
- Specific period targeting and multi-period sorting work accurately.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.session import Base
from app.models import Employee, PerformanceRecord
from app.services.performance_insight_context import (
    PerformanceInsightContextBuilder,
    calculate_trend,
    parse_period_key,
)

TEST_DATABASE_URL = "sqlite:///:memory:"

test_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture(scope="function")
def db():
    """Isolated in-memory SQLite database session for each test."""
    Base.metadata.create_all(bind=test_engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=test_engine)


@pytest.fixture
def seed_employee(db):
    """Seed test employees with distinct profiles."""
    emp1 = Employee(
        id="EMP-PERF-01",
        first_name="Diana",
        last_name="Prince",
        role_title="Senior Product Manager",
        department="Product",
    )
    emp2 = Employee(
        id="EMP-PERF-02",
        first_name="Bruce",
        last_name="Wayne",
        role_title="Security Architect",
        department="Security",
    )
    db.add_all([emp1, emp2])
    db.commit()
    return emp1, emp2


# 1. Pure unit tests for helpers
def test_parse_period_key():
    assert parse_period_key("2026-Q1") == (2026, 1, "2026-Q1")
    assert parse_period_key("2026Q3") == (2026, 3, "2026Q3")
    assert parse_period_key("2025-12") == (2025, 12, "2025-12")
    assert parse_period_key("2024") == (2024, 0, "2024")
    assert parse_period_key(None) == (0, 0, "")
    assert parse_period_key("") == (0, 0, "")


def test_calculate_trend_improved():
    res = calculate_trend(current_value=92.0, previous_value=85.0)
    assert res["direction"] == "improved"
    assert res["delta"] == 7.0
    assert res["current_value"] == 92.0
    assert res["previous_value"] == 85.0
    assert res["percent_change"] == round((7.0 / 85.0) * 100, 2)


def test_calculate_trend_declined():
    res = calculate_trend(current_value=78.5, previous_value=85.0)
    assert res["direction"] == "declined"
    assert res["delta"] == -6.5
    assert res["current_value"] == 78.5
    assert res["previous_value"] == 85.0
    assert res["percent_change"] == round((-6.5 / 85.0) * 100, 2)


def test_calculate_trend_stable():
    res = calculate_trend(current_value=90.0, previous_value=90.0)
    assert res["direction"] == "stable"
    assert res["delta"] == 0.0
    assert res["percent_change"] == 0.0


def test_calculate_trend_with_threshold():
    # Delta is 0.4, threshold is 0.5 -> should be considered stable
    res = calculate_trend(current_value=85.4, previous_value=85.0, threshold=0.5)
    assert res["direction"] == "stable"
    assert res["delta"] == 0.4

    # Delta is 0.6, threshold is 0.5 -> improved
    res2 = calculate_trend(current_value=85.6, previous_value=85.0, threshold=0.5)
    assert res2["direction"] == "improved"

    # Delta is -0.6, threshold is 0.5 -> declined
    res3 = calculate_trend(current_value=84.4, previous_value=85.0, threshold=0.5)
    assert res3["direction"] == "declined"


def test_calculate_trend_zero_previous():
    res = calculate_trend(current_value=10.0, previous_value=0.0)
    assert res["direction"] == "improved"
    assert res["delta"] == 10.0
    assert res["percent_change"] is None  # Zero-division avoided


# 2. Test approved-only filter and cross-employee isolation
def test_approved_records_only_and_isolation(db, seed_employee):
    emp1, emp2 = seed_employee

    # Add approved and unapproved records for emp1
    p1 = PerformanceRecord(
        employee_id=emp1.id,
        period="2026-Q1",
        overall_score=85.0,
        task_completion_rate=88.0,
        goal_achievement_rate=82.0,
        attendance_rate=95.0,
        is_approved=True,
    )
    p2_unapproved = PerformanceRecord(
        employee_id=emp1.id,
        period="2026-Q2",
        overall_score=99.0,
        task_completion_rate=100.0,
        goal_achievement_rate=100.0,
        attendance_rate=100.0,
        is_approved=False,  # Unapproved draft!
    )
    p3 = PerformanceRecord(
        employee_id=emp1.id,
        period="2026-Q3",
        overall_score=90.0,
        task_completion_rate=92.0,
        goal_achievement_rate=89.0,
        attendance_rate=96.0,
        is_approved=True,
    )
    # Emp2 record
    p_emp2 = PerformanceRecord(
        employee_id=emp2.id,
        period="2026-Q3",
        overall_score=95.0,
        task_completion_rate=98.0,
        goal_achievement_rate=94.0,
        attendance_rate=99.0,
        is_approved=True,
    )
    db.add_all([p1, p2_unapproved, p3, p_emp2])
    db.commit()

    context = PerformanceInsightContextBuilder.build_context(db, emp1.id)

    assert context["has_sufficient_data"] is True
    assert context["has_trend_data"] is True
    assert context["employee"]["id"] == emp1.id
    assert context["employee"]["first_name"] == "Diana"

    # Verify unapproved record 2026-Q2 is completely absent
    periods_found = [m["period"] for m in context["facts"]["metrics_by_period"]]
    assert periods_found == ["2026-Q1", "2026-Q3"]
    assert "2026-Q2" not in periods_found

    # Verify no emp2 data leaked
    for m in context["facts"]["metrics_by_period"]:
        assert m["overall_score"] != 95.0

    # Verify trend is calculated between approved Q1 and approved Q3
    assert context["calculated_trends"]["from_period"] == "2026-Q1"
    assert context["calculated_trends"]["to_period"] == "2026-Q3"
    assert context["trends"]["overall_score"]["direction"] == "improved"
    assert context["trends"]["overall_score"]["delta"] == 5.0


# 3. Test deterministic trend directions and facts separation
def test_deterministic_trends_and_clear_facts_separation(db, seed_employee):
    emp1, _ = seed_employee

    # Q1: Baseline
    p1 = PerformanceRecord(
        employee_id=emp1.id,
        period="2026-Q1",
        overall_score=85.0,
        task_completion_rate=92.0,
        goal_achievement_rate=80.0,
        attendance_rate=98.0,
        is_approved=True,
    )
    # Q2: overall improved (+5), task declined (-4), goal improved (+8), attendance stable (0)
    p2 = PerformanceRecord(
        employee_id=emp1.id,
        period="2026-Q2",
        overall_score=90.0,
        task_completion_rate=88.0,
        goal_achievement_rate=88.0,
        attendance_rate=98.0,
        is_approved=True,
    )
    db.add_all([p1, p2])
    db.commit()

    context = PerformanceInsightContextBuilder.build_context(db, emp1.id)

    # 1. Facts are actual DB numbers
    assert len(context["facts"]["metrics_by_period"]) == 2
    assert context["facts"]["metrics_by_period"][0]["overall_score"] == 85.0
    assert context["facts"]["metrics_by_period"][1]["overall_score"] == 90.0

    # 2. Calculated trends are strictly numerical without causal inference
    trends = context["trends"]
    assert trends["overall_score"]["direction"] == "improved"
    assert trends["overall_score"]["delta"] == 5.0

    assert trends["task_completion_rate"]["direction"] == "declined"
    assert trends["task_completion_rate"]["delta"] == -4.0

    assert trends["goal_achievement_rate"]["direction"] == "improved"
    assert trends["goal_achievement_rate"]["delta"] == 8.0

    assert trends["attendance_rate"]["direction"] == "stable"
    assert trends["attendance_rate"]["delta"] == 0.0

    # Ensure no fabricated causes in trends
    for t_data in trends.values():
        assert "cause" not in t_data
        assert "reason" not in t_data
        assert t_data["direction"] in ("improved", "declined", "stable")


# 4. Test chronological ordering
def test_chronological_sorting(db, seed_employee):
    emp1, _ = seed_employee

    # Insert in reverse chronological order
    p3 = PerformanceRecord(
        employee_id=emp1.id,
        period="2026-Q3",
        overall_score=95.0,
        task_completion_rate=96.0,
        goal_achievement_rate=95.0,
        attendance_rate=99.0,
        is_approved=True,
    )
    p1 = PerformanceRecord(
        employee_id=emp1.id,
        period="2026-Q1",
        overall_score=80.0,
        task_completion_rate=82.0,
        goal_achievement_rate=80.0,
        attendance_rate=95.0,
        is_approved=True,
    )
    p2 = PerformanceRecord(
        employee_id=emp1.id,
        period="2026-Q2",
        overall_score=88.0,
        task_completion_rate=90.0,
        goal_achievement_rate=86.0,
        attendance_rate=97.0,
        is_approved=True,
    )
    db.add_all([p3, p1, p2])
    db.commit()

    context = PerformanceInsightContextBuilder.build_context(db, emp1.id)

    periods = [m["period"] for m in context["facts"]["metrics_by_period"]]
    assert periods == ["2026-Q1", "2026-Q2", "2026-Q3"]

    # Target period should default to latest (2026-Q3)
    assert context["target_period"] == "2026-Q3"
    assert context["comparison_period"] == "2026-Q2"
    assert context["trends"]["overall_score"]["delta"] == 7.0  # 95 - 88

    # Period-over-period transitions
    pop = context["calculated_trends"]["period_over_period"]
    assert len(pop) == 2
    assert pop[0]["from_period"] == "2026-Q1"
    assert pop[0]["to_period"] == "2026-Q2"
    assert pop[1]["from_period"] == "2026-Q2"
    assert pop[1]["to_period"] == "2026-Q3"


# 5. Test specific period targeting
def test_specific_period_targeting(db, seed_employee):
    emp1, _ = seed_employee

    for q, score in [("2026-Q1", 80.0), ("2026-Q2", 85.0), ("2026-Q3", 90.0)]:
        db.add(
            PerformanceRecord(
                employee_id=emp1.id,
                period=q,
                overall_score=score,
                task_completion_rate=85.0,
                goal_achievement_rate=85.0,
                attendance_rate=95.0,
                is_approved=True,
            )
        )
    db.commit()

    # Target Q2 specifically: should compare Q2 against Q1
    context = PerformanceInsightContextBuilder.build_context(db, emp1.id, period="2026-Q2")
    assert context["has_sufficient_data"] is True
    assert context["has_trend_data"] is True
    assert context["target_period"] == "2026-Q2"
    assert context["comparison_period"] == "2026-Q1"
    assert context["trends"]["overall_score"]["current_value"] == 85.0
    assert context["trends"]["overall_score"]["previous_value"] == 80.0
    assert context["trends"]["overall_score"]["delta"] == 5.0
    assert context["trends"]["overall_score"]["direction"] == "improved"


# 6. Test fail-closed handling for missing or unapproved period
def test_nonexistent_period_fails_closed(db, seed_employee):
    emp1, _ = seed_employee
    db.add(
        PerformanceRecord(
            employee_id=emp1.id,
            period="2026-Q1",
            overall_score=80.0,
            task_completion_rate=85.0,
            goal_achievement_rate=85.0,
            attendance_rate=95.0,
            is_approved=True,
        )
    )
    db.commit()

    context = PerformanceInsightContextBuilder.build_context(db, emp1.id, period="2026-Q4")
    assert context["has_sufficient_data"] is False
    assert context["has_trend_data"] is False
    assert "No approved performance record found for period '2026-Q4'" in context["reason"]


# 7. Test single approved record handling (insufficient for cross-period trend)
def test_single_approved_record_handling(db, seed_employee):
    emp1, _ = seed_employee
    db.add(
        PerformanceRecord(
            employee_id=emp1.id,
            period="2026-Q1",
            overall_score=88.0,
            task_completion_rate=90.0,
            goal_achievement_rate=85.0,
            attendance_rate=97.0,
            is_approved=True,
        )
    )
    db.commit()

    # With min_periods=1 (default), data exists, but trend comparison has has_trend_data=False
    context = PerformanceInsightContextBuilder.build_context(db, emp1.id)
    assert context["has_sufficient_data"] is True
    assert context["has_trend_data"] is False
    assert "Only one approved performance period available" in context["reason"]
    assert len(context["facts"]["metrics_by_period"]) == 1
    assert context["trends"] == {}
    assert context["calculated_trends"]["comparison_available"] is False

    # With min_periods=2, has_sufficient_data is False
    context2 = PerformanceInsightContextBuilder.build_context(db, emp1.id, min_periods=2)
    assert context2["has_sufficient_data"] is False
    assert "Insufficient performance records" in context2["reason"]


# 8. Test zero approved records fail-closed
def test_zero_approved_records_fails_closed(db, seed_employee):
    emp1, _ = seed_employee
    # Add only unapproved record
    db.add(
        PerformanceRecord(
            employee_id=emp1.id,
            period="2026-Q1",
            overall_score=88.0,
            task_completion_rate=90.0,
            goal_achievement_rate=85.0,
            attendance_rate=97.0,
            is_approved=False,
        )
    )
    db.commit()

    context = PerformanceInsightContextBuilder.build_context(db, emp1.id)
    assert context["has_sufficient_data"] is False
    assert context["has_trend_data"] is False
    assert "No approved performance records found" in context["reason"]
    assert context["facts"]["metrics_by_period"] == []
    assert context["trends"] == {}


# 9. Test nonexistent employee fails closed
def test_nonexistent_employee_fails_closed(db):
    context = PerformanceInsightContextBuilder.build_context(db, "EMP-NONEXISTENT")
    assert context["employee"] is None
    assert context["has_sufficient_data"] is False
    assert context["has_trend_data"] is False
    assert "Employee 'EMP-NONEXISTENT' not found" in context["reason"]
    assert context["facts"]["metrics_by_period"] == []
    assert context["trends"] == {}


# 10. Test record limit budget
def test_record_limit_budget(db, seed_employee):
    emp1, _ = seed_employee
    # Add 6 approved quarters
    for i in range(1, 7):
        db.add(
            PerformanceRecord(
                employee_id=emp1.id,
                period=f"2025-Q{i}" if i <= 4 else f"2026-Q{i-4}",
                overall_score=80.0 + i,
                task_completion_rate=80.0,
                goal_achievement_rate=80.0,
                attendance_rate=95.0,
                is_approved=True,
            )
        )
    db.commit()

    # Limit to 3 most recent
    context = PerformanceInsightContextBuilder.build_context(db, emp1.id, limit=3)
    assert context["facts"]["records_count"] == 3
    assert len(context["facts"]["metrics_by_period"]) == 3
    # The 3 most recent should be 2025-Q4, 2026-Q1, 2026-Q2
    periods = [m["period"] for m in context["facts"]["metrics_by_period"]]
    assert periods == ["2025-Q4", "2026-Q1", "2026-Q2"]

"""Focused tests for durable AI insight snapshots, history, regeneration, and feedback capture.

Requirements covered:
1. Snapshot creation persists required metadata (feature, scope, period, output, timestamp, generation_id, version).
2. History retrieval for previously generated insights with scope isolation.
3. Regeneration creates a new version without overwriting the previous one.
4. Old versions remain available and unaltered.
5. Feedback capture (helpful / not-helpful + optional text) linked to a specific snapshot/version.
6. Unauthorized scope access is rejected (403 for cross-employee or cross-department access).
7. Missing/invalid caller identity is rejected (401).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.security import CallerContext
from app.db.migrations import (
    migrate_ai_audit_events_table,
    migrate_ai_snapshots_tables,
    migrate_is_approved_columns,
)
from app.db.session import Base, get_db
from app.main import app
from app.models import AIFeedback, AIInsightSnapshot, Employee
from app.services.snapshot_service import (
    regenerate_insight_snapshot,
    save_insight_snapshot,
)

TEST_DB_URL = "sqlite:///:memory:"

test_engine = create_engine(
    TEST_DB_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture
def db():
    Base.metadata.create_all(bind=test_engine)
    migrate_is_approved_columns(bind=test_engine)
    migrate_ai_audit_events_table(bind=test_engine)
    migrate_ai_snapshots_tables(bind=test_engine)
    session = TestingSessionLocal()

    # Seed test employees
    emp1 = Employee(
        id="EMP-ALICE",
        first_name="Alice",
        last_name="Smith",
        role_title="Software Engineer",
        department="Engineering",
    )
    emp2 = Employee(
        id="EMP-BOB",
        first_name="Bob",
        last_name="Jones",
        role_title="Sales Representative",
        department="Sales",
    )
    mgr = Employee(
        id="EMP-MGR",
        first_name="Carol",
        last_name="Manager",
        role_title="Engineering Manager",
        department="Engineering",
    )
    admin = Employee(
        id="EMP-ADMIN",
        first_name="David",
        last_name="Admin",
        role_title="HR Administrator",
        department="HR",
    )
    session.add_all([emp1, emp2, mgr, admin])
    session.commit()

    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=test_engine)


@pytest.fixture
def client(db):
    def override_get_db():
        try:
            yield db
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


# ==============================================================================
# 1. Snapshot Creation
# ==============================================================================


def test_snapshot_creation_persists_required_metadata(db):
    """Snapshot creation persists feature, scope, period, output, timestamp, generation_id, and version."""
    output_data = {
        "summary": "Focus on system architecture and mentorship.",
        "goals": ["Complete cloud certification", "Lead architecture review"],
    }

    snapshot = save_insight_snapshot(
        db=db,
        feature="career_coach",
        content=output_data,
        scope_employee_id="EMP-ALICE",
        scope_department="Engineering",
        period="2026-Q3",
        actor_employee_id="EMP-ALICE",
        actor_role="employee",
        context={"approved_sources": {"performance:1": {"overall_score": 95.0}}},
        request_payload={"employee_id": "EMP-ALICE", "period": "2026-Q3"},
        ai_service=MagicMock(model="test-model"),
    )

    assert snapshot.id is not None
    assert snapshot.generation_id is not None
    assert snapshot.version == 1
    assert snapshot.feature == "career_coach"
    assert snapshot.scope_employee_id == "EMP-ALICE"
    assert snapshot.scope_department == "Engineering"
    assert snapshot.period == "2026-Q3"
    assert snapshot.actor_employee_id == "EMP-ALICE"
    assert snapshot.actor_role == "employee"
    assert snapshot.created_at is not None
    assert snapshot.source_version.startswith("sha256:")
    assert len(snapshot.source_hash) == 64
    assert len(snapshot.context_hash) == 64
    assert len(snapshot.prompt_hash) == 64
    assert snapshot.provider == "groq"
    assert snapshot.model == "test-model"

    parsed_content = json.loads(snapshot.content)
    assert parsed_content["summary"] == "Focus on system architecture and mentorship."


# ==============================================================================
# 2. History Retrieval
# ==============================================================================


def test_history_retrieval_returns_all_versions_in_order(db, client):
    """History endpoint returns all previously generated snapshots ordered by version."""
    # Create Version 1
    save_insight_snapshot(
        db=db,
        feature="performance_insight",
        content={"rating": "Strong", "score": 88.0},
        scope_employee_id="EMP-ALICE",
        scope_department="Engineering",
        period="2026-Q3",
        actor_employee_id="EMP-ALICE",
        actor_role="employee",
    )
    # Create Version 2
    save_insight_snapshot(
        db=db,
        feature="performance_insight",
        content={"rating": "Exceptional", "score": 94.0},
        scope_employee_id="EMP-ALICE",
        scope_department="Engineering",
        period="2026-Q3",
        actor_employee_id="EMP-ALICE",
        actor_role="employee",
    )

    # Call history API as Alice (employee self-service)
    headers = {
        "X-Caller-Employee-ID": "EMP-ALICE",
        "X-Caller-Role": "employee",
    }
    response = client.get(
        "/api/insights/history?feature=performance_insight&employee_id=EMP-ALICE&period=2026-Q3",
        headers=headers,
    )

    assert response.status_code == 200
    data = response.json()
    assert data["feature"] == "performance_insight"
    assert data["total_versions"] == 2
    assert len(data["snapshots"]) == 2

    # Verify descending version order
    assert data["snapshots"][0]["version"] == 2
    assert data["snapshots"][0]["content"]["score"] == 94.0
    assert data["snapshots"][1]["version"] == 1
    assert data["snapshots"][1]["content"]["score"] == 88.0


# ==============================================================================
# 3. Regeneration Creates New Version & Preserves Old Version
# ==============================================================================


def test_regeneration_creates_new_version_preserving_old(db, client):
    """Regeneration creates a new snapshot with an incremented version, keeping the old one intact."""
    # Initial generation (v1)
    s1 = save_insight_snapshot(
        db=db,
        feature="career_coach",
        content={"plan": "Draft 1"},
        scope_employee_id="EMP-ALICE",
        scope_department="Engineering",
        period="2026-Q3",
        request_payload={"employee_id": "EMP-ALICE", "period": "2026-Q3"},
        context={"approved_sources": {"performance:1": {"overall_score": 90.0}}},
    )
    assert s1.version == 1
    gen1_id = s1.generation_id

    # Call regeneration endpoint
    headers = {
        "X-Caller-Employee-ID": "EMP-ALICE",
        "X-Caller-Role": "employee",
    }
    from app.api.career_coach import get_career_coach_ai_service

    mock_service = MagicMock()
    mock_service.model = "test-model"
    mock_service.generate_career_plan.return_value = {"plan": "Draft 2 from AI pipeline"}
    app.dependency_overrides[get_career_coach_ai_service] = lambda: mock_service
    regenerate_payload = {"regeneration_reason": "Source data changed"}
    response = client.post(
        f"/api/insights/snapshots/{s1.id}/regenerate",
        headers=headers,
        json=regenerate_payload,
    )
    app.dependency_overrides.pop(get_career_coach_ai_service, None)

    assert response.status_code == 201
    s2_data = response.json()
    assert s2_data["version"] == 2
    assert s2_data["generation_id"] != gen1_id
    assert s2_data["content"]["plan"] == "Draft 2 from AI pipeline"
    assert s2_data["previous_snapshot_id"] == s1.id
    assert s2_data["regeneration_reason"] == "Source data changed"
    assert s2_data["regenerated_at"] is not None
    assert s2_data["source_changed"] is True
    mock_service.generate_career_plan.assert_called_once()

    # Verify old version remains available and unaltered in DB
    v1_in_db = db.query(AIInsightSnapshot).filter(AIInsightSnapshot.id == s1.id).first()
    assert v1_in_db is not None
    assert v1_in_db.version == 1
    assert v1_in_db.generation_id == gen1_id
    assert json.loads(v1_in_db.content)["plan"] == "Draft 1"

    # Total snapshots in DB for this scope is now 2
    total = (
        db.query(AIInsightSnapshot)
        .filter(
            AIInsightSnapshot.feature == "career_coach",
            AIInsightSnapshot.scope_employee_id == "EMP-ALICE",
        )
        .count()
    )
    assert total == 2


def test_regeneration_records_source_hash_change_and_lineage(db):
    """Regeneration compares structured source context and records explicit lineage."""
    first = save_insight_snapshot(
        db=db,
        feature="performance_insight",
        content={"score": 90},
        scope_employee_id="EMP-ALICE",
        scope_department="Engineering",
        period="2026-Q3",
        request_payload={"employee_id": "EMP-ALICE", "period": "2026-Q3"},
        context={"approved_sources": {"performance:1": {"score": 90}}},
        ai_service=MagicMock(model="test-model"),
    )
    caller = CallerContext("EMP-ALICE", "employee", "Engineering")

    def generator(previous):
        assert previous.id == first.id
        service = MagicMock(model="test-model")
        return (
            {"score": 95},
            {"approved_sources": {"performance:1": {"score": 95}}},
            service,
            {"employee_id": "EMP-ALICE", "period": "2026-Q3"},
            "2026-Q3",
        )

    second = regenerate_insight_snapshot(
        db=db,
        caller=caller,
        previous_snapshot_id=first.id,
        generator=generator,
        regeneration_reason="Approved performance score changed",
    )

    assert second.previous_snapshot_id == first.id
    assert second.version == 2
    assert second.source_changed is True
    assert second.source_hash != first.source_hash
    assert second.regenerated_at is not None
    assert second.regeneration_reason == "Approved performance score changed"
    assert db.query(AIInsightSnapshot).filter(AIInsightSnapshot.id == first.id).one().version == 1


# ==============================================================================
# 4. AI Feedback Capture
# ==============================================================================


def test_feedback_capture_linked_to_specific_snapshot(db, client):
    """Users can submit helpfulness ratings and optional feedback text for an insight snapshot."""
    snapshot = save_insight_snapshot(
        db=db,
        feature="skill_gap",
        content={"gaps": ["Kubernetes", "Docker"]},
        scope_employee_id="EMP-ALICE",
        scope_department="Engineering",
        period="2026-Q3",
    )

    headers = {
        "X-Caller-Employee-ID": "EMP-ALICE",
        "X-Caller-Role": "employee",
    }
    feedback_payload = {
        "is_helpful": True,
        "feedback_text": "Accurate skill identification, roadmap recommendations were very practical.",
    }
    response = client.post(
        f"/api/insights/snapshots/{snapshot.id}/feedback",
        headers=headers,
        json=feedback_payload,
    )

    assert response.status_code == 201
    fb_data = response.json()
    assert fb_data["snapshot_id"] == snapshot.id
    assert fb_data["is_helpful"] is True
    assert "Accurate skill identification" in fb_data["feedback_text"]
    assert fb_data["actor_employee_id"] == "EMP-ALICE"
    assert fb_data["actor_role"] == "employee"

    # Verify record in DB
    fb_record = db.query(AIFeedback).filter(AIFeedback.snapshot_id == snapshot.id).first()
    assert fb_record is not None
    assert fb_record.is_helpful is True
    assert fb_record.feedback_text == feedback_payload["feedback_text"]

    # Verify retrieval endpoint
    get_fb_resp = client.get(
        f"/api/insights/snapshots/{snapshot.id}/feedback",
        headers=headers,
    )
    assert get_fb_resp.status_code == 200
    feedbacks = get_fb_resp.json()
    assert len(feedbacks) == 1
    assert feedbacks[0]["id"] == fb_data["id"]


# ==============================================================================
# 5. Scope Isolation & Authorization Enforcement
# ==============================================================================


def test_unauthorized_employee_cannot_access_other_employee_snapshots(db, client):
    """Employee Bob cannot view Alice's insight snapshots or history."""
    snapshot = save_insight_snapshot(
        db=db,
        feature="evaluation_draft",
        content={"narrative": "Alice achieved exceptional performance."},
        scope_employee_id="EMP-ALICE",
        scope_department="Engineering",
        period="2026-Q3",
    )

    # Bob attempts to get Alice's snapshot
    bob_headers = {
        "X-Caller-Employee-ID": "EMP-BOB",
        "X-Caller-Role": "employee",
    }
    resp = client.get(f"/api/insights/snapshots/{snapshot.id}", headers=bob_headers)
    assert resp.status_code == 403

    # Bob attempts to query Alice's history
    hist_resp = client.get(
        "/api/insights/history?feature=evaluation_draft&employee_id=EMP-ALICE",
        headers=bob_headers,
    )
    assert hist_resp.status_code == 403

    # Bob attempts to submit feedback on Alice's snapshot
    fb_resp = client.post(
        f"/api/insights/snapshots/{snapshot.id}/feedback",
        headers=bob_headers,
        json={"is_helpful": False, "feedback_text": "Unauthorized comment"},
    )
    assert fb_resp.status_code == 403


def test_unauthorized_manager_cannot_access_other_department_snapshots(db, client):
    """Engineering manager Carol cannot access Bob's snapshots in Sales department."""
    bob_snapshot = save_insight_snapshot(
        db=db,
        feature="team_insight",
        content={"summary": "Sales team insights"},
        scope_employee_id=None,
        scope_department="Sales",
        period="2026-Q3",
    )

    mgr_headers = {
        "X-Caller-Employee-ID": "EMP-MGR",
        "X-Caller-Role": "manager",
    }
    resp = client.get(f"/api/insights/snapshots/{bob_snapshot.id}", headers=mgr_headers)
    assert resp.status_code == 403


def test_missing_or_invalid_headers_rejected_with_401(client):
    """Missing trusted gateway headers return 401 Unauthorized."""
    resp = client.get("/api/insights/history?feature=career_coach")
    assert resp.status_code == 401


# ==============================================================================
# 6. Automatic Snapshot Persistence in Existing AI Routes
# ==============================================================================


def test_career_coach_endpoint_persists_snapshot_automatically(db, client):
    """Calling existing Career Coach AI route persists a snapshot without altering response contract."""
    from app.schemas.career_coach import (
        CareerCoachSuccessResponse,
        DevelopmentAreaItem,
        DevelopmentPlanAction,
        EvidenceItem,
        FollowUp,
        PriorityLevel,
        StrengthItem,
    )

    mock_response = CareerCoachSuccessResponse(
        employee_id="EMP-ALICE",
        strengths=[
            StrengthItem(
                title="Python Mastery",
                description="Strong proficiency in backend services.",
                evidence=[
                    EvidenceItem(
                        source_type="skill",
                        source_id=1,
                        claim="Proficient in backend Python services.",
                    )
                ],
            )
        ],
        development_areas=[
            DevelopmentAreaItem(
                title="Cloud Architecture",
                description="Need deeper exposure to distributed design.",
                priority=PriorityLevel.HIGH,
                evidence=[
                    EvidenceItem(
                        source_type="skill",
                        source_id=2,
                        claim="Basic cloud architecture knowledge observed.",
                    )
                ],
            )
        ],
        development_plan=[
            DevelopmentPlanAction(
                action="Lead architecture design session.",
                reason="Strengthen cross-functional technical leadership.",
                measurable_target="Conduct 2 peer review sessions.",
                suggested_timeline="Q4 2026",
            )
        ],
        follow_up=FollowUp(
            checkpoint="End of Q4 2026",
            review_focus="Evaluate design review deliverables.",
        ),
    )

    from app.api.career_coach import get_career_coach_ai_service

    mock_svc = MagicMock()
    mock_svc.generate_career_plan.return_value = mock_response

    app.dependency_overrides[get_career_coach_ai_service] = lambda: mock_svc
    try:
        headers = {
            "X-Caller-Employee-ID": "EMP-ALICE",
            "X-Caller-Role": "employee",
        }
        res = client.post(
            "/api/career-coach",
            headers=headers,
            json={"employee_id": "EMP-ALICE", "period": "2026-Q3"},
        )

        assert res.status_code == 200
        data = res.json()
        assert data["employee_id"] == "EMP-ALICE"
        assert data["strengths"][0]["title"] == "Python Mastery"
    finally:
        app.dependency_overrides.pop(get_career_coach_ai_service, None)

    # Verify snapshot was automatically recorded in DB
    snapshot = (
        db.query(AIInsightSnapshot)
        .filter(
            AIInsightSnapshot.feature == "career_coach",
            AIInsightSnapshot.scope_employee_id == "EMP-ALICE",
        )
        .first()
    )
    assert snapshot is not None
    assert snapshot.version == 1
    assert snapshot.scope_employee_id == "EMP-ALICE"
    assert "Python Mastery" in snapshot.content

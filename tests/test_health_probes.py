"""Focused tests for AI Liveness and Readiness Probes.

Covers:
- Liveness returns 200 without requiring DB
- Readiness returns 200 when DB and provider configuration are available
- Readiness returns 503 when DB is unavailable
- Readiness returns 503 when required provider configuration is missing or invalid
- Readiness does not expose secrets (keys, passwords, connection strings, stack traces)
- Liveness remains independent from DB / provider failures (stays 200 even when ready returns 503)
- Critical DB initialization failure cannot make readiness falsely report healthy
- Backwards compatibility of legacy /health endpoint
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.session import Base, get_db
from app.main import app

TEST_DATABASE_URL = "sqlite:///:memory:"

test_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture(scope="function")
def db_session():
    """Provides an isolated database session."""
    Base.metadata.create_all(bind=test_engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=test_engine)


@pytest.fixture(scope="function")
def client(db_session: Session):
    """FastAPI TestClient with get_db overridden."""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    # Reset app state between test runs
    app.state.db_initialized = True
    app.state.db_initialization_error = None

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
    app.state.db_initialized = True
    app.state.db_initialization_error = None


# =============================================================================
# 1. LIVENESS PROBE TESTS
# =============================================================================


def test_liveness_returns_200_without_requiring_db(client: TestClient):
    """Liveness probe indicates application process is running without checking DB."""
    resp = client.get("/health/live")
    assert resp.status_code == 200
    assert resp.json() == {"status": "alive"}

    # Also test /live alias
    resp_alias = client.get("/live")
    assert resp_alias.status_code == 200
    assert resp_alias.json() == {"status": "alive"}


def test_liveness_remains_200_even_when_db_and_provider_fail(client: TestClient, monkeypatch):
    """Liveness remains completely independent from DB and AI provider availability."""
    # Break DB dependency
    mock_db = MagicMock()
    mock_db.execute.side_effect = OperationalError("connection refused", {}, None)
    app.dependency_overrides[get_db] = lambda: iter([mock_db])

    # Break provider configuration
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    # Set critical DB initialization error
    app.state.db_initialization_error = "FATAL: schema initialization failed"

    # Readiness should fail (503)
    ready_resp = client.get("/health/ready")
    assert ready_resp.status_code == 503

    # But Liveness MUST stay 200 (alive)
    live_resp = client.get("/health/live")
    assert live_resp.status_code == 200
    assert live_resp.json()["status"] == "alive"


# =============================================================================
# 2. READINESS PROBE TESTS: SUCCESS SCENARIO
# =============================================================================


def test_readiness_returns_200_when_db_and_provider_are_available(
    client: TestClient, monkeypatch
):
    """Readiness probe returns 200 when DB is connected and provider is configured."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_valid_mock_key_12345")
    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-120b")

    resp = client.get("/health/ready")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ready", "reason": None}

    # Also test /ready and /api/health/ready aliases
    resp_alias = client.get("/ready")
    assert resp_alias.status_code == 200
    assert resp_alias.json()["status"] == "ready"

    resp_api = client.get("/api/health/ready")
    assert resp_api.status_code == 200
    assert resp_api.json()["status"] == "ready"


# =============================================================================
# 3. READINESS PROBE TESTS: DATABASE FAILURE SCENARIOS
# =============================================================================


def test_readiness_returns_503_when_db_ping_fails(client: TestClient, monkeypatch):
    """Readiness returns 503 with reason 'database_unavailable' when DB connection check fails."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_valid_mock_key_12345")

    mock_db = MagicMock()
    mock_db.execute.side_effect = OperationalError("Can't connect to MySQL server", {}, None)
    app.dependency_overrides[get_db] = lambda: iter([mock_db])

    resp = client.get("/health/ready")
    assert resp.status_code == 503
    data = resp.json()
    assert data["status"] == "not_ready"
    assert data["reason"] == "database_unavailable"


def test_readiness_returns_503_when_db_session_cannot_be_acquired(
    client: TestClient, monkeypatch
):
    """Readiness returns 503 when database session generator raises an exception."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_valid_mock_key_12345")

    def broken_get_db():
        raise OperationalError("Connection pool exhausted", {}, None)

    app.dependency_overrides[get_db] = broken_get_db

    resp = client.get("/health/ready")
    assert resp.status_code == 503
    data = resp.json()
    assert data["status"] == "not_ready"
    assert data["reason"] == "database_unavailable"


def test_critical_db_init_failure_cannot_make_readiness_falsely_healthy(
    client: TestClient, monkeypatch
):
    """If critical DB schema initialization failed at startup, readiness returns 503."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_valid_mock_key_12345")

    # Simulate critical startup DB initialization failure
    app.state.db_initialization_error = "Table creation failed during startup"
    app.state.db_initialized = False

    resp = client.get("/health/ready")
    assert resp.status_code == 503
    data = resp.json()
    assert data["status"] == "not_ready"
    assert data["reason"] == "database_initialization_failed"


# =============================================================================
# 4. READINESS PROBE TESTS: AI PROVIDER CONFIGURATION FAILURES
# =============================================================================


def test_readiness_returns_503_when_groq_api_key_is_missing(client: TestClient, monkeypatch):
    """Readiness returns 503 when GROQ_API_KEY environment variable is not set."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    resp = client.get("/health/ready")
    assert resp.status_code == 503
    data = resp.json()
    assert data["status"] == "not_ready"
    assert data["reason"] == "ai_provider_unconfigured"


def test_readiness_returns_503_when_groq_api_key_is_empty_or_whitespace(
    client: TestClient, monkeypatch
):
    """Readiness returns 503 when GROQ_API_KEY is empty or only whitespace."""
    monkeypatch.setenv("GROQ_API_KEY", "   ")

    resp = client.get("/health/ready")
    assert resp.status_code == 503
    data = resp.json()
    assert data["status"] == "not_ready"
    assert data["reason"] == "ai_provider_unconfigured"


def test_readiness_returns_503_when_groq_model_is_invalid(client: TestClient, monkeypatch):
    """Readiness returns 503 when GROQ_MODEL is explicitly configured to empty/whitespace."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_valid_mock_key_12345")
    monkeypatch.setenv("GROQ_MODEL", "   ")

    resp = client.get("/health/ready")
    assert resp.status_code == 503
    data = resp.json()
    assert data["status"] == "not_ready"
    assert data["reason"] == "invalid_ai_provider_configuration"


# =============================================================================
# 5. SECRECY & DATA PROTECTION TESTS
# =============================================================================


def test_readiness_does_not_expose_secrets_or_stack_traces(client: TestClient, monkeypatch):
    """Readiness responses must never expose API keys, DB credentials, or stack traces."""
    secret_key = "gsk_super_secret_production_key_xyz987"
    monkeypatch.setenv("GROQ_API_KEY", secret_key)

    # 1. Success case
    resp_success = client.get("/health/ready")
    assert resp_success.status_code == 200
    success_text = resp_success.text
    assert secret_key not in success_text
    assert "gsk_" not in success_text

    # 2. Database failure case with sensitive exception message
    mock_db = MagicMock()
    mock_db.execute.side_effect = OperationalError(
        "Access denied for user 'root'@'127.0.0.1' (using password: YES)", {}, None
    )
    app.dependency_overrides[get_db] = lambda: iter([mock_db])

    resp_fail = client.get("/health/ready")
    assert resp_fail.status_code == 503
    fail_text = resp_fail.text
    assert "password" not in fail_text.lower()
    assert "root" not in fail_text
    assert "127.0.0.1" not in fail_text
    assert "traceback" not in fail_text.lower()
    assert secret_key not in fail_text

    # 3. Critical DB init error containing sensitive path or details
    app.state.db_initialization_error = "Sensitive error with mysql://user:pass@host/db"
    resp_init_fail = client.get("/health/ready")
    assert resp_init_fail.status_code == 503
    init_text = resp_init_fail.text
    assert "user:pass" not in init_text
    assert "mysql://" not in init_text
    assert resp_init_fail.json() == {
        "status": "not_ready",
        "reason": "database_initialization_failed",
    }


# =============================================================================
# 6. BACKWARDS COMPATIBILITY
# =============================================================================


def test_legacy_health_check_preserved(client: TestClient):
    """The legacy /health endpoint continues returning HTTP 200 {"status": "ok", ...}."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "Smart HR Management System" in data["service"]


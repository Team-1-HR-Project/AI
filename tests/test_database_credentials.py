"""Focused tests for explicit database credentials and removal of root/blank-password fallbacks.

Covers:
- Explicit valid DB credentials are accepted
- Missing DB username is rejected
- Missing DB password is rejected
- Empty/whitespace DB password is rejected
- No fallback to root/blank-password credentials occurs
- DATABASE_URL with blank password is rejected
- DATABASE_URL with missing username is rejected
- Existing readiness behavior correctly reports configuration failure (503)
- Liveness remains independent and returns 200
- No credentials/secrets appear in error messages or responses
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.db.session import (
    DatabaseConfigurationError,
    get_database_url,
    init_engine,
)
from app.main import app

# =============================================================================
# 1. CREDENTIAL VALIDATION & REJECTION TESTS
# =============================================================================


def test_explicit_valid_db_credentials_accepted(monkeypatch):
    """Explicit valid username and password from environment variables are accepted."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("DB_USER", "ai_service_user")
    monkeypatch.setenv("DB_PASSWORD", "StrongSecretPassword987#")
    monkeypatch.setenv("DB_HOST", "db-internal.example.com")
    monkeypatch.setenv("DB_PORT", "3306")
    monkeypatch.setenv("DB_NAME", "hr_system")

    url = get_database_url()
    assert "ai_service_user" in url
    assert "db-internal.example.com" in url
    assert "hr_system" in url
    assert "mysql+pymysql://" in url


def test_explicit_valid_database_url_accepted(monkeypatch):
    """Explicit valid connection string with credentials in DATABASE_URL is accepted."""
    valid_url = "mysql+pymysql://prod_agent:ComplexPass456!@10.0.0.5:3306/hr_db?charset=utf8mb4"
    monkeypatch.setenv("DATABASE_URL", valid_url)

    url = get_database_url()
    assert url == valid_url


def test_sqlite_database_url_accepted_without_credentials(monkeypatch):
    """SQLite in-memory or file database URLs are accepted without user/password."""
    sqlite_url = "sqlite:///:memory:"
    monkeypatch.setenv("DATABASE_URL", sqlite_url)

    url = get_database_url()
    assert url == sqlite_url


def test_missing_db_username_rejected(monkeypatch):
    """Missing DB_USER with no DATABASE_URL raises DatabaseConfigurationError."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("DB_USER", raising=False)
    monkeypatch.setenv("DB_PASSWORD", "SomePassword123")

    with pytest.raises(DatabaseConfigurationError) as exc_info:
        get_database_url()

    assert "username" in str(exc_info.value).lower()
    # Confirm no silent fallback to root occurred
    assert "root" not in str(exc_info.value).lower()


def test_missing_db_password_rejected(monkeypatch):
    """Missing DB_PASSWORD raises DatabaseConfigurationError without defaulting to empty string."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("DB_USER", "ai_worker")
    monkeypatch.delenv("DB_PASSWORD", raising=False)

    with pytest.raises(DatabaseConfigurationError) as exc_info:
        get_database_url()

    assert "password" in str(exc_info.value).lower()


def test_empty_db_password_rejected(monkeypatch):
    """Empty or whitespace-only DB_PASSWORD is explicitly rejected."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("DB_USER", "ai_worker")
    monkeypatch.setenv("DB_PASSWORD", "   ")

    with pytest.raises(DatabaseConfigurationError) as exc_info:
        get_database_url()

    assert "password" in str(exc_info.value).lower()


def test_no_fallback_to_root_or_blank_password(monkeypatch):
    """When no credentials are provided, no implicit fallback to root:empty occurs."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("DB_USER", raising=False)
    monkeypatch.delenv("DB_PASSWORD", raising=False)

    with pytest.raises(DatabaseConfigurationError) as exc_info:
        get_database_url()

    # Must fail on missing credentials and not silently return mysql://root:@127.0.0.1
    assert "missing explicit database username" in str(exc_info.value).lower()


def test_database_url_with_blank_password_rejected(monkeypatch):
    """DATABASE_URL pointing to MySQL with an empty password is strictly rejected."""
    blank_pass_url = "mysql+pymysql://root:@127.0.0.1:3306/hr_system?charset=utf8mb4"
    monkeypatch.setenv("DATABASE_URL", blank_pass_url)

    with pytest.raises(DatabaseConfigurationError) as exc_info:
        get_database_url()

    assert "password" in str(exc_info.value).lower()
    assert "blank passwords" in str(exc_info.value).lower()


def test_database_url_with_missing_username_rejected(monkeypatch):
    """DATABASE_URL missing a username is strictly rejected."""
    missing_user_url = "mysql+pymysql://@127.0.0.1:3306/hr_system"
    monkeypatch.setenv("DATABASE_URL", missing_user_url)

    with pytest.raises(DatabaseConfigurationError) as exc_info:
        get_database_url()

    assert "username" in str(exc_info.value).lower()


# =============================================================================
# 2. READINESS & LIVENESS BEHAVIOR ON CONFIGURATION FAILURE
# =============================================================================


def test_readiness_reports_failure_when_database_unconfigured(monkeypatch):
    """Readiness probe returns HTTP 503 when required database credentials are not configured."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("DB_USER", raising=False)
    monkeypatch.delenv("DB_PASSWORD", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_mock_key_12345")

    # Clear dependency overrides to test natural unconfigured state
    app.dependency_overrides.clear()
    init_engine()

    with TestClient(app) as client:
        # Readiness MUST fail with 503
        ready_resp = client.get("/health/ready")
        assert ready_resp.status_code == 503
        data = ready_resp.json()
        assert data["status"] == "not_ready"
        assert data["reason"] in ("database_unavailable", "database_configuration_invalid", "database_initialization_failed")

        # Liveness MUST remain 200 (alive)
        live_resp = client.get("/health/live")
        assert live_resp.status_code == 200
        assert live_resp.json()["status"] == "alive"


# =============================================================================
# 3. SECRECY & DATA PROTECTION TESTS
# =============================================================================


def test_no_credentials_or_secrets_exposed_in_errors(monkeypatch):
    """Database configuration error messages must never expose passwords."""
    sensitive_pass = "SuperSecret_Sensitive_Password_999!"
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("DB_USER", "test_user")
    monkeypatch.setenv("DB_PASSWORD", sensitive_pass)

    # Valid config shouldn't raise
    url = get_database_url()
    assert "SuperSecret_Sensitive_Password_999" in url

    # Test error message when password is empty
    monkeypatch.setenv("DB_PASSWORD", "")
    with pytest.raises(DatabaseConfigurationError) as exc_info:
        get_database_url()

    error_str = str(exc_info.value)
    assert sensitive_pass not in error_str
    assert "password" in error_str.lower()

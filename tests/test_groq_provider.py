"""Comprehensive unit tests for the centralized Groq provider module.

Covers:
1. Validation of missing/empty API key
2. Validation of missing/empty/invalid model name
3. Custom base URL handling
4. Timeout and deadline configuration
5. Retry on transient errors (RateLimitError, APIConnectionError, APITimeoutError, 5xx APIError)
6. No retry on 4xx/client errors (AuthenticationError, BadRequestError)
7. Respect deadline exceeding (before call and during retry)
8. Correct client construction
9. Successful completion execution and empty response detection
10. Service integration tests verifying services use the centralized provider
"""

from __future__ import annotations

import os
import time
from unittest.mock import MagicMock, patch

import pytest
from groq import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    RateLimitError,
)

from app.core.groq_provider import (
    DEFAULT_DEADLINE_SECONDS,
    DEFAULT_GROQ_BASE_URL,
    DEFAULT_GROQ_MODEL,
    DEFAULT_TIMEOUT_SECONDS,
    MAX_RETRIES,
    GroqClientError,
    GroqConfigurationError,
    GroqDeadlineExceededError,
    GroqEmptyResponseError,
    GroqTimeoutError,
    create_groq_client,
    execute_chat_completion,
    resolve_groq_config,
    validate_groq_environment,
)
from app.services.attention_signal_ai import AttentionSignalAIService
from app.services.career_coach_ai import CareerCoachAIService
from app.services.evaluation_draft_ai import EvaluationDraftAIService
from app.services.performance_insight_ai import PerformanceInsightAIService
from app.services.policy_ai import PolicyAIService
from app.services.skill_gap_ai import SkillGapAIService
from app.services.team_insight_ai import TeamInsightAIService

# =============================================================================
# 1. Validation of missing / empty API key
# =============================================================================


def test_missing_api_key_when_required_raises_error():
    """Missing API key when require_api_key=True raises GroqConfigurationError."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(GroqConfigurationError) as exc:
            resolve_groq_config(api_key=None, require_api_key=True)
        assert "GROQ_API_KEY is not configured" in str(exc.value)


def test_empty_or_whitespace_api_key_when_required_raises_error():
    """Empty or whitespace API key when require_api_key=True raises GroqConfigurationError."""
    with patch.dict(os.environ, {"GROQ_API_KEY": "   "}, clear=True):
        with pytest.raises(GroqConfigurationError) as exc:
            resolve_groq_config(api_key="", require_api_key=True)
        assert "GROQ_API_KEY is not configured" in str(exc.value)


def test_create_groq_client_without_api_key_raises_error():
    """create_groq_client raises GroqConfigurationError when no API key is available."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(GroqConfigurationError) as exc:
            create_groq_client()
        assert "GROQ_API_KEY is not configured" in str(exc.value)


def test_validate_groq_environment_flags_missing_api_key():
    """validate_groq_environment returns unconfigured when key is missing or blank."""
    with patch.dict(os.environ, {}, clear=True):
        ok, reason = validate_groq_environment()
        assert not ok
        assert reason == "ai_provider_unconfigured"

    with patch.dict(os.environ, {"GROQ_API_KEY": "  \t "}, clear=True):
        ok, reason = validate_groq_environment()
        assert not ok
        assert reason == "ai_provider_unconfigured"


# =============================================================================
# 2. Validation of missing / empty / invalid model name
# =============================================================================


def test_missing_or_empty_model_raises_error():
    """Explicitly empty model name fails fast with GroqConfigurationError."""
    with pytest.raises(GroqConfigurationError) as exc:
        resolve_groq_config(model="")
    assert "GROQ_MODEL configuration is missing or invalid" in str(exc.value)

    with pytest.raises(GroqConfigurationError) as exc:
        resolve_groq_config(model="   ")
    assert "GROQ_MODEL configuration is missing or invalid" in str(exc.value)


def test_empty_model_in_env_raises_error():
    """Empty GROQ_MODEL in environment fails fast with GroqConfigurationError."""
    with patch.dict(os.environ, {"GROQ_MODEL": "   "}):
        with pytest.raises(GroqConfigurationError) as exc:
            resolve_groq_config()
        assert "GROQ_MODEL configuration is missing or invalid" in str(exc.value)


def test_default_model_applied_when_not_specified():
    """When model is not specified and env is unset, standard default is applied."""
    with patch.dict(os.environ, {}, clear=True):
        config = resolve_groq_config()
        assert config.model == DEFAULT_GROQ_MODEL
        assert config.model == "openai/gpt-oss-120b"


def test_custom_model_preserved():
    """Valid custom model name is stripped and preserved."""
    config = resolve_groq_config(model="  llama-3.3-70b-versatile  ")
    assert config.model == "llama-3.3-70b-versatile"


# =============================================================================
# 3. Custom base URL handling
# =============================================================================


def test_default_base_url_applied():
    """Default base URL is https://api.groq.com."""
    with patch.dict(os.environ, {}, clear=True):
        config = resolve_groq_config()
        assert config.base_url == DEFAULT_GROQ_BASE_URL
        assert config.base_url == "https://api.groq.com"


def test_custom_base_url_from_argument():
    """Explicit base_url argument is respected."""
    config = resolve_groq_config(base_url="https://custom.groq.proxy/v1")
    assert config.base_url == "https://custom.groq.proxy/v1"


def test_custom_base_url_from_env():
    """GROQ_BASE_URL environment variable is respected."""
    with patch.dict(os.environ, {"GROQ_BASE_URL": "https://gateway.internal.corp/ai"}):
        config = resolve_groq_config()
        assert config.base_url == "https://gateway.internal.corp/ai"


def test_create_groq_client_passes_base_url():
    """create_groq_client passes configured base_url to Groq constructor."""
    with patch("app.core.groq_provider.Groq") as mock_groq_cls:
        config = resolve_groq_config(
            api_key="gsk_test_123",
            base_url="https://custom.groq.endpoint",
        )
        create_groq_client(config)
        mock_groq_cls.assert_called_once_with(
            api_key="gsk_test_123",
            base_url="https://custom.groq.endpoint",
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )


# =============================================================================
# 4. Timeout and deadline configuration
# =============================================================================


def test_default_timeout_and_deadline():
    """Standard defaults are 30.0s timeout and 25.0s deadline."""
    with patch.dict(os.environ, {}, clear=True):
        config = resolve_groq_config()
        assert config.timeout == DEFAULT_TIMEOUT_SECONDS
        assert config.timeout == 30.0
        assert config.deadline == DEFAULT_DEADLINE_SECONDS
        assert config.deadline == 25.0
        assert config.max_retries == MAX_RETRIES
        assert config.max_retries == 2


def test_timeout_and_deadline_from_env():
    """Timeouts and deadlines can be configured via environment variables."""
    with patch.dict(
        os.environ,
        {
            "GROQ_TIMEOUT_SECONDS": "45.5",
            "GROQ_DEADLINE_SECONDS": "40.0",
            "GROQ_MAX_RETRIES": "3",
        },
    ):
        config = resolve_groq_config()
        assert config.timeout == 45.5
        assert config.deadline == 40.0
        assert config.max_retries == 3


def test_invalid_timeout_and_deadline_values_rejected():
    """Non-positive or non-numeric timeout/deadline values raise GroqConfigurationError."""
    with pytest.raises(GroqConfigurationError):
        resolve_groq_config(timeout=0)

    with pytest.raises(GroqConfigurationError):
        resolve_groq_config(timeout=-5.0)

    with pytest.raises(GroqConfigurationError):
        resolve_groq_config(deadline=0)

    with pytest.raises(GroqConfigurationError):
        resolve_groq_config(deadline=-1.0)


# =============================================================================
# 5. Retry on transient errors (RateLimitError, APIConnectionError, APITimeoutError, 5xx)
# =============================================================================


def test_retry_on_rate_limit_error_succeeds_on_second_attempt():
    """RateLimitError triggers retry with backoff and succeeds on subsequent attempt."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = '{"result": "success"}'
    mock_response.choices = [mock_choice]

    mock_client.chat.completions.create.side_effect = [
        RateLimitError(message="Rate limit reached", response=MagicMock(), body=None),
        mock_response,
    ]

    with patch("time.sleep"):
        result = execute_chat_completion(
            client=mock_client,
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": "hello"}],
            max_retries=2,
        )

    assert result == '{"result": "success"}'
    assert mock_client.chat.completions.create.call_count == 2


def test_retry_on_connection_error_succeeds_on_second_attempt():
    """APIConnectionError triggers retry and succeeds on subsequent attempt."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = '{"connected": true}'
    mock_response.choices = [mock_choice]

    mock_client.chat.completions.create.side_effect = [
        APIConnectionError(request=MagicMock()),
        mock_response,
    ]

    with patch("time.sleep"):
        result = execute_chat_completion(
            client=mock_client,
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": "ping"}],
            max_retries=2,
        )

    assert result == '{"connected": true}'
    assert mock_client.chat.completions.create.call_count == 2


def test_retry_on_api_timeout_error_succeeds_on_second_attempt():
    """APITimeoutError triggers retry and succeeds on subsequent attempt."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = '{"completed": true}'
    mock_response.choices = [mock_choice]

    mock_client.chat.completions.create.side_effect = [
        APITimeoutError(request=MagicMock()),
        mock_response,
    ]

    with patch("time.sleep"):
        result = execute_chat_completion(
            client=mock_client,
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": "ping"}],
            max_retries=2,
        )

    assert result == '{"completed": true}'
    assert mock_client.chat.completions.create.call_count == 2


def test_retry_on_5xx_api_error_succeeds():
    """APIError with 503 status code triggers retry and succeeds."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = '{"recovered": true}'
    mock_response.choices = [mock_choice]

    error_503 = APIError(message="Service Unavailable", request=MagicMock(), body=None)
    error_503.status_code = 503

    mock_client.chat.completions.create.side_effect = [
        error_503,
        mock_response,
    ]

    with patch("time.sleep"):
        result = execute_chat_completion(
            client=mock_client,
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": "test"}],
            max_retries=2,
        )

    assert result == '{"recovered": true}'
    assert mock_client.chat.completions.create.call_count == 2


def test_exhausted_retries_raises_provider_error():
    """When retries are exhausted, GroqTimeoutError/GroqMaxRetriesExceededError is raised."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = APITimeoutError(request=MagicMock())

    with patch("time.sleep"):
        with pytest.raises(GroqTimeoutError) as exc:
            execute_chat_completion(
                client=mock_client,
                model="openai/gpt-oss-120b",
                messages=[{"role": "user", "content": "test"}],
                max_retries=1,
            )
        assert "failed after 2 attempts" in str(exc.value)
        assert "Provider connection timeout" in str(exc.value)
    assert mock_client.chat.completions.create.call_count == 2


# =============================================================================
# 6. No retry on 4xx / client errors
# =============================================================================


def test_no_retry_on_authentication_error():
    """AuthenticationError fails fast immediately without retrying."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = AuthenticationError(
        message="Invalid API Key",
        response=MagicMock(),
        body=None,
    )

    with pytest.raises(GroqClientError) as exc:
        execute_chat_completion(
            client=mock_client,
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": "test"}],
            max_retries=2,
        )

    assert "configuration or request formatting error" in str(exc.value)
    assert mock_client.chat.completions.create.call_count == 1


def test_no_retry_on_bad_request_error():
    """BadRequestError fails fast immediately without retrying."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = BadRequestError(
        message="Bad Request syntax",
        response=MagicMock(),
        body=None,
    )

    with pytest.raises(GroqClientError) as exc:
        execute_chat_completion(
            client=mock_client,
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": "test"}],
            max_retries=2,
        )

    assert "configuration or request formatting error" in str(exc.value)
    assert mock_client.chat.completions.create.call_count == 1


def test_no_retry_on_4xx_api_error():
    """APIError with 400 status code fails fast without retrying."""
    mock_client = MagicMock()
    error_400 = APIError(message="Invalid parameters", request=MagicMock(), body=None)
    error_400.status_code = 400
    mock_client.chat.completions.create.side_effect = error_400

    with pytest.raises(GroqClientError) as exc:
        execute_chat_completion(
            client=mock_client,
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": "test"}],
            max_retries=2,
        )

    assert "configuration or request formatting error" in str(exc.value)
    assert mock_client.chat.completions.create.call_count == 1


# =============================================================================
# 7. Respect deadline exceeding
# =============================================================================


def test_deadline_already_exceeded_fails_fast_without_calling_client():
    """If deadline has already elapsed, fails immediately without invoking Groq."""
    mock_client = MagicMock()
    expired_deadline = time.monotonic() - 10.0

    with pytest.raises(GroqDeadlineExceededError) as exc:
        execute_chat_completion(
            client=mock_client,
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": "test"}],
            deadline=expired_deadline,
        )

    assert "deadline exceeded" in str(exc.value)
    mock_client.chat.completions.create.assert_not_called()


def test_deadline_expires_during_retry_backoff():
    """If deadline would be exceeded during backoff, fails immediately with deadline error."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = APITimeoutError(request=MagicMock())

    # Set deadline to expire right after attempt 1
    short_deadline = time.monotonic() + 0.05

    with pytest.raises(GroqDeadlineExceededError) as exc:
        execute_chat_completion(
            client=mock_client,
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": "test"}],
            deadline=short_deadline,
            max_retries=2,
            initial_backoff=1.0,  # backoff exceeds remaining deadline
        )

    assert "deadline exceeded" in str(exc.value)
    assert mock_client.chat.completions.create.call_count == 1


# =============================================================================
# 8. Correct client construction
# =============================================================================


def test_create_groq_client_with_explicit_parameters():
    """create_groq_client instantiates Groq with exact key, base_url, and timeout."""
    with patch("app.core.groq_provider.Groq") as mock_groq_cls:
        create_groq_client(
            api_key="gsk_custom_key_xyz",
            base_url="https://custom.groq.com/v1",
            timeout=15.0,
        )
        mock_groq_cls.assert_called_once_with(
            api_key="gsk_custom_key_xyz",
            base_url="https://custom.groq.com/v1",
            timeout=15.0,
        )


# =============================================================================
# 9. Successful completion execution and empty response detection
# =============================================================================


def test_successful_chat_completion():
    """execute_chat_completion successfully parses and strips response content."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "  {\"analysis\": \"complete\"}  \n"
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response

    result = execute_chat_completion(
        client=mock_client,
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": "prompt"}],
    )
    assert result == '{"analysis": "complete"}'


def test_empty_response_content_raises_empty_response_error():
    """Empty or whitespace response content raises GroqEmptyResponseError."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "   "
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response

    with pytest.raises(GroqEmptyResponseError) as exc:
        execute_chat_completion(
            client=mock_client,
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": "prompt"}],
        )
    assert "Empty response returned" in str(exc.value)


# =============================================================================
# 10. Service integration tests verifying all 7 services use centralized provider
# =============================================================================


def test_career_coach_service_uses_centralized_provider():
    """CareerCoachAIService resolves configuration and client via groq_provider."""
    mock_client = MagicMock()
    service = CareerCoachAIService(
        api_key="gsk_service_test",
        model="custom/model-cc",
        base_url="https://custom.groq.cc",
        client=mock_client,
        timeout=12.0,
        max_retries=1,
    )
    assert service.api_key == "gsk_service_test"
    assert service.model == "custom/model-cc"
    assert service.base_url == "https://custom.groq.cc"
    assert service.timeout == 12.0
    assert service.max_retries == 1
    assert service._get_client() is mock_client


def test_policy_ai_service_uses_centralized_provider():
    """PolicyAIService resolves configuration and client via groq_provider."""
    mock_client = MagicMock()
    service = PolicyAIService(
        api_key="gsk_service_test",
        model="custom/model-pol",
        base_url="https://custom.groq.pol",
        client=mock_client,
        timeout=14.0,
        max_retries=1,
    )
    assert service.api_key == "gsk_service_test"
    assert service.model == "custom/model-pol"
    assert service.base_url == "https://custom.groq.pol"
    assert service.timeout == 14.0
    assert service.max_retries == 1
    assert service._get_client() is mock_client


def test_performance_insight_service_uses_centralized_provider():
    """PerformanceInsightAIService resolves configuration and client via groq_provider."""
    mock_client = MagicMock()
    service = PerformanceInsightAIService(
        api_key="gsk_service_test",
        model="custom/model-pi",
        base_url="https://custom.groq.pi",
        client=mock_client,
        timeout=16.0,
        max_retries=1,
    )
    assert service.api_key == "gsk_service_test"
    assert service.model == "custom/model-pi"
    assert service.base_url == "https://custom.groq.pi"
    assert service.timeout == 16.0
    assert service.max_retries == 1
    assert service._get_client() is mock_client


def test_evaluation_draft_service_uses_centralized_provider():
    """EvaluationDraftAIService resolves configuration and client via groq_provider."""
    mock_client = MagicMock()
    service = EvaluationDraftAIService(
        api_key="gsk_service_test",
        model="custom/model-eval",
        base_url="https://custom.groq.eval",
        client=mock_client,
        timeout=18.0,
        max_retries=1,
    )
    assert service.api_key == "gsk_service_test"
    assert service.model == "custom/model-eval"
    assert service.base_url == "https://custom.groq.eval"
    assert service.timeout == 18.0
    assert service.max_retries == 1
    assert service._get_client() is mock_client


def test_skill_gap_service_uses_centralized_provider():
    """SkillGapAIService resolves configuration and client via groq_provider."""
    mock_client = MagicMock()
    service = SkillGapAIService(
        api_key="gsk_service_test",
        model="custom/model-sg",
        base_url="https://custom.groq.sg",
        client=mock_client,
        timeout=20.0,
        max_retries=1,
    )
    assert service.api_key == "gsk_service_test"
    assert service.model == "custom/model-sg"
    assert service.base_url == "https://custom.groq.sg"
    assert service.timeout == 20.0
    assert service.max_retries == 1
    assert service._get_client() is mock_client


def test_attention_signal_service_uses_centralized_provider():
    """AttentionSignalAIService resolves configuration and client via groq_provider."""
    mock_client = MagicMock()
    service = AttentionSignalAIService(
        api_key="gsk_service_test",
        model="custom/model-as",
        base_url="https://custom.groq.as",
        client=mock_client,
        timeout=22.0,
        max_retries=1,
    )
    assert service.api_key == "gsk_service_test"
    assert service.model == "custom/model-as"
    assert service.base_url == "https://custom.groq.as"
    assert service.timeout == 22.0
    assert service.max_retries == 1
    assert service._get_client() is mock_client


def test_team_insight_service_uses_centralized_provider():
    """TeamInsightAIService resolves configuration and client via groq_provider."""
    mock_client = MagicMock()
    service = TeamInsightAIService(
        api_key="gsk_service_test",
        model="custom/model-ti",
        base_url="https://custom.groq.ti",
        client=mock_client,
        timeout=24.0,
        max_retries=1,
    )
    assert service.api_key == "gsk_service_test"
    assert service.model == "custom/model-ti"
    assert service.base_url == "https://custom.groq.ti"
    assert service.timeout == 24.0
    assert service.max_retries == 1
    assert service._get_client() is mock_client


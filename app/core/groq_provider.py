"""Centralized Groq AI provider configuration, client management, and resilient execution.

Provides unified:
- Environment variable resolution and validation (API key, model, base URL, timeouts, retries)
- Client lifecycle management with consistent configuration
- Resilient chat completion execution with exponential backoff, jitter, and deadline enforcement
- Normalized provider error hierarchy
"""

from __future__ import annotations

import logging
import os
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from groq import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    Groq,
    RateLimitError,
)

logger = logging.getLogger(__name__)

# Standard Defaults
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
DEFAULT_GROQ_BASE_URL = "https://api.groq.com"
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_DEADLINE_SECONDS = 25.0
MAX_RETRIES = 2
INITIAL_BACKOFF_SECONDS = 0.5


# Exception Hierarchy
class GroqProviderError(Exception):
    """Base exception for all Groq provider operations."""


class GroqConfigurationError(GroqProviderError):
    """Raised when provider configuration (model, API key, timeouts) is invalid."""


class GroqDeadlineExceededError(GroqProviderError):
    """Raised when request deadline is exceeded before or during invocation."""


class GroqClientError(GroqProviderError):
    """Raised on non-retryable 4xx client errors (e.g. invalid request format, auth error)."""


class GroqTransientError(GroqProviderError):
    """Base class for transient provider errors."""


class GroqRateLimitError(GroqTransientError):
    """Raised when rate limit is exceeded after retries."""


class GroqTimeoutError(GroqTransientError):
    """Raised when provider connection or response times out."""


class GroqConnectionError(GroqTransientError):
    """Raised when provider connection fails."""


class GroqEmptyResponseError(GroqProviderError):
    """Raised when the provider returns an empty response body."""


class GroqMaxRetriesExceededError(GroqProviderError):
    """Raised when retries are exhausted for transient failures."""


@dataclass(frozen=True)
class GroqProviderConfig:
    """Immutable provider configuration."""

    api_key: str | None
    model: str
    base_url: str
    timeout: float
    deadline: float
    max_retries: int
    initial_backoff: float


def resolve_groq_config(
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    timeout: float | None = None,
    deadline: float | None = None,
    max_retries: int | None = None,
    initial_backoff: float | None = None,
    require_api_key: bool = False,
) -> GroqProviderConfig:
    """Resolves and validates Groq provider configuration from explicit args and environment."""
    # 1. API Key
    resolved_api_key = api_key if api_key is not None else os.getenv("GROQ_API_KEY")
    if resolved_api_key is not None:
        resolved_api_key = resolved_api_key.strip() or None

    if require_api_key and not resolved_api_key:
        raise GroqConfigurationError(
            "GROQ_API_KEY is not configured. Please set the GROQ_API_KEY environment variable."
        )

    # 2. Model
    configured_model = model if model is not None else os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
    if not configured_model or not isinstance(configured_model, str) or not configured_model.strip():
        raise GroqConfigurationError("GROQ_MODEL configuration is missing or invalid.")
    resolved_model = configured_model.strip()

    # 3. Base URL
    configured_base_url = (
        base_url if base_url is not None else os.getenv("GROQ_BASE_URL", DEFAULT_GROQ_BASE_URL)
    )
    if not configured_base_url or not isinstance(configured_base_url, str) or not configured_base_url.strip():
        resolved_base_url = DEFAULT_GROQ_BASE_URL
    else:
        resolved_base_url = configured_base_url.strip()

    # 4. Timeout
    if timeout is not None:
        try:
            resolved_timeout = float(timeout)
        except (ValueError, TypeError) as exc:
            raise GroqConfigurationError(f"Invalid timeout value: {timeout}") from exc
    else:
        try:
            resolved_timeout = float(os.getenv("GROQ_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS)))
        except (ValueError, TypeError) as exc:
            raise GroqConfigurationError("GROQ_TIMEOUT_SECONDS configuration is invalid.") from exc
    if resolved_timeout <= 0:
        raise GroqConfigurationError("Timeout must be greater than 0.")

    # 5. Deadline
    if deadline is not None:
        try:
            resolved_deadline = float(deadline)
        except (ValueError, TypeError) as exc:
            raise GroqConfigurationError(f"Invalid deadline value: {deadline}") from exc
    else:
        try:
            resolved_deadline = float(os.getenv("GROQ_DEADLINE_SECONDS", str(DEFAULT_DEADLINE_SECONDS)))
        except (ValueError, TypeError) as exc:
            raise GroqConfigurationError("GROQ_DEADLINE_SECONDS configuration is invalid.") from exc
    if resolved_deadline <= 0:
        raise GroqConfigurationError("Deadline must be greater than 0.")

    # 6. Max Retries
    if max_retries is not None:
        try:
            resolved_max_retries = int(max_retries)
        except (ValueError, TypeError) as exc:
            raise GroqConfigurationError(f"Invalid max_retries value: {max_retries}") from exc
    else:
        try:
            resolved_max_retries = int(os.getenv("GROQ_MAX_RETRIES", str(MAX_RETRIES)))
        except (ValueError, TypeError) as exc:
            raise GroqConfigurationError("GROQ_MAX_RETRIES configuration is invalid.") from exc
    if resolved_max_retries < 0:
        raise GroqConfigurationError("Max retries must be non-negative.")

    # 7. Initial Backoff
    if initial_backoff is not None:
        try:
            resolved_backoff = float(initial_backoff)
        except (ValueError, TypeError) as exc:
            raise GroqConfigurationError(f"Invalid initial_backoff value: {initial_backoff}") from exc
    else:
        try:
            resolved_backoff = float(
                os.getenv("GROQ_INITIAL_BACKOFF_SECONDS", str(INITIAL_BACKOFF_SECONDS))
            )
        except (ValueError, TypeError) as exc:
            raise GroqConfigurationError(
                "GROQ_INITIAL_BACKOFF_SECONDS configuration is invalid."
            ) from exc
    if resolved_backoff <= 0:
        raise GroqConfigurationError("Initial backoff must be greater than 0.")

    return GroqProviderConfig(
        api_key=resolved_api_key,
        model=resolved_model,
        base_url=resolved_base_url,
        timeout=resolved_timeout,
        deadline=resolved_deadline,
        max_retries=resolved_max_retries,
        initial_backoff=resolved_backoff,
    )


def validate_groq_environment() -> tuple[bool, str | None]:
    """Validates Groq environment configuration without making external network calls."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or not api_key.strip():
        return False, "ai_provider_unconfigured"

    model = os.getenv("GROQ_MODEL")
    if model is not None and not model.strip():
        return False, "invalid_ai_provider_configuration"

    return True, None


def create_groq_client(
    config: GroqProviderConfig | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float | None = None,
) -> Groq:
    """Builds a Groq client instance with centralized configuration and defaults."""
    if api_key is not None:
        resolved_api_key = api_key
    elif config and config.api_key is not None:
        resolved_api_key = config.api_key
    else:
        resolved_api_key = os.getenv("GROQ_API_KEY")

    if not resolved_api_key or not resolved_api_key.strip():
        raise GroqConfigurationError(
            "GROQ_API_KEY is not configured. Please set the GROQ_API_KEY environment variable."
        )

    resolved_base_url = (
        base_url
        or (config.base_url if config else None)
        or os.getenv("GROQ_BASE_URL", DEFAULT_GROQ_BASE_URL)
    )
    resolved_timeout = (
        timeout
        if timeout is not None
        else (config.timeout if config else DEFAULT_TIMEOUT_SECONDS)
    )

    return Groq(
        api_key=resolved_api_key.strip(),
        base_url=resolved_base_url.strip(),
        timeout=resolved_timeout,
    )


def execute_chat_completion(
    client: Groq,
    model: str,
    messages: Sequence[dict[str, Any]],
    temperature: float = 0.1,
    max_tokens: int | None = None,
    response_format: dict[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    deadline: float | None = None,
    max_retries: int = MAX_RETRIES,
    initial_backoff: float = INITIAL_BACKOFF_SECONDS,
    error_label: str = "AI provider",
) -> str:
    """Executes a chat completion request with retry, jitter, and deadline enforcement."""
    attempts = max_retries + 1
    last_exception: Exception | None = None

    for attempt in range(attempts):
        if deadline is not None and time.monotonic() >= deadline:
            raise GroqDeadlineExceededError(
                f"{error_label} request deadline exceeded before provider invocation. Service temporarily unavailable."
            )

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if response_format is not None:
            kwargs["response_format"] = response_format

        # Effective timeout bounded by remaining deadline if deadline is set
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise GroqDeadlineExceededError(
                    f"{error_label} request deadline exceeded before provider invocation. Service temporarily unavailable."
                )
            effective_timeout = min(timeout, max(0.5, remaining))
            kwargs["timeout"] = effective_timeout
        else:
            kwargs["timeout"] = timeout

        try:
            response = client.chat.completions.create(**kwargs)
            choice = response.choices[0] if response.choices else None
            content = choice.message.content if choice and choice.message else None
            if not content or not content.strip():
                raise GroqEmptyResponseError(f"Empty response returned from {error_label}.")
            return content.strip()

        except (AuthenticationError, BadRequestError) as exc:
            logger.error("Groq non-retryable client error: %s", exc)
            raise GroqClientError(
                f"{error_label} configuration or request formatting error."
            ) from exc

        except (APIConnectionError, APITimeoutError) as exc:
            last_exception = exc
            logger.warning(
                "Groq transient connection error (%s) on attempt %d/%d: %s",
                type(exc).__name__,
                attempt + 1,
                attempts,
                exc,
            )

        except RateLimitError as exc:
            last_exception = exc
            logger.warning(
                "Groq rate limit encountered on attempt %d/%d: %s",
                attempt + 1,
                attempts,
                exc,
            )

        except APIError as exc:
            status_code = getattr(exc, "status_code", None)
            if status_code and status_code in (400, 401, 403, 404, 422):
                logger.error("Groq non-retryable API error (%d): %s", status_code, exc)
                raise GroqClientError(
                    f"{error_label} configuration or request formatting error."
                ) from exc

            last_exception = exc
            logger.warning("Groq API error on attempt %d/%d: %s", attempt + 1, attempts, exc)

        except GroqEmptyResponseError:
            raise

        except Exception as exc:
            logger.error("Unexpected error during %s call: %s", error_label, exc)
            raise GroqProviderError(f"Unexpected {error_label} communication failure.") from exc

        if attempt < attempts - 1:
            backoff = initial_backoff * (2**attempt) + random.uniform(0.05, 0.2)
            if deadline is not None and (time.monotonic() + backoff >= deadline):
                raise GroqDeadlineExceededError(
                    f"{error_label} request deadline exceeded during retry backoff. Service temporarily unavailable."
                )
            time.sleep(backoff)

    # Classify error message and exception type upon retry exhaustion
    if isinstance(last_exception, APITimeoutError):
        raise GroqTimeoutError(
            f"{error_label} failed after {attempts} attempts. Provider connection timeout. Service temporarily unavailable. Last error: {last_exception}"
        )
    if isinstance(last_exception, RateLimitError):
        raise GroqRateLimitError(
            f"{error_label} failed after {attempts} attempts. Provider rate limit reached. Service temporarily unavailable. Last error: {last_exception}"
        )
    if isinstance(last_exception, APIConnectionError):
        raise GroqConnectionError(
            f"{error_label} failed after {attempts} attempts. Provider connection error (APIConnectionError). Service temporarily unavailable. Last error: {last_exception}"
        )

    raise GroqMaxRetriesExceededError(
        f"{error_label} failed after {attempts} attempts. Last error: {last_exception}"
    )

"""HTTP client for dispatching requests from HR Backend Gateway to internal AI Service.

Enforces:
- Service authentication headers (X-Gateway-Service-Token).
- Trusted caller identity and role headers (X-Caller-Employee-ID, X-Caller-Role).
- Translation of AI Service errors (401, 403, 404, 422, 429, 502, 503, 504) into
  the public HR API envelope:
  {
    "success": false,
    "message": "...",
    "errors": [...]
  }
- Safe handling of timeouts and connection errors without leaking internals.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from httpx import ASGITransport

from app.gateway.security import (
    HRApiException,
    HRCallerContext,
    get_service_auth_headers,
)

AI_SERVICE_URL = os.getenv("AI_SERVICE_URL", "").strip()


async def call_ai_service(
    method: str,
    path: str,
    caller: HRCallerContext,
    json_body: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
) -> tuple[int, Any]:
    """Dispatches a request to the internal AI Service with trusted caller context."""
    from app.main import app as ai_app

    headers = get_service_auth_headers(caller)
    headers["Accept"] = "application/json"

    # Filter out None values from params
    clean_params = {k: v for k, v in (params or {}).items() if v is not None}

    try:
        timeout = httpx.Timeout(connect=5.0, read=45.0, write=10.0, pool=5.0)
        if AI_SERVICE_URL:
            async with httpx.AsyncClient(base_url=AI_SERVICE_URL, timeout=timeout) as client:
                response = await client.request(
                    method=method,
                    url=path,
                    headers=headers,
                    json=json_body,
                    params=clean_params,
                )
        else:
            transport = ASGITransport(app=ai_app)
            async with httpx.AsyncClient(transport=transport, base_url="http://internal-ai-service", timeout=timeout) as client:
                response = await client.request(
                    method=method,
                    url=path,
                    headers=headers,
                    json=json_body,
                    params=clean_params,
                )
    except httpx.TimeoutException:
        raise HRApiException(
            status_code=504,
            message="AI service request timed out. Please try again later.",
        )
    except (httpx.ConnectError, httpx.NetworkError):
        raise HRApiException(
            status_code=502,
            message="AI service temporarily unavailable. Please try again later.",
        )

    # Process response
    try:
        resp_json = response.json()
    except (ValueError, TypeError):
        resp_json = {"detail": response.text}

    status_code = response.status_code

    if 200 <= status_code < 300:
        return status_code, resp_json

    # Map internal AI service errors to public HR error envelope
    detail = resp_json.get("detail") if isinstance(resp_json, dict) else str(resp_json)
    if isinstance(detail, list):
        # Validation error from FastAPI
        raise HRApiException(
            status_code=422,
            message="Validation error.",
            errors=detail,
        )

    safe_message = str(detail) if detail else "AI operation failed."

    if status_code == 401:
        raise HRApiException(status_code=401, message=safe_message)
    elif status_code == 403:
        raise HRApiException(status_code=403, message=safe_message)
    elif status_code == 404:
        raise HRApiException(status_code=404, message=safe_message)
    elif status_code == 422:
        raise HRApiException(status_code=422, message="Validation error.", errors=[safe_message])
    elif status_code == 429:
        raise HRApiException(status_code=429, message="Rate limit exceeded. Please retry later.")
    elif status_code == 502:
        raise HRApiException(status_code=502, message="AI provider is temporarily unavailable. Please try again later.")
    elif status_code == 503:
        raise HRApiException(status_code=503, message="AI service dependency is unavailable.")
    elif status_code == 504:
        raise HRApiException(status_code=504, message="AI provider timed out.")
    else:
        raise HRApiException(status_code=status_code, message=safe_message)

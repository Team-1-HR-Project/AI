"""JWT encoding and decoding utility for HR Backend Gateway authentication.

Uses standard Python standard library (hmac, hashlib, json, base64) to implement
RFC 7519 compliant HMAC-SHA256 (HS256) JSON Web Tokens.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from typing import Any

DEFAULT_JWT_SECRET = os.getenv("JWT_SECRET_KEY", os.getenv("JWT_SECRET", "smart-hr-gateway-jwt-super-secret-key-2026"))


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(data: str) -> bytes:
    padding = "=" * ((4 - len(data) % 4) % 4)
    return base64.urlsafe_b64decode(data + padding)


def create_jwt_token(
    payload: dict[str, Any],
    secret: str = DEFAULT_JWT_SECRET,
    expires_in_seconds: int = 86400,
) -> str:
    """Creates a signed HS256 JWT string."""
    header = {"typ": "JWT", "alg": "HS256"}
    now = int(time.time())
    token_payload = dict(payload)
    if "iat" not in token_payload:
        token_payload["iat"] = now
    if "exp" not in token_payload:
        token_payload["exp"] = now + expires_in_seconds

    header_bytes = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    payload_bytes = json.dumps(token_payload, separators=(",", ":"), sort_keys=True).encode("utf-8")

    signing_input = f"{_b64url_encode(header_bytes)}.{_b64url_encode(payload_bytes)}".encode("ascii")
    signature = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    return f"{signing_input.decode('ascii')}.{_b64url_encode(signature)}"


def decode_jwt_token(token: str, secret: str = DEFAULT_JWT_SECRET, verify_exp: bool = True) -> dict[str, Any]:
    """Decodes and validates a signed HS256 JWT string.

    Raises ValueError on invalid token, invalid signature, or expired token.
    """
    parts = token.strip().split(".")
    if len(parts) != 3:
        raise ValueError("Invalid JWT token format: expected 3 parts")

    header_b64, payload_b64, signature_b64 = parts
    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")

    try:
        expected_sig = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
        provided_sig = _b64url_decode(signature_b64)
        if not hmac.compare_digest(expected_sig, provided_sig):
            raise ValueError("Invalid JWT signature")
    except Exception as exc:
        raise ValueError(f"Signature verification failed: {exc}") from exc

    try:
        payload_bytes = _b64url_decode(payload_b64)
        payload = json.loads(payload_bytes.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"Failed to decode JWT payload: {exc}") from exc

    if verify_exp and "exp" in payload:
        exp = payload["exp"]
        if isinstance(exp, (int, float)) and time.time() > exp:
            raise ValueError("JWT token has expired")

    return payload

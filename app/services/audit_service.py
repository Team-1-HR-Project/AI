"""Reusable AI Audit Service and Context Manager for compliance and governance.

Captures:
- event_id
- timestamp (UTC)
- actor_employee_id & actor_role (from caller context)
- authorized_scope (employee, department, session)
- AI feature name & endpoint
- provider & model (dynamically retrieved from active service instance/config)
- outcome (success, insufficient_data, unauthorized, provider_error, etc.)
- reference_id (opaque UUID when errors occur)

Privacy Guarantees:
- Zero prompt storage
- Zero AI response storage
- Zero secrets / API keys / passwords
- Minimal, sanitized actor and scope references
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.security import CallerContext
from app.models import AIAuditEvent, utc_now

logger = logging.getLogger(__name__)

DEFAULT_PROVIDER = "groq"
DEFAULT_MODEL = "openai/gpt-oss-120b"


class AIAuditOutcome:
    SUCCESS = "success"
    INSUFFICIENT_DATA = "insufficient_data"
    UNAUTHORIZED = "unauthorized"
    VALIDATION_ERROR = "validation_error"
    PROVIDER_ERROR = "provider_error"
    DEPENDENCY_ERROR = "dependency_error"
    GROUNDING_ERROR = "grounding_error"
    SAFETY_ERROR = "safety_error"


class AIAuditContext:
    """Manages the audit lifecycle for an AI endpoint invocation."""

    def __init__(
        self,
        db: Session | None,
        feature: str,
        endpoint: str,
        caller: CallerContext | None = None,
        ai_service: Any = None,
        scope_employee_id: str | None = None,
        scope_department: str | None = None,
        scope_session_id: str | None = None,
    ):
        self.db = db
        self.feature = feature
        self.endpoint = endpoint
        self.caller = caller
        self.ai_service = ai_service

        self.scope_employee_id = scope_employee_id
        self.scope_department = scope_department
        self.scope_session_id = scope_session_id

        self.outcome = AIAuditOutcome.SUCCESS
        self.reference_id: str | None = None
        self._flushed = False

    def set_scope(
        self,
        employee_id: str | None = None,
        department: str | None = None,
        session_id: str | None = None,
    ) -> None:
        """Updates authorized scope parameters during request lifecycle."""
        if employee_id is not None:
            self.scope_employee_id = employee_id
        if department is not None:
            self.scope_department = department
        if session_id is not None:
            self.scope_session_id = session_id

    def record_outcome(
        self,
        outcome: str,
        reference_id: str | None = None,
    ) -> None:
        """Explicitly sets the outcome and optional failure reference ID."""
        self.outcome = outcome
        if reference_id is not None:
            self.reference_id = reference_id

    def record_response(self, response: Any) -> None:
        """Derives outcome from the response object status."""
        status_val = getattr(response, "status", None)
        if isinstance(response, dict):
            status_val = response.get("status")

        if status_val in ("insufficient_data", "unsupported"):
            self.outcome = AIAuditOutcome.INSUFFICIENT_DATA
        elif status_val == "success":
            self.outcome = AIAuditOutcome.SUCCESS
        elif status_val:
            self.outcome = str(status_val)

    def flush(self) -> AIAuditEvent | None:
        """Persists the audit event safely to the database without leaking secrets or breaking the caller."""
        if self._flushed:
            return None
        self._flushed = True

        if self.db is None:
            return None

        # 1. Resolve actor identity strictly from caller context
        actor_id = self.caller.employee_id if self.caller else None
        actor_role = self.caller.role if self.caller else None

        # 2. Resolve provider and model from service or environment
        provider = DEFAULT_PROVIDER
        model = (
            getattr(self.ai_service, "model", None)
            or os.getenv("GROQ_MODEL", DEFAULT_MODEL)
        )

        event_uuid = str(uuid.uuid4())
        event = AIAuditEvent(
            id=event_uuid,
            event_id=event_uuid,
            timestamp=utc_now(),
            actor_employee_id=actor_id,
            actor_role=actor_role,
            scope_employee_id=self.scope_employee_id,
            scope_department=self.scope_department,
            scope_session_id=self.scope_session_id,
            feature=self.feature,
            endpoint=self.endpoint,
            provider=provider,
            model=model,
            outcome=self.outcome,
            reference_id=self.reference_id,
        )

        try:
            self.db.add(event)
            self.db.commit()
            return event
        except (SQLAlchemyError, OSError, RuntimeError) as exc:
            logger.warning("Audit persistence non-fatal failure: %s", exc)
            try:
                self.db.rollback()
            except (SQLAlchemyError, OSError, RuntimeError) as rb_exc:
                logger.debug("Audit rollback error: %s", rb_exc)
            return None


class AIAuditService:
    """Service providing helper methods for AI audit operations."""

    @staticmethod
    def create_context(
        db: Session | None,
        feature: str,
        endpoint: str,
        caller: CallerContext | None = None,
        ai_service: Any = None,
        scope_employee_id: str | None = None,
        scope_department: str | None = None,
        scope_session_id: str | None = None,
    ) -> AIAuditContext:
        """Creates an audit context tracker for an AI request."""
        return AIAuditContext(
            db=db,
            feature=feature,
            endpoint=endpoint,
            caller=caller,
            ai_service=ai_service,
            scope_employee_id=scope_employee_id,
            scope_department=scope_department,
            scope_session_id=scope_session_id,
        )

    @staticmethod
    def record_event(
        db: Session | None,
        *,
        feature: str,
        endpoint: str,
        caller: CallerContext | None = None,
        actor_employee_id: str | None = None,
        actor_role: str | None = None,
        scope_employee_id: str | None = None,
        scope_department: str | None = None,
        scope_session_id: str | None = None,
        provider: str = DEFAULT_PROVIDER,
        model: str | None = None,
        outcome: str = AIAuditOutcome.SUCCESS,
        reference_id: str | None = None,
        ai_service: Any = None,
    ) -> AIAuditEvent | None:
        """Directly records an AI audit event without context manager wrapping."""
        if db is None:
            return None

        actor_id = caller.employee_id if caller else actor_employee_id
        role = caller.role if caller else actor_role

        resolved_model = (
            model
            or getattr(ai_service, "model", None)
            or os.getenv("GROQ_MODEL", DEFAULT_MODEL)
        )

        event_uuid = str(uuid.uuid4())
        event = AIAuditEvent(
            id=event_uuid,
            event_id=event_uuid,
            timestamp=utc_now(),
            actor_employee_id=actor_id,
            actor_role=role,
            scope_employee_id=scope_employee_id,
            scope_department=scope_department,
            scope_session_id=scope_session_id,
            feature=feature,
            endpoint=endpoint,
            provider=provider,
            model=resolved_model,
            outcome=outcome,
            reference_id=reference_id,
        )

        try:
            db.add(event)
            db.commit()
            return event
        except (SQLAlchemyError, OSError, RuntimeError) as exc:
            logger.warning("Audit persistence non-fatal failure: %s", exc)
            try:
                db.rollback()
            except (SQLAlchemyError, OSError, RuntimeError) as rb_exc:
                logger.debug("Audit rollback error: %s", rb_exc)
            return None

"""Durable AI insight snapshots, history retrieval, regeneration, and feedback capture service."""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from collections.abc import Callable
from typing import Any

from fastapi import HTTPException, status
from sqlalchemy import func
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.security import (
    CallerContext,
    CallerRole,
    authorize_department_scope,
    authorize_employee_scope,
)
from app.models import AIFeedback, AIInsightSnapshot, utc_now

logger = logging.getLogger(__name__)


def _canonical_json(value: Any) -> str:
    def normalize(item: Any) -> Any:
        if isinstance(item, dict):
            return {str(key): normalize(val) for key, val in item.items()}
        if isinstance(item, (list, tuple, set)):
            return [normalize(value) for value in item]
        if isinstance(item, (str, int, float, bool)) or item is None:
            return item
        return str(item)

    return json.dumps(normalize(value), sort_keys=True, separators=(",", ":"))


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def snapshot_metadata(
    *,
    feature: str,
    context: Any,
    request_payload: dict[str, Any] | None,
    ai_service: Any,
) -> dict[str, Any]:
    """Build deterministic reproducibility metadata without storing raw prompts."""
    source_payload = context.get("approved_sources", context) if isinstance(context, dict) else context
    source_hash = _sha256(source_payload)
    context_hash = _sha256(context)
    prompt_hash = _sha256({"feature": feature, "request": request_payload or {}, "context": context})
    return {
        "source_version": f"sha256:{source_hash[:16]}",
        "source_hash": source_hash,
        "context_hash": context_hash,
        "prompt_hash": prompt_hash,
        "provider": "groq",
        "model": str(getattr(ai_service, "model", "")) or None,
    }


def serialize_content(content: Any) -> str:
    """Serializes Pydantic models or dicts to JSON string."""
    if hasattr(content, "model_dump_json"):
        return content.model_dump_json()
    if hasattr(content, "model_dump"):
        return json.dumps(content.model_dump(), default=str)
    if isinstance(content, str):
        return content
    return json.dumps(content, default=str)


def deserialize_content(content_str: str) -> Any:
    """Safely parses JSON string back to dict or list; returns raw string on parse error."""
    try:
        return json.loads(content_str)
    except (json.JSONDecodeError, TypeError):
        return content_str


def save_insight_snapshot(
    db: Session,
    feature: str,
    content: Any,
    scope_employee_id: str | None = None,
    scope_department: str | None = None,
    period: str | None = None,
    actor_employee_id: str | None = None,
    actor_role: str | None = None,
    request_payload: dict[str, Any] | None = None,
    context: Any = None,
    ai_service: Any = None,
    generation_id: str | None = None,
    previous_snapshot_id: str | None = None,
    regeneration_reason: str | None = None,
    source_changed: bool | None = None,
    regenerated_at: Any = None,
    metadata: dict[str, Any] | None = None,
) -> AIInsightSnapshot:
    """Durable persistence for generated AI insights.

    Automatically calculates incremental versioning per (feature, scope, period).
    Assigns a unique generation_id and creation timestamp.
    """
    clean_feature = feature.strip().lower()
    clean_emp_id = scope_employee_id.strip() if scope_employee_id else None
    clean_dept = scope_department.strip() if scope_department else None
    clean_period = period.strip() if period else None

    if generation_id:
        existing_generation = (
            db.query(AIInsightSnapshot)
            .filter(AIInsightSnapshot.generation_id == generation_id)
            .first()
        )
        if existing_generation:
            return existing_generation

    # Calculate next version for this specific scope and period
    version_query = db.query(func.max(AIInsightSnapshot.version)).filter(
        AIInsightSnapshot.feature == clean_feature
    )
    if clean_emp_id:
        version_query = version_query.filter(AIInsightSnapshot.scope_employee_id == clean_emp_id)
    else:
        version_query = version_query.filter(AIInsightSnapshot.scope_employee_id.is_(None))

    if clean_dept:
        version_query = version_query.filter(AIInsightSnapshot.scope_department == clean_dept)
    else:
        version_query = version_query.filter(AIInsightSnapshot.scope_department.is_(None))

    if clean_period:
        version_query = version_query.filter(AIInsightSnapshot.period == clean_period)
    else:
        version_query = version_query.filter(AIInsightSnapshot.period.is_(None))

    current_max_version = version_query.scalar() or 0
    next_version = current_max_version + 1

    metadata = metadata or snapshot_metadata(
        feature=clean_feature,
        context=context if context is not None else {"request": request_payload or {}},
        request_payload=request_payload,
        ai_service=ai_service,
    )
    snapshot = AIInsightSnapshot(
        id=str(uuid.uuid4()),
        generation_id=generation_id or str(uuid.uuid4()),
        version=next_version,
        feature=clean_feature,
        source_version=metadata.get("source_version"),
        source_hash=metadata.get("source_hash"),
        context_hash=metadata.get("context_hash"),
        prompt_hash=metadata.get("prompt_hash"),
        provider=metadata.get("provider"),
        model=metadata.get("model"),
        scope_employee_id=clean_emp_id,
        scope_department=clean_dept,
        period=clean_period,
        content=serialize_content(content),
        request_payload=serialize_content(request_payload) if request_payload is not None else None,
        actor_employee_id=actor_employee_id,
        actor_role=actor_role,
        previous_snapshot_id=previous_snapshot_id,
        regenerated_at=regenerated_at,
        regeneration_reason=regeneration_reason,
        source_changed=source_changed,
        created_at=utc_now(),
    )
    db.add(snapshot)
    db.commit()
    db.refresh(snapshot)
    logger.info(
        "Saved AI insight snapshot id=%s (gen_id=%s, feature=%s, version=%d)",
        snapshot.id,
        snapshot.generation_id,
        snapshot.feature,
        snapshot.version,
    )
    return snapshot


def persist_insight_snapshot(
    *,
    db: Session,
    feature: str,
    content: Any,
    context: Any,
    request_payload: dict[str, Any],
    scope_employee_id: str | None,
    scope_department: str | None,
    period: str | None,
    caller: CallerContext,
    ai_service: Any,
    generation_id: str | None = None,
    previous_snapshot_id: str | None = None,
    regeneration_reason: str | None = None,
    source_changed: bool | None = None,
    regenerated_at: Any = None,
) -> AIInsightSnapshot | None:
    """Persist one generated result with deterministic metadata."""
    try:
        return save_insight_snapshot(
            db=db,
            feature=feature,
            content=content,
            scope_employee_id=scope_employee_id,
            scope_department=scope_department,
            period=period,
            actor_employee_id=caller.employee_id if caller else None,
            actor_role=caller.role if caller else None,
            request_payload=request_payload,
            context=context,
            ai_service=ai_service,
            generation_id=generation_id,
            previous_snapshot_id=previous_snapshot_id,
            regeneration_reason=regeneration_reason,
            source_changed=source_changed,
            regenerated_at=regenerated_at,
        )
    except (SQLAlchemyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        # Snapshot persistence must not change the response contract of an AI route.
        logger.warning("AI insight snapshot persistence failed: %s", exc)
        db.rollback()
        return None


def verify_snapshot_authorization(
    caller: CallerContext,
    snapshot: AIInsightSnapshot,
    db: Session,
) -> None:
    """Enforces caller authorization and scope isolation on an existing snapshot."""
    if snapshot.scope_employee_id:
        authorize_employee_scope(
            caller=caller,
            target_employee_id=snapshot.scope_employee_id,
            db=db,
            is_manager_only=False,
        )
    elif snapshot.scope_department:
        authorize_department_scope(
            caller=caller,
            requested_department=snapshot.scope_department,
        )
    else:
        # A snapshot without an explicit scope must never be enumerable by a
        # normal employee/manager.  Only HR administrators may inspect such a
        # legacy/system record until it is repaired or re-scoped.
        if caller.role != CallerRole.HR_ADMIN.value:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden: snapshot has no authorized employee or department scope.",
            )


def get_snapshot_by_id(
    db: Session,
    caller: CallerContext,
    snapshot_id: str,
) -> AIInsightSnapshot:
    """Retrieves a single snapshot by ID, enforcing caller authorization."""
    snapshot = db.query(AIInsightSnapshot).filter(AIInsightSnapshot.id == snapshot_id.strip()).first()
    if not snapshot:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"AI insight snapshot '{snapshot_id}' not found.",
        )
    verify_snapshot_authorization(caller=caller, snapshot=snapshot, db=db)
    return snapshot


def get_insight_history(
    db: Session,
    caller: CallerContext,
    feature: str,
    employee_id: str | None = None,
    department: str | None = None,
    period: str | None = None,
    page: int = 1,
    page_size: int = 50,
) -> list[AIInsightSnapshot]:
    """Retrieves version history for previously generated insights with strict scope isolation."""
    clean_feature = feature.strip().lower()

    # Scope authorization checks
    target_emp_id: str | None = None
    target_dept: str | None = None

    if employee_id:
        target_emp_id = authorize_employee_scope(
            caller=caller,
            target_employee_id=employee_id,
            db=db,
            is_manager_only=False,
        )
    elif caller.role == CallerRole.EMPLOYEE.value:
        # Default employee role to self scope
        target_emp_id = caller.employee_id

    if department:
        target_dept = authorize_department_scope(
            caller=caller,
            requested_department=department,
        )
    elif caller.role == CallerRole.MANAGER.value and not target_emp_id:
        target_dept = caller.department

    query = db.query(AIInsightSnapshot).filter(AIInsightSnapshot.feature == clean_feature)

    if target_emp_id:
        query = query.filter(AIInsightSnapshot.scope_employee_id == target_emp_id)
    if target_dept:
        query = query.filter(AIInsightSnapshot.scope_department == target_dept)
    if period:
        query = query.filter(AIInsightSnapshot.period == period.strip())

    # Bound result size to avoid unbounded history reads.  Fetch one extra row
    # so callers can expose a stable has_more indicator without leaking data.
    safe_page = max(1, page)
    safe_page_size = min(max(1, page_size), 100)
    snapshots = (
        query.order_by(AIInsightSnapshot.version.desc(), AIInsightSnapshot.created_at.desc())
        .offset((safe_page - 1) * safe_page_size)
        .limit(safe_page_size + 1)
        .all()
    )

    # Double check scope authorization on all retrieved items
    authorized_snapshots: list[AIInsightSnapshot] = []
    for s in snapshots:
        try:
            verify_snapshot_authorization(caller=caller, snapshot=s, db=db)
            authorized_snapshots.append(s)
        except HTTPException:
            continue

    return authorized_snapshots


def record_ai_feedback(
    db: Session,
    caller: CallerContext,
    snapshot_id: str,
    is_helpful: bool,
    feedback_text: str | None = None,
) -> AIFeedback:
    """Records user feedback (helpful / not helpful + optional text) linked to an insight snapshot."""
    snapshot = get_snapshot_by_id(db=db, caller=caller, snapshot_id=snapshot_id)

    feedback = AIFeedback(
        id=str(uuid.uuid4()),
        snapshot_id=snapshot.id,
        actor_employee_id=caller.employee_id,
        actor_role=caller.role,
        is_helpful=is_helpful,
        feedback_text=feedback_text.strip() if feedback_text else None,
        created_at=utc_now(),
    )
    db.add(feedback)
    db.commit()
    db.refresh(feedback)
    logger.info(
        "Recorded AI feedback id=%s on snapshot_id=%s (helpful=%s, caller=%s)",
        feedback.id,
        snapshot.id,
        feedback.is_helpful,
        caller.employee_id,
    )
    return feedback


def regenerate_insight_snapshot(
    db: Session,
    caller: CallerContext,
    previous_snapshot_id: str,
    generator: Callable[[AIInsightSnapshot], tuple[Any, Any, Any, dict[str, Any], Any]],
    regeneration_reason: str | None = None,
) -> AIInsightSnapshot:
    """Regenerates through the existing feature pipeline without accepting generated content."""
    previous_snapshot = get_snapshot_by_id(db=db, caller=caller, snapshot_id=previous_snapshot_id)
    new_content, current_context, ai_service, request_payload, current_period = generator(previous_snapshot)
    metadata = snapshot_metadata(
        feature=previous_snapshot.feature,
        context=current_context,
        request_payload=request_payload,
        ai_service=ai_service,
    )
    new_snapshot = save_insight_snapshot(
        db=db,
        feature=previous_snapshot.feature,
        content=new_content,
        scope_employee_id=previous_snapshot.scope_employee_id,
        scope_department=previous_snapshot.scope_department,
        period=current_period or previous_snapshot.period,
        actor_employee_id=caller.employee_id,
        actor_role=caller.role,
        request_payload=request_payload,
        context=current_context,
        ai_service=ai_service,
        previous_snapshot_id=previous_snapshot.id,
        regeneration_reason=regeneration_reason or "manual_regeneration",
        source_changed=(
            previous_snapshot.source_hash is not None
            and metadata.get("source_hash") != previous_snapshot.source_hash
        ),
        regenerated_at=utc_now(),
        metadata=metadata,
    )
    logger.info(
        "Regenerated insight snapshot: old_id=%s (v%d) -> new_id=%s (v%d)",
        previous_snapshot.id,
        previous_snapshot.version,
        new_snapshot.id,
        new_snapshot.version,
    )
    return new_snapshot

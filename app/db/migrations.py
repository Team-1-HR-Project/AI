"""Database migration utility for adding `is_approved` column to HR tables.

Ensures idempotent, safe addition of `is_approved` column across:
- performance_records
- goals
- skills
- task_outcomes
- evaluation_themes

Backfill Policy:
- Legacy existing records default to `is_approved = False` (0) so they do NOT
  silently become approved.
- Newly created records default to `False` (0) unless explicitly approved by workflow.
"""

import logging
from typing import Any

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

HR_TABLES_REQUIRING_IS_APPROVED = [
    "performance_records",
    "goals",
    "skills",
    "task_outcomes",
    "evaluation_themes",
]


def check_is_approved_columns(bind: Engine) -> dict[str, bool]:
    """Inspects the given engine/bind and returns whether `is_approved` column exists per table."""
    inspector = inspect(bind)
    existing_tables: set[str] = set(inspector.get_table_names())
    status: dict[str, bool] = {}

    for table_name in HR_TABLES_REQUIRING_IS_APPROVED:
        if table_name not in existing_tables:
            status[table_name] = False
            continue
        columns = [col["name"] for col in inspector.get_columns(table_name)]
        status[table_name] = "is_approved" in columns

    return status


def migrate_is_approved_columns(
    bind: Engine,
    default_for_legacy: bool = False,
) -> dict[str, Any]:
    """Safely and idempotently adds `is_approved` column to the 5 HR tables if missing.

    Parameters
    ----------
    bind : Engine
        SQLAlchemy engine for MySQL or SQLite.
    default_for_legacy : bool, default False
        Explicit backfill value for existing records.
        Per security policy: legacy rows must NOT silently become approved,
        so default_for_legacy defaults to False (0).

    Returns
    -------
    dict[str, Any]
        Migration report detailing added columns, existing columns, and backfill counts.
    """
    inspector = inspect(bind)
    existing_tables: set[str] = set(inspector.get_table_names())
    dialect_name: str = bind.dialect.name.lower()

    report: dict[str, Any] = {
        "dialect": dialect_name,
        "tables_checked": [],
        "columns_added": [],
        "already_present": [],
        "tables_missing": [],
        "backfill_default": default_for_legacy,
    }

    legacy_int_val = 1 if default_for_legacy else 0

    with bind.begin() as conn:
        for table_name in HR_TABLES_REQUIRING_IS_APPROVED:
            report["tables_checked"].append(table_name)

            if table_name not in existing_tables:
                report["tables_missing"].append(table_name)
                logger.info("Table '%s' does not exist yet. Skipping column migration.", table_name)
                continue

            columns = [col["name"] for col in inspector.get_columns(table_name)]
            if "is_approved" in columns:
                report["already_present"].append(table_name)
                logger.info("Column 'is_approved' already present in table '%s'.", table_name)
                continue

            # Column is missing: add it safely without destroying data
            logger.info("Adding 'is_approved' column to table '%s' (backfill: %s)...", table_name, default_for_legacy)
            if dialect_name in ("mysql", "mariadb"):
                alter_sql = text(
                    f"ALTER TABLE `{table_name}` "
                    f"ADD COLUMN `is_approved` TINYINT(1) NOT NULL DEFAULT {legacy_int_val}"
                )
            else:
                # SQLite / standard SQL
                alter_sql = text(
                    f"ALTER TABLE {table_name} "
                    f"ADD COLUMN is_approved BOOLEAN NOT NULL DEFAULT {legacy_int_val}"
                )

            conn.execute(alter_sql)
            report["columns_added"].append(table_name)

    return report


def migrate_chat_message_embedding_column(bind: Engine) -> dict[str, Any]:
    """Safely and idempotently adds `embedding` column to `chat_messages` table if missing.

    Parameters
    ----------
    bind : Engine
        SQLAlchemy engine for MySQL or SQLite.

    Returns
    -------
    dict[str, Any]
        Migration report detailing whether column was added or was already present.
    """
    inspector = inspect(bind)
    existing_tables: set[str] = set(inspector.get_table_names())
    dialect_name: str = bind.dialect.name.lower()

    report: dict[str, Any] = {
        "table": "chat_messages",
        "dialect": dialect_name,
        "column_added": False,
        "already_present": False,
        "table_missing": False,
    }

    if "chat_messages" not in existing_tables:
        report["table_missing"] = True
        logger.info("Table 'chat_messages' does not exist yet. Skipping column migration.")
        return report

    columns = [col["name"] for col in inspector.get_columns("chat_messages")]
    if "embedding" in columns:
        report["already_present"] = True
        logger.info("Column 'embedding' already present in table 'chat_messages'.")
        return report

    logger.info("Adding 'embedding' column to table 'chat_messages'...")
    with bind.begin() as conn:
        if dialect_name in ("mysql", "mariadb"):
            alter_sql = text("ALTER TABLE `chat_messages` ADD COLUMN `embedding` LONGTEXT NULL")
        else:
            # SQLite / standard SQL
            alter_sql = text("ALTER TABLE chat_messages ADD COLUMN embedding TEXT NULL")

        conn.execute(alter_sql)
        report["column_added"] = True

    return report


def migrate_ai_audit_events_table(bind: Engine) -> dict[str, Any]:
    """Safely and idempotently ensures the `ai_audit_events` table exists.

    Parameters
    ----------
    bind : Engine
        SQLAlchemy engine for MySQL or SQLite.

    Returns
    -------
    dict[str, Any]
        Migration report detailing whether table was created or was already present.
    """
    inspector = inspect(bind)
    existing_tables: set[str] = set(inspector.get_table_names())
    report: dict[str, Any] = {
        "table": "ai_audit_events",
        "table_created": False,
        "already_present": False,
    }

    if "ai_audit_events" in existing_tables:
        report["already_present"] = True
        return report

    logger.info("Creating 'ai_audit_events' table...")
    from app.models import AIAuditEvent

    AIAuditEvent.__table__.create(bind=bind, checkfirst=True)
    report["table_created"] = True
    return report


def migrate_ai_snapshots_tables(bind: Engine) -> dict[str, Any]:
    """Safely and idempotently ensures `ai_insight_snapshots` and `ai_feedbacks` tables exist.

    Parameters
    ----------
    bind : Engine
        SQLAlchemy engine for MySQL or SQLite.

    Returns
    -------
    dict[str, Any]
        Migration report detailing whether tables were created or already present.
    """
    inspector = inspect(bind)
    existing_tables: set[str] = set(inspector.get_table_names())
    report: dict[str, Any] = {
        "snapshots_table_created": False,
        "snapshots_already_present": False,
        "feedbacks_table_created": False,
        "feedbacks_already_present": False,
    }

    from app.models import AIFeedback, AIInsightSnapshot

    if "ai_insight_snapshots" in existing_tables:
        report["snapshots_already_present"] = True
        _migrate_ai_snapshot_columns(bind)
    else:
        logger.info("Creating 'ai_insight_snapshots' table...")
        AIInsightSnapshot.__table__.create(bind=bind, checkfirst=True)
        report["snapshots_table_created"] = True

    if "ai_feedbacks" in existing_tables:
        report["feedbacks_already_present"] = True
    else:
        logger.info("Creating 'ai_feedbacks' table...")
        AIFeedback.__table__.create(bind=bind, checkfirst=True)
        report["feedbacks_table_created"] = True

    return report


def _migrate_ai_snapshot_columns(bind: Engine) -> None:
    """Adds reproducibility and lineage columns to pre-existing snapshot tables."""
    inspector = inspect(bind)
    columns = {column["name"] for column in inspector.get_columns("ai_insight_snapshots")}
    missing = {
        "source_version": "VARCHAR(100)",
        "source_hash": "VARCHAR(64)",
        "context_hash": "VARCHAR(64)",
        "prompt_hash": "VARCHAR(64)",
        "provider": "VARCHAR(50)",
        "model": "VARCHAR(150)",
        "request_payload": "TEXT",
        "previous_snapshot_id": "VARCHAR(36)",
        "regenerated_at": "DATETIME",
        "regeneration_reason": "TEXT",
        "source_changed": "BOOLEAN",
    }
    with bind.begin() as conn:
        for name, sql_type in missing.items():
            if name in columns:
                continue
            conn.execute(text(f"ALTER TABLE ai_insight_snapshots ADD COLUMN {name} {sql_type} NULL"))

"""Database session and connection management with explicit credential enforcement."""

from __future__ import annotations

import os
from urllib.parse import quote_plus

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv()


class DatabaseConfigurationError(RuntimeError):
    """Raised when required database credentials or configuration are missing or invalid."""


class _UnconfiguredEngine:
    """Placeholder engine that raises an OperationalError when database credentials are not configured."""

    def __init__(self, error_message: str):
        self._error_message = error_message

    def connect(self, *args, **kwargs):
        raise OperationalError(
            self._error_message,
            params={},
            orig=Exception(self._error_message),
        )

    def execute(self, *args, **kwargs):
        raise OperationalError(
            self._error_message,
            params={},
            orig=Exception(self._error_message),
        )

    def _run_ddl_visitor(self, *args, **kwargs):
        raise OperationalError(
            self._error_message,
            params={},
            orig=Exception(self._error_message),
        )

    def __repr__(self) -> str:
        return "<UnconfiguredEngine: credentials missing or invalid>"


def get_database_url() -> str:
    """Builds and validates the database connection URL from environment variables.

    Rules:
    - Never falls back to 'root' user or empty/blank password.
    - Explicit database credentials are required.
    - SQLite URLs (e.g. in-memory or test files) are permitted without credentials.
    - For network databases (MySQL, PostgreSQL, etc.), both username and a non-empty
      password are strictly required.
    """
    raw_url = os.getenv("DATABASE_URL")
    if raw_url and raw_url.strip():
        clean_url = raw_url.strip()
        if clean_url.startswith("sqlite"):
            return clean_url
        try:
            parsed = make_url(clean_url)
        except (SQLAlchemyError, ValueError):
            raise DatabaseConfigurationError(
                "DATABASE_URL is malformed and could not be parsed."
            ) from None

        if not parsed.username or not parsed.username.strip():
            raise DatabaseConfigurationError(
                "DATABASE_URL is missing an explicit database username."
            )
        if not parsed.password or not parsed.password.strip():
            raise DatabaseConfigurationError(
                "DATABASE_URL has a missing or empty database password. "
                "Blank passwords and default root fallbacks are prohibited."
            )
        return clean_url

    # When DATABASE_URL is not set, require explicit DB_USER and DB_PASSWORD
    db_user = os.getenv("DB_USER")
    if not db_user or not db_user.strip():
        raise DatabaseConfigurationError(
            "Missing explicit database username. Configure DB_USER or DATABASE_URL."
        )

    db_password = os.getenv("DB_PASSWORD")
    if not db_password or not db_password.strip():
        raise DatabaseConfigurationError(
            "Missing or empty explicit database password. Configure DB_PASSWORD or DATABASE_URL. "
            "Blank passwords and default root fallbacks are prohibited."
        )

    db_host = os.getenv("DB_HOST", "127.0.0.1")
    db_port = os.getenv("DB_PORT", "3306")
    db_name = os.getenv("DB_NAME", "hr_system")

    encoded_user = quote_plus(db_user.strip())
    encoded_pass = quote_plus(db_password.strip())
    return f"mysql+pymysql://{encoded_user}:{encoded_pass}@{db_host}:{db_port}/{db_name}?charset=utf8mb4"


def init_engine(url: str | None = None):
    """Initializes or reconfigures the database engine and SessionLocal.

    If url is not provided, resolves and validates from environment variables via get_database_url().
    """
    global DATABASE_URL, DB_CONFIG_ERROR, engine, SessionLocal
    try:
        if url is not None:
            clean_url = url.strip()
            if not clean_url.startswith("sqlite"):
                parsed = make_url(clean_url)
                if not parsed.username or not parsed.username.strip():
                    raise DatabaseConfigurationError(
                        "DATABASE_URL is missing an explicit database username."
                    )
                if not parsed.password or not parsed.password.strip():
                    raise DatabaseConfigurationError(
                        "DATABASE_URL has a missing or empty database password."
                    )
            DATABASE_URL = clean_url
        else:
            DATABASE_URL = get_database_url()

        if DATABASE_URL.startswith("sqlite"):
            engine = create_engine(
                DATABASE_URL,
                connect_args={"check_same_thread": False},
                echo=False,
            )
        else:
            engine = create_engine(
                DATABASE_URL,
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=False,
            )
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        DB_CONFIG_ERROR = None
        return engine, SessionLocal
    except DatabaseConfigurationError as exc:
        DATABASE_URL = None
        DB_CONFIG_ERROR = str(exc)
        engine = _UnconfiguredEngine(DB_CONFIG_ERROR)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        return engine, SessionLocal


# Global state initialization
DATABASE_URL: str | None = None
DB_CONFIG_ERROR: str | None = None
engine = None
SessionLocal = None

init_engine()

Base = declarative_base()


def get_db():
    if DB_CONFIG_ERROR is not None:
        raise DatabaseConfigurationError(DB_CONFIG_ERROR)
    if SessionLocal is None:
        raise DatabaseConfigurationError("Database session factory is not initialized.")
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

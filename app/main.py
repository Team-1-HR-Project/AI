import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Ensure project root is in sys.path when executed directly
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv()

import logging

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

import app.models
from app.api.attention_signal import router as attention_signal_router
from app.api.career_coach import router as career_coach_router
from app.api.evaluation_draft import router as evaluation_draft_router
from app.api.health import router as health_router
from app.api.insight_snapshots import router as insight_snapshots_router
from app.api.performance_insight import router as performance_insight_router
from app.api.policy_assistant import router as policy_assistant_router
from app.api.skill_gap import router as skill_gap_router
from app.api.team_insight import router as team_insight_router
from app.db.migrations import (
    migrate_ai_audit_events_table,
    migrate_ai_snapshots_tables,
    migrate_chat_message_embedding_column,
    migrate_is_approved_columns,
)
from app.db.session import DB_CONFIG_ERROR, Base, engine, get_db
from app.gateway import HRApiException, hr_gateway_router
from app.services.shared_hr_data import is_shared_hr_schema

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # When dependency overrides are active (e.g. in test suites), allow caller to manage test database
    if get_db in app.dependency_overrides:
        app.state.db_initialized = True
        app.state.db_initialization_error = None
        yield
        return

    if DB_CONFIG_ERROR is not None:
        logger.warning("Database configuration error at startup: %s", DB_CONFIG_ERROR)
        app.state.db_initialization_error = "database_configuration_invalid"
        app.state.db_initialized = False
        yield
        return

    app.state.db_initialized = False
    app.state.db_initialization_error = None
    # Initialize database schema and ensure idempotent column migrations at startup
    try:
        if is_shared_hr_schema(engine):
            logger.info("Detected shared Laravel HR schema; skipping HR-owned table creation and migrations.")
        else:
            Base.metadata.create_all(bind=engine)
            migrate_is_approved_columns(bind=engine, default_for_legacy=False)
            migrate_chat_message_embedding_column(bind=engine)
        migrate_ai_audit_events_table(bind=engine)
        migrate_ai_snapshots_tables(bind=engine)
        app.state.db_initialized = True
    except (SQLAlchemyError, OSError) as exc:
        logger.warning("Database schema initialization warning: %s", exc)
        app.state.db_initialization_error = str(exc)
        app.state.db_initialized = False
    yield


app = FastAPI(
    title="Smart HR Management System - AI Service",
    version="0.1.0",
    lifespan=lifespan,
)


@app.exception_handler(HRApiException)
async def hr_api_exception_handler(request: Request, exc: HRApiException):
    """Formats gateway exceptions in the standard HR API envelope."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": exc.message,
            "errors": exc.errors,
        },
    )


@app.exception_handler(RequestValidationError)
async def hr_validation_exception_handler(request: Request, exc: RequestValidationError):
    """Formats validation errors on HR gateway routes into standard HR envelope."""
    if request.url.path.startswith("/api/ai/"):
        err_msgs = []
        for e in exc.errors():
            loc = " -> ".join(str(item) for item in e.get("loc", []))
            err_msgs.append(f"{loc}: {e.get('msg')}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "success": False,
                "message": "Validation error",
                "errors": err_msgs,
            },
        )
    from fastapi.exception_handlers import request_validation_exception_handler

    return await request_validation_exception_handler(request, exc)


def _custom_openapi():
    """Publish the gateway contract explicitly, including runtime auth/errors."""
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(title=app.title, version=app.version, routes=app.routes)

    components = schema.setdefault("components", {})
    security_schemes = components.setdefault("securitySchemes", {})
    security_schemes["bearerAuth"] = {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "JWT",
        "description": "JWT Bearer token provided by Flutter or Web clients.",
    }

    schemas = components.setdefault("schemas", {})
    schemas["HRApiErrorResponse"] = {
        "type": "object",
        "properties": {
            "success": {"type": "boolean", "example": False},
            "message": {"type": "string", "example": "An error occurred."},
            "errors": {"type": "array", "items": {"type": "string"}, "example": []},
        },
        "required": ["success", "message"],
    }

    protected_prefixes = ("/api/", "/insights/")
    for path, methods in schema.get("paths", {}).items():
        if path.startswith("/api/ai/"):
            for operation in methods.values():
                if not isinstance(operation, dict):
                    continue
                operation["security"] = [{"bearerAuth": []}]
                responses = operation.setdefault("responses", {})
                responses["401"] = {
                    "description": "Unauthenticated - Missing or invalid JWT Bearer token",
                    "content": {"application/json": {"schema": {"$ref": "#/components/schemas/HRApiErrorResponse"}}},
                }
                responses["403"] = {
                    "description": "Forbidden - Insufficient permissions or outside scope",
                    "content": {"application/json": {"schema": {"$ref": "#/components/schemas/HRApiErrorResponse"}}},
                }
                responses["404"] = {
                    "description": "Employee or requested resource not found",
                    "content": {"application/json": {"schema": {"$ref": "#/components/schemas/HRApiErrorResponse"}}},
                }
                responses["422"] = {
                    "description": "Validation error",
                    "content": {"application/json": {"schema": {"$ref": "#/components/schemas/HRApiErrorResponse"}}},
                }
                responses["429"] = {
                    "description": "Rate limit exceeded",
                    "content": {"application/json": {"schema": {"$ref": "#/components/schemas/HRApiErrorResponse"}}},
                }
                responses["502"] = {
                    "description": "AI provider is temporarily unavailable",
                    "content": {"application/json": {"schema": {"$ref": "#/components/schemas/HRApiErrorResponse"}}},
                }
                responses["503"] = {
                    "description": "AI service dependency is unavailable",
                    "content": {"application/json": {"schema": {"$ref": "#/components/schemas/HRApiErrorResponse"}}},
                }
                responses["504"] = {
                    "description": "AI provider timed out",
                    "content": {"application/json": {"schema": {"$ref": "#/components/schemas/HRApiErrorResponse"}}},
                }
        elif path.startswith(protected_prefixes):
            for operation in methods.values():
                if not isinstance(operation, dict):
                    continue
                responses = operation.setdefault("responses", {})
                responses.setdefault("401", {"description": "Missing or invalid gateway authentication"})
                responses.setdefault("403", {"description": "Caller is outside the permitted scope"})
                responses.setdefault("404", {"description": "Requested resource was not found"})
                responses.setdefault("429", {"description": "Rate limit exceeded; retry according to gateway policy"})
                responses.setdefault("502", {"description": "AI provider is temporarily unavailable"})
                responses.setdefault("503", {"description": "AI service dependency is unavailable"})
                responses.setdefault("504", {"description": "AI provider timed out"})
                for parameter in operation.get("parameters", []):
                    if parameter.get("name") == "X-Caller-Role":
                        parameter["required"] = True
    app.openapi_schema = schema
    return schema


app.openapi = _custom_openapi

app.state.db_initialized = False
app.state.db_initialization_error = None

# Register Health and API Routers
app.include_router(health_router)
app.include_router(health_router, prefix="/api")
app.include_router(career_coach_router, prefix="/api")
app.include_router(performance_insight_router, prefix="/api")
app.include_router(policy_assistant_router, prefix="/api")
app.include_router(evaluation_draft_router, prefix="/api")
app.include_router(skill_gap_router, prefix="/api")
app.include_router(attention_signal_router, prefix="/api")
app.include_router(team_insight_router, prefix="/api")
app.include_router(insight_snapshots_router, prefix="/api/insights")
app.include_router(insight_snapshots_router, prefix="/insights", include_in_schema=False)
app.include_router(hr_gateway_router)



@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok", "service": "Smart HR Management System - AI Service"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)



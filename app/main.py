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

from fastapi import FastAPI
from sqlalchemy.exc import SQLAlchemyError

import app.models
from app.api.career_coach import router as career_coach_router
from app.api.policy_assistant import router as policy_assistant_router
from app.db.session import Base, engine

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize database schema at startup if database is available
    try:
        Base.metadata.create_all(bind=engine)
    except (SQLAlchemyError, OSError) as exc:
        logger.warning("Database schema initialization warning: %s", exc)
    yield


app = FastAPI(
    title="Smart HR Management System - AI Service",
    version="0.1.0",
    lifespan=lifespan,
)

# Register API Routers
app.include_router(career_coach_router, prefix="/api")
app.include_router(policy_assistant_router, prefix="/api")



@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok", "service": "Smart HR Management System - AI Service"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)



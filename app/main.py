from dotenv import load_dotenv

# Automatically load environment variables from .env file at startup
load_dotenv()

from fastapi import FastAPI
from app.db.session import engine, Base
import app.models  # Register models
from app.api.career_coach import router as career_coach_router

# Initialize SQLite database schema
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Smart HR Management System - AI Service",
    version="0.1.0"
)

# Register API Router
app.include_router(career_coach_router, prefix="/api")

@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok", "service": "Smart HR Management System - AI Service"}

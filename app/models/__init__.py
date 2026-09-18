import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from app.db.session import Base


def utc_now():
    return datetime.now(timezone.utc)


class Employee(Base):
    __tablename__ = "employees"

    id = Column(String(50), primary_key=True, index=True)  # e.g., "EMP-001"
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    role_title = Column(String(100), nullable=False)
    department = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    # 1-to-many relationships (each record belongs to exactly one Employee)
    performance_records = relationship("PerformanceRecord", back_populates="employee", cascade="all, delete-orphan")
    goals = relationship("Goal", back_populates="employee", cascade="all, delete-orphan")
    skills = relationship("Skill", back_populates="employee", cascade="all, delete-orphan")
    task_outcomes = relationship("TaskOutcome", back_populates="employee", cascade="all, delete-orphan")
    evaluation_themes = relationship("EvaluationTheme", back_populates="employee", cascade="all, delete-orphan")
    chat_sessions = relationship("ChatSession", back_populates="employee", cascade="all, delete-orphan")


class PerformanceRecord(Base):
    __tablename__ = "performance_records"

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(String(50), ForeignKey("employees.id"), nullable=False, index=True)
    period = Column(String(20), nullable=False, index=True)  # e.g., "2026-Q3"
    overall_score = Column(Float, nullable=False)
    task_completion_rate = Column(Float, nullable=False)     # percentage e.g., 92.5
    goal_achievement_rate = Column(Float, nullable=False)    # percentage e.g., 88.0
    attendance_rate = Column(Float, nullable=False)          # percentage e.g., 99.0
    is_approved = Column(Boolean, default=False, server_default="0", nullable=False)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    employee = relationship("Employee", back_populates="performance_records")


class Goal(Base):
    __tablename__ = "goals"

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(String(50), ForeignKey("employees.id"), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    progress = Column(Float, default=0.0, nullable=False)    # 0 to 100
    status = Column(String(50), default="in_progress", nullable=False)  # "in_progress", "completed", "delayed"
    deadline = Column(String(50), nullable=True)
    period = Column(String(20), nullable=True)               # e.g., "2026-Q3"
    is_approved = Column(Boolean, default=False, server_default="0", nullable=False)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    employee = relationship("Employee", back_populates="goals")


class Skill(Base):
    __tablename__ = "skills"

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(String(50), ForeignKey("employees.id"), nullable=False, index=True)
    name = Column(String(100), nullable=False)
    level = Column(String(50), nullable=False)               # "Beginner", "Intermediate", "Advanced", "Expert"
    evidence = Column(Text, nullable=True)
    is_approved = Column(Boolean, default=False, server_default="0", nullable=False)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    employee = relationship("Employee", back_populates="skills")


class TaskOutcome(Base):
    __tablename__ = "task_outcomes"

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(String(50), ForeignKey("employees.id"), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    status = Column(String(50), nullable=False)              # "completed", "in_progress", "blocked"
    outcome = Column(Text, nullable=True)
    completion_date = Column(String(50), nullable=True)
    period = Column(String(20), nullable=True)               # e.g., "2026-Q3"
    is_approved = Column(Boolean, default=False, server_default="0", nullable=False)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    employee = relationship("Employee", back_populates="task_outcomes")


class EvaluationTheme(Base):
    __tablename__ = "evaluation_themes"

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(String(50), ForeignKey("employees.id"), nullable=False, index=True)
    theme = Column(String(150), nullable=False)
    sentiment = Column(String(50), nullable=False)           # "positive", "neutral", "needs_improvement"
    evidence = Column(Text, nullable=False)
    period = Column(String(20), nullable=True)               # e.g., "2026-Q3"
    is_approved = Column(Boolean, default=False, server_default="0", nullable=False)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    employee = relationship("Employee", back_populates="evaluation_themes")


class CompanyPolicy(Base):
    __tablename__ = "company_policies"

    id = Column(Integer, primary_key=True, index=True)
    policy_code = Column(String(50), unique=True, index=True, nullable=False)
    title = Column(String(255), nullable=False)
    category = Column(String(100), nullable=False)
    content = Column(Text, nullable=False)
    summary = Column(Text, nullable=True)
    version = Column(String(20), default="1.0", nullable=False)
    is_active = Column(Boolean, default=True, server_default="1", nullable=False)
    is_approved = Column(Boolean, default=True, server_default="1", nullable=False)
    created_at = Column(DateTime, default=utc_now, nullable=False)


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    employee_id = Column(String(50), ForeignKey("employees.id"), nullable=False, index=True)
    title = Column(String(255), nullable=True)
    summary = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utc_now, nullable=False)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now, nullable=False)

    employee = relationship("Employee", back_populates="chat_sessions")
    messages = relationship(
        "ChatMessage",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="ChatMessage.created_at",
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    session_id = Column(String(36), ForeignKey("chat_sessions.id"), nullable=False, index=True)
    role = Column(String(20), nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)
    embedding = Column(Text, nullable=True)  # JSON-serialized vector embedding
    created_at = Column(DateTime, default=utc_now, nullable=False)

    session = relationship("ChatSession", back_populates="messages")



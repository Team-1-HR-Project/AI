from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Float, Text, DateTime, ForeignKey
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


class PerformanceRecord(Base):
    __tablename__ = "performance_records"

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(String(50), ForeignKey("employees.id"), nullable=False, index=True)
    period = Column(String(20), nullable=False, index=True)  # e.g., "2026-Q3"
    overall_score = Column(Float, nullable=False)
    task_completion_rate = Column(Float, nullable=False)     # percentage e.g., 92.5
    goal_achievement_rate = Column(Float, nullable=False)    # percentage e.g., 88.0
    attendance_rate = Column(Float, nullable=False)          # percentage e.g., 99.0
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
    created_at = Column(DateTime, default=utc_now, nullable=False)

    employee = relationship("Employee", back_populates="goals")


class Skill(Base):
    __tablename__ = "skills"

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(String(50), ForeignKey("employees.id"), nullable=False, index=True)
    name = Column(String(100), nullable=False)
    level = Column(String(50), nullable=False)               # "Beginner", "Intermediate", "Advanced", "Expert"
    evidence = Column(Text, nullable=True)
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
    created_at = Column(DateTime, default=utc_now, nullable=False)

    employee = relationship("Employee", back_populates="evaluation_themes")

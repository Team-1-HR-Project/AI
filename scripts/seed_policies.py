"""Seed script for company policies in MySQL and SQLite."""

import os
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.db.session import Base, SessionLocal, engine
from app.models import (
    CompanyPolicy,
    Employee,
    EvaluationTheme,
    Goal,
    PerformanceRecord,
    Skill,
    TaskOutcome,
)

DEMO_POLICIES = [
    {
        "policy_code": "POL-LEAVE-001",
        "title": "Annual Leave & Time Off Policy",
        "category": "Leave & Attendance",
        "version": "1.0",
        "summary": "Rules regarding employee annual leave entitlements, monthly accrual rates, advance notice periods, and year-end rollover limitations.",
        "content": (
            "1. Entitlement & Accrual:\n"
            "All regular full-time employees are entitled to 21 working days of paid annual leave per calendar year. "
            "Leave accrues on a monthly basis at the rate of 1.75 working days per completed month of active employment.\n\n"
            "2. Request & Approval Procedure:\n"
            "Leave requests exceeding three (3) consecutive business days must be submitted through the company HR portal "
            "at least two (2) weeks in advance. All annual leave requires prior written approval from the direct department manager.\n\n"
            "3. Rollover & Expiration:\n"
            "A maximum of five (5) unused annual leave days may be carried forward into the subsequent calendar year. "
            "Any carried-over days must be fully utilized by March 31st of that year, after which they will lapse.\n\n"
            "4. Unpaid & Emergency Leave:\n"
            "Absences beyond accrued leave balances or taken without prior approval will be classified as unpaid leave "
            "and may trigger management review."
        ),
        "is_active": True,
        "is_approved": True,
    },
    {
        "policy_code": "POL-WORK-001",
        "title": "Standard Working Hours & Flexible Schedule Policy",
        "category": "Workplace & Attendance",
        "version": "1.0",
        "summary": "Standard 40-hour workweek specifications, mandatory core collaboration hours, lunch breaks, and overtime pre-approval requirements.",
        "content": (
            "1. Standard Workweek:\n"
            "The standard working week consists of 40 hours, scheduled Monday through Friday. "
            "Standard office hours are from 9:00 AM to 5:00 PM, including a one-hour meal break.\n\n"
            "2. Flexible Hours & Core Collaboration Window:\n"
            "Employees may request flexible start and end times between 8:00 AM and 6:00 PM subject to departmental approval. "
            "However, all employees must be present, online, and actively available during mandatory Core Collaboration Hours from 10:00 AM to 3:00 PM.\n\n"
            "3. Overtime Guidelines:\n"
            "Non-exempt employees working beyond 40 hours in a given workweek must receive advance written approval "
            "from their department manager prior to working additional hours."
        ),
        "is_active": True,
        "is_approved": True,
    },
    {
        "policy_code": "POL-REMOTE-001",
        "title": "Hybrid & Remote Work Arrangement Policy",
        "category": "Workplace Guidelines",
        "version": "1.0",
        "summary": "Eligibility criteria, maximum remote days, cybersecurity and IT requirements, and communication availability expectations for hybrid and remote work.",
        "content": (
            "1. Eligibility:\n"
            "Employees who have successfully passed their initial probationary period (typically 90 days) "
            "and maintain satisfactory performance ratings are eligible to request hybrid work arrangements of up to two (2) remote days per week.\n\n"
            "2. Workspace & Technology Requirements:\n"
            "Remote employees must operate from a secure, private location with high-speed internet (at least 25 Mbps download). "
            "All work must be conducted on company-managed devices with corporate VPN enabled at all times.\n\n"
            "3. Communication & Meeting Participation:\n"
            "Remote team members must remain reachable via Slack and corporate email during standard core working hours "
            "and are expected to participate in scheduled video meetings with webcams active when requested."
        ),
        "is_active": True,
        "is_approved": True,
    },
    {
        "policy_code": "POL-CONDUCT-001",
        "title": "Code of Professional Conduct & Workplace Ethics",
        "category": "Code of Conduct",
        "version": "1.0",
        "summary": "Zero-tolerance anti-harassment standards, confidentiality guidelines for customer and company data, conflict of interest declarations, and reporting protocols.",
        "content": (
            "1. Workplace Respect & Zero Tolerance:\n"
            "The company is committed to providing a professional, respectful work environment free from discrimination, "
            "sexual harassment, bullying, or retaliation. Any verified violation will result in disciplinary action up to and including immediate termination.\n\n"
            "2. Confidentiality & Data Protection:\n"
            "Employees must protect all proprietary trade secrets, customer records, employee personal data, and source code. "
            "Confidential information must not be stored on personal devices or transmitted through unauthorized public networks or non-company channels.\n\n"
            "3. Conflict of Interest:\n"
            "Employees must avoid any outside commercial, consulting, or personal activity that conflicts with the company's business interests. "
            "Any secondary employment or commercial ventures must be formally disclosed to HR for review."
        ),
        "is_active": True,
        "is_approved": True,
    },
]


def seed_company_policies(db=None) -> list[CompanyPolicy]:
    """Ensures policy table exists and seeds demo company policies idempotently."""
    should_close = False
    if db is None:
        Base.metadata.create_all(bind=engine)
        db = SessionLocal()
        should_close = True

    inserted_or_updated = []
    try:
        for policy_data in DEMO_POLICIES:
            existing = (
                db.query(CompanyPolicy)
                .filter(CompanyPolicy.policy_code == policy_data["policy_code"])
                .first()
            )
            if existing:
                # Update fields to ensure latest content and active/approved flags
                existing.title = policy_data["title"]
                existing.category = policy_data["category"]
                existing.version = policy_data["version"]
                existing.summary = policy_data["summary"]
                existing.content = policy_data["content"]
                existing.is_active = policy_data["is_active"]
                existing.is_approved = policy_data["is_approved"]
                inserted_or_updated.append(existing)
            else:
                new_policy = CompanyPolicy(
                    policy_code=policy_data["policy_code"],
                    title=policy_data["title"],
                    category=policy_data["category"],
                    version=policy_data["version"],
                    summary=policy_data["summary"],
                    content=policy_data["content"],
                    is_active=policy_data["is_active"],
                    is_approved=policy_data["is_approved"],
                    created_at=datetime.now(timezone.utc),
                )
                db.add(new_policy)
                inserted_or_updated.append(new_policy)

        db.commit()
        return inserted_or_updated
    except Exception as exc:
        db.rollback()
        print(f"Error seeding company policies: {exc}")
        raise
    finally:
        if should_close:
            db.close()


# ===========================================================================
# Demo Performance Insight Data (Development / Manual Testing Only)
# ===========================================================================
DEMO_PERF_INSIGHT_EMPLOYEE = {
    "id": "EMP-PERF-DEMO",
    "first_name": "Demo",
    "last_name": "Performance",
    "role_title": "Senior Systems Engineer",
    "department": "Engineering",
}

DEMO_PERF_INSIGHT_RECORDS = [
    {
        "period": "2026-Q2",
        "overall_score": 82.0,
        "task_completion_rate": 85.0,
        "goal_achievement_rate": 80.0,
        "attendance_rate": 96.0,
        "is_approved": True,
    },
    {
        "period": "2026-Q3",
        "overall_score": 90.0,
        "task_completion_rate": 93.0,
        "goal_achievement_rate": 88.0,
        "attendance_rate": 94.0,
        "is_approved": True,
    },
]


def seed_performance_insight_demo(db=None) -> tuple[Employee, list[PerformanceRecord]]:
    """Seeds dedicated demo employee and approved performance records for manual testing.

    Idempotent:
    - If employee exists, updates profile attributes; otherwise creates it.
    - If records exist for (employee_id, period), updates metrics; otherwise creates them.
    - Cleans up any duplicate records for the same period.
    - Leaves all other employee and performance records untouched.
    - This data is strictly for development, demo, and manual testing.
    """
    should_close = False
    if db is None:
        Base.metadata.create_all(bind=engine)
        db = SessionLocal()
        should_close = True

    try:
        emp_data = DEMO_PERF_INSIGHT_EMPLOYEE
        employee = db.query(Employee).filter(Employee.id == emp_data["id"]).first()
        if employee:
            employee.first_name = emp_data["first_name"]
            employee.last_name = emp_data["last_name"]
            employee.role_title = emp_data["role_title"]
            employee.department = emp_data["department"]
        else:
            employee = Employee(
                id=emp_data["id"],
                first_name=emp_data["first_name"],
                last_name=emp_data["last_name"],
                role_title=emp_data["role_title"],
                department=emp_data["department"],
                created_at=datetime.now(timezone.utc),
            )
            db.add(employee)

        db.flush()

        seeded_records = []
        for rec_data in DEMO_PERF_INSIGHT_RECORDS:
            existing_records = (
                db.query(PerformanceRecord)
                .filter(
                    PerformanceRecord.employee_id == emp_data["id"],
                    PerformanceRecord.period == rec_data["period"],
                )
                .all()
            )
            if existing_records:
                record = existing_records[0]
                record.overall_score = rec_data["overall_score"]
                record.task_completion_rate = rec_data["task_completion_rate"]
                record.goal_achievement_rate = rec_data["goal_achievement_rate"]
                record.attendance_rate = rec_data["attendance_rate"]
                record.is_approved = rec_data["is_approved"]
                for duplicate in existing_records[1:]:
                    db.delete(duplicate)
                seeded_records.append(record)
            else:
                record = PerformanceRecord(
                    employee_id=emp_data["id"],
                    period=rec_data["period"],
                    overall_score=rec_data["overall_score"],
                    task_completion_rate=rec_data["task_completion_rate"],
                    goal_achievement_rate=rec_data["goal_achievement_rate"],
                    attendance_rate=rec_data["attendance_rate"],
                    is_approved=rec_data["is_approved"],
                    created_at=datetime.now(timezone.utc),
                )
                db.add(record)
                seeded_records.append(record)

        db.commit()
        return employee, seeded_records
    except Exception as exc:
        db.rollback()
        print(f"Error seeding performance insight demo data: {exc}")
        raise
    finally:
        if should_close:
            db.close()


# ===========================================================================
# Manual Test Employee Data (EMP-MANUAL-TEST)
# ===========================================================================
DEMO_MANUAL_TEST_EMPLOYEE = {
    "id": "EMP-MANUAL-TEST",
    "first_name": "Alex",
    "last_name": "Taylor",
    "role_title": "Senior Backend Engineer",
    "department": "Platform Engineering",
}

DEMO_MANUAL_TEST_PERFORMANCE = {
    "period": "2026-Q3",
    "overall_score": 93.5,
    "task_completion_rate": 96.0,
    "goal_achievement_rate": 91.0,
    "attendance_rate": 99.0,
    "is_approved": True,
}

DEMO_MANUAL_TEST_GOAL = {
    "title": "Migrate distributed caching to Redis Cluster",
    "progress": 85.0,
    "status": "in_progress",
    "deadline": "2026-10-30",
    "period": "2026-Q3",
    "is_approved": True,
}

DEMO_MANUAL_TEST_SKILL = {
    "name": "Python, FastAPI & Async Architecture",
    "level": "Expert",
    "evidence": "Architected core event-driven API gateway with 99.95% uptime",
    "is_approved": True,
}

DEMO_MANUAL_TEST_TASK_OUTCOME = {
    "title": "Database Query Indexing & Connection Pooling Refactor",
    "status": "completed",
    "outcome": "Reduced P99 API response latency by 42% under load",
    "completion_date": "2026-08-20",
    "period": "2026-Q3",
    "is_approved": True,
}

DEMO_MANUAL_TEST_EVALUATION_THEME = {
    "theme": "Technical Problem Solving & Mentorship",
    "sentiment": "positive",
    "evidence": "Excellent root cause analysis; opportunity to run more knowledge sharing sessions for junior peers",
    "period": "2026-Q3",
    "is_approved": True,
}


def seed_manual_test_data(db=None) -> Employee:
    """Ensures EMP-MANUAL-TEST and its related records exist for manual testing and CI.

    Idempotent:
    - If employee exists, updates profile attributes; otherwise creates it.
    - If related records exist, updates their fields; otherwise creates them.
    - Leaves all other employee and performance records untouched.
    """
    should_close = False
    if db is None:
        Base.metadata.create_all(bind=engine)
        db = SessionLocal()
        should_close = True

    try:
        emp_data = DEMO_MANUAL_TEST_EMPLOYEE
        employee = db.query(Employee).filter(Employee.id == emp_data["id"]).first()
        if employee:
            employee.first_name = emp_data["first_name"]
            employee.last_name = emp_data["last_name"]
            employee.role_title = emp_data["role_title"]
            employee.department = emp_data["department"]
        else:
            employee = Employee(
                id=emp_data["id"],
                first_name=emp_data["first_name"],
                last_name=emp_data["last_name"],
                role_title=emp_data["role_title"],
                department=emp_data["department"],
                created_at=datetime.now(timezone.utc),
            )
            db.add(employee)

        db.flush()

        # Performance Record
        perf_data = DEMO_MANUAL_TEST_PERFORMANCE
        perf = (
            db.query(PerformanceRecord)
            .filter(
                PerformanceRecord.employee_id == emp_data["id"],
                PerformanceRecord.period == perf_data["period"],
            )
            .first()
        )
        if perf:
            perf.overall_score = perf_data["overall_score"]
            perf.task_completion_rate = perf_data["task_completion_rate"]
            perf.goal_achievement_rate = perf_data["goal_achievement_rate"]
            perf.attendance_rate = perf_data["attendance_rate"]
            perf.is_approved = perf_data["is_approved"]
        else:
            perf = PerformanceRecord(
                employee_id=emp_data["id"],
                period=perf_data["period"],
                overall_score=perf_data["overall_score"],
                task_completion_rate=perf_data["task_completion_rate"],
                goal_achievement_rate=perf_data["goal_achievement_rate"],
                attendance_rate=perf_data["attendance_rate"],
                is_approved=perf_data["is_approved"],
                created_at=datetime.now(timezone.utc),
            )
            db.add(perf)

        # Goal
        goal_data = DEMO_MANUAL_TEST_GOAL
        goal = (
            db.query(Goal)
            .filter(
                Goal.employee_id == emp_data["id"],
                Goal.title == goal_data["title"],
            )
            .first()
        )
        if goal:
            goal.progress = goal_data["progress"]
            goal.status = goal_data["status"]
            goal.deadline = goal_data["deadline"]
            goal.period = goal_data["period"]
            goal.is_approved = goal_data["is_approved"]
        else:
            goal = Goal(
                employee_id=emp_data["id"],
                title=goal_data["title"],
                progress=goal_data["progress"],
                status=goal_data["status"],
                deadline=goal_data["deadline"],
                period=goal_data["period"],
                is_approved=goal_data["is_approved"],
                created_at=datetime.now(timezone.utc),
            )
            db.add(goal)

        # Skill
        skill_data = DEMO_MANUAL_TEST_SKILL
        skill = (
            db.query(Skill)
            .filter(
                Skill.employee_id == emp_data["id"],
                Skill.name == skill_data["name"],
            )
            .first()
        )
        if skill:
            skill.level = skill_data["level"]
            skill.evidence = skill_data["evidence"]
            skill.is_approved = skill_data["is_approved"]
        else:
            skill = Skill(
                employee_id=emp_data["id"],
                name=skill_data["name"],
                level=skill_data["level"],
                evidence=skill_data["evidence"],
                is_approved=skill_data["is_approved"],
                created_at=datetime.now(timezone.utc),
            )
            db.add(skill)

        # Task Outcome
        task_data = DEMO_MANUAL_TEST_TASK_OUTCOME
        task = (
            db.query(TaskOutcome)
            .filter(
                TaskOutcome.employee_id == emp_data["id"],
                TaskOutcome.title == task_data["title"],
            )
            .first()
        )
        if task:
            task.status = task_data["status"]
            task.outcome = task_data["outcome"]
            task.completion_date = task_data["completion_date"]
            task.period = task_data["period"]
            task.is_approved = task_data["is_approved"]
        else:
            task = TaskOutcome(
                employee_id=emp_data["id"],
                title=task_data["title"],
                status=task_data["status"],
                outcome=task_data["outcome"],
                completion_date=task_data["completion_date"],
                period=task_data["period"],
                is_approved=task_data["is_approved"],
                created_at=datetime.now(timezone.utc),
            )
            db.add(task)

        # Evaluation Theme
        theme_data = DEMO_MANUAL_TEST_EVALUATION_THEME
        theme = (
            db.query(EvaluationTheme)
            .filter(
                EvaluationTheme.employee_id == emp_data["id"],
                EvaluationTheme.theme == theme_data["theme"],
            )
            .first()
        )
        if theme:
            theme.sentiment = theme_data["sentiment"]
            theme.evidence = theme_data["evidence"]
            theme.period = theme_data["period"]
            theme.is_approved = theme_data["is_approved"]
        else:
            theme = EvaluationTheme(
                employee_id=emp_data["id"],
                theme=theme_data["theme"],
                sentiment=theme_data["sentiment"],
                evidence=theme_data["evidence"],
                period=theme_data["period"],
                is_approved=theme_data["is_approved"],
                created_at=datetime.now(timezone.utc),
            )
            db.add(theme)

        db.commit()
        return employee
    except Exception as exc:
        db.rollback()
        print(f"Error seeding manual test data: {exc}")
        raise
    finally:
        if should_close:
            db.close()


def seed_all(db=None):
    """Seeds company policies, manual test data, and demo performance insight data."""
    seed_company_policies(db=db)
    seed_manual_test_data(db=db)
    seed_performance_insight_demo(db=db)


if __name__ == "__main__":
    seed_all()


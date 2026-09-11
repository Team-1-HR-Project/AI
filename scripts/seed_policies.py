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
from app.models import CompanyPolicy

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


def seed_company_policies() -> list[CompanyPolicy]:
    """Ensures policy table exists and seeds demo company policies idempotently."""
    print("Ensuring database tables exist...")
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
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
                print(f"Updated policy: {existing.policy_code} - {existing.title}")
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
                print(f"Inserted policy: {new_policy.policy_code} - {new_policy.title}")

        db.commit()
        print(f"Successfully seeded {len(inserted_or_updated)} company policies.")
        return inserted_or_updated
    except Exception as exc:
        db.rollback()
        print(f"Error seeding company policies: {exc}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    seed_company_policies()

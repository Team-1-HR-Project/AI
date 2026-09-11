import os
import sqlite3
import sys
from datetime import datetime, timezone

import pymysql
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


def parse_date(date_str):
    if not date_str:
        return datetime.now(timezone.utc)
    if isinstance(date_str, datetime):
        return date_str if date_str.tzinfo else date_str.replace(tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(date_str)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        for fmt in ('%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d'):
            try:
                dt = datetime.strptime(date_str, fmt).replace(tzinfo=timezone.utc)
                return dt
            except ValueError:
                pass
    return datetime.now(timezone.utc)

def ensure_database_exists():
    db_user = os.getenv('DB_USER', 'root')
    db_password = os.getenv('DB_PASSWORD', '')
    db_host = os.getenv('DB_HOST', '127.0.0.1')
    db_port = int(os.getenv('DB_PORT', '3306'))
    db_name = os.getenv('DB_NAME', 'hr_system')

    print(f'Connecting to MySQL server at {db_host}:{db_port} as user {db_user}...')
    conn = pymysql.connect(
        host=db_host,
        port=db_port,
        user=db_user,
        password=db_password,
        charset='utf8mb4',
        autocommit=True
    )
    with conn.cursor() as cursor:
        cursor.execute(f'CREATE DATABASE IF NOT EXISTS {db_name} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;')
        print(f'Database {db_name} verified / created successfully.')
    conn.close()

def migrate_data():
    sqlite_path = os.path.join(PROJECT_ROOT, 'hr_system.db')
    if not os.path.exists(sqlite_path):
        print(f'SQLite database file not found at {sqlite_path}. Schema will be created without data migration.')
        Base.metadata.create_all(bind=engine)
        return

    print('Creating tables in MySQL if not present...')
    Base.metadata.create_all(bind=engine)

    print(f'Reading records from SQLite source: {sqlite_path}...')
    sqlite_conn = sqlite3.connect(sqlite_path)
    sqlite_conn.row_factory = sqlite3.Row

    db = SessionLocal()
    try:
        sqlite_employees = sqlite_conn.execute('SELECT * FROM employees').fetchall()
        print(f'Migrating {len(sqlite_employees)} Employees...')
        for row in sqlite_employees:
            existing = db.query(Employee).filter(Employee.id == row['id']).first()
            if not existing:
                emp = Employee(
                    id=row['id'],
                    first_name=row['first_name'],
                    last_name=row['last_name'],
                    role_title=row['role_title'],
                    department=row['department'],
                    created_at=parse_date(row['created_at']),
                )
                db.add(emp)
        db.commit()

        sqlite_perf = sqlite_conn.execute('SELECT * FROM performance_records').fetchall()
        print(f'Migrating {len(sqlite_perf)} Performance Records...')
        for row in sqlite_perf:
            existing = db.query(PerformanceRecord).filter(PerformanceRecord.id == row['id']).first()
            if not existing:
                rec = PerformanceRecord(
                    id=row['id'],
                    employee_id=row['employee_id'],
                    period=row['period'],
                    overall_score=row['overall_score'],
                    task_completion_rate=row['task_completion_rate'],
                    goal_achievement_rate=row['goal_achievement_rate'],
                    attendance_rate=row['attendance_rate'],
                    created_at=parse_date(row['created_at']),
                )
                db.add(rec)
        db.commit()

        sqlite_goals = sqlite_conn.execute('SELECT * FROM goals').fetchall()
        print(f'Migrating {len(sqlite_goals)} Goals...')
        for row in sqlite_goals:
            existing = db.query(Goal).filter(Goal.id == row['id']).first()
            if not existing:
                g = Goal(
                    id=row['id'],
                    employee_id=row['employee_id'],
                    title=row['title'],
                    progress=row['progress'],
                    status=row['status'],
                    deadline=row['deadline'],
                    period=row['period'],
                    created_at=parse_date(row['created_at']),
                )
                db.add(g)
        db.commit()

        sqlite_skills = sqlite_conn.execute('SELECT * FROM skills').fetchall()
        print(f'Migrating {len(sqlite_skills)} Skills...')
        for row in sqlite_skills:
            existing = db.query(Skill).filter(Skill.id == row['id']).first()
            if not existing:
                s = Skill(
                    id=row['id'],
                    employee_id=row['employee_id'],
                    name=row['name'],
                    level=row['level'],
                    evidence=row['evidence'],
                    created_at=parse_date(row['created_at']),
                )
                db.add(s)
        db.commit()

        sqlite_tasks = sqlite_conn.execute('SELECT * FROM task_outcomes').fetchall()
        print(f'Migrating {len(sqlite_tasks)} Task Outcomes...')
        for row in sqlite_tasks:
            existing = db.query(TaskOutcome).filter(TaskOutcome.id == row['id']).first()
            if not existing:
                t = TaskOutcome(
                    id=row['id'],
                    employee_id=row['employee_id'],
                    title=row['title'],
                    status=row['status'],
                    outcome=row['outcome'],
                    completion_date=row['completion_date'],
                    period=row['period'],
                    created_at=parse_date(row['created_at']),
                )
                db.add(t)
        db.commit()

        sqlite_themes = sqlite_conn.execute('SELECT * FROM evaluation_themes').fetchall()
        print(f'Migrating {len(sqlite_themes)} Evaluation Themes...')
        for row in sqlite_themes:
            existing = db.query(EvaluationTheme).filter(EvaluationTheme.id == row['id']).first()
            if not existing:
                th = EvaluationTheme(
                    id=row['id'],
                    employee_id=row['employee_id'],
                    theme=row['theme'],
                    sentiment=row['sentiment'],
                    evidence=row['evidence'],
                    period=row['period'],
                    created_at=parse_date(row['created_at']),
                )
                db.add(th)
        db.commit()

        try:
            sqlite_policies = sqlite_conn.execute('SELECT * FROM company_policies').fetchall()
            print(f'Migrating {len(sqlite_policies)} Company Policies...')
            for row in sqlite_policies:
                existing = db.query(CompanyPolicy).filter(CompanyPolicy.policy_code == row['policy_code']).first()
                if not existing:
                    pol = CompanyPolicy(
                        id=row['id'],
                        policy_code=row['policy_code'],
                        title=row['title'],
                        category=row['category'],
                        content=row['content'],
                        summary=row['summary'],
                        version=row['version'],
                        is_active=bool(row['is_active']),
                        is_approved=bool(row['is_approved']),
                        created_at=parse_date(row['created_at']),
                    )
                    db.add(pol)
            db.commit()
        except sqlite3.OperationalError:
            pass

        print('Verification Report in MySQL:')
        counts = {
            'employees': db.query(Employee).count(),
            'performance_records': db.query(PerformanceRecord).count(),
            'goals': db.query(Goal).count(),
            'skills': db.query(Skill).count(),
            'task_outcomes': db.query(TaskOutcome).count(),
            'evaluation_themes': db.query(EvaluationTheme).count(),
            'company_policies': db.query(CompanyPolicy).count(),
        }
        for table, count in counts.items():
            print(f'  Table {table}: {count} records in MySQL')

        print('Data migration from SQLite to MySQL completed successfully!')
    finally:
        sqlite_conn.close()
        db.close()

if __name__ == '__main__':
    ensure_database_exists()
    migrate_data()

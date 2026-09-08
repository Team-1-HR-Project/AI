from typing import Optional, Dict, Any, List
from sqlalchemy.orm import Session
from app.models import (
    Employee,
    PerformanceRecord,
    Goal,
    Skill,
    TaskOutcome,
    EvaluationTheme,
)

REQUIRED_CATEGORIES = [
    "performance",
    "goals",
    "skills",
    "task_outcomes",
    "evaluation_themes",
]

class CareerCoachContextBuilder:
    @staticmethod
    def build_context(
        db: Session,
        employee_id: str,
        period: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Gathers and prepares sanitized employee data for the AI Career Coach.
        Guarantees strict employee data isolation and omits sensitive fields.
        Detects missing data categories without fabricating facts.
        """
        employee = db.query(Employee).filter(Employee.id == employee_id).first()
        if not employee:
            return {
                "has_sufficient_data": False,
                "missing_categories": REQUIRED_CATEGORIES.copy(),
                "error": f"Employee with id '{employee_id}' not found.",
                "context": None,
            }

        # 1. Employee baseline info (safe fields only)
        employee_dict = {
            "id": employee.id,
            "role_title": employee.role_title,
            "department": employee.department,
        }

        # 2. Performance records (strictly isolated to employee.id)
        perf_query = db.query(PerformanceRecord).filter(
            PerformanceRecord.employee_id == employee.id
        )
        if period:
            perf_query = perf_query.filter(PerformanceRecord.period == period)
        perf_records = perf_query.all()

        performance_data: List[Dict[str, Any]] = [
            {
                "period": p.period,
                "overall_score": p.overall_score,
                "task_completion_rate": p.task_completion_rate,
                "goal_achievement_rate": p.goal_achievement_rate,
                "attendance_rate": p.attendance_rate,
            }
            for p in perf_records
        ]

        # 3. Goals (strictly isolated to employee.id)
        goal_query = db.query(Goal).filter(
            Goal.employee_id == employee.id
        )
        if period:
            # Match period or unassigned general goals
            goal_query = goal_query.filter((Goal.period == period) | (Goal.period.is_(None)))
        goals = goal_query.all()

        goals_data: List[Dict[str, Any]] = [
            {
                "title": g.title,
                "progress": g.progress,
                "status": g.status,
                "deadline": g.deadline,
                "period": g.period,
            }
            for g in goals
        ]

        # 4. Skills (strictly isolated to employee.id - skills are cumulative across periods)
        skills = db.query(Skill).filter(
            Skill.employee_id == employee.id
        ).all()

        skills_data: List[Dict[str, Any]] = [
            {
                "name": s.name,
                "level": s.level,
                "evidence": s.evidence,
            }
            for s in skills
        ]

        # 5. Task Outcomes (strictly isolated to employee.id)
        task_query = db.query(TaskOutcome).filter(
            TaskOutcome.employee_id == employee.id
        )
        if period:
            task_query = task_query.filter((TaskOutcome.period == period) | (TaskOutcome.period.is_(None)))
        tasks = task_query.all()

        task_outcomes_data: List[Dict[str, Any]] = [
            {
                "title": t.title,
                "status": t.status,
                "outcome": t.outcome,
                "completion_date": t.completion_date,
                "period": t.period,
            }
            for t in tasks
        ]

        # 6. Evaluation Themes (strictly isolated to employee.id)
        theme_query = db.query(EvaluationTheme).filter(
            EvaluationTheme.employee_id == employee.id
        )
        if period:
            theme_query = theme_query.filter((EvaluationTheme.period == period) | (EvaluationTheme.period.is_(None)))
        themes = theme_query.all()

        evaluation_themes_data: List[Dict[str, Any]] = [
            {
                "theme": et.theme,
                "sentiment": et.sentiment,
                "evidence": et.evidence,
                "period": et.period,
            }
            for et in themes
        ]

        # Assemble structured context
        context = {
            "employee": employee_dict,
            "performance": performance_data,
            "goals": goals_data,
            "skills": skills_data,
            "task_outcomes": task_outcomes_data,
            "evaluation_themes": evaluation_themes_data,
        }

        # Check for missing categories
        missing_categories: List[str] = []
        if not performance_data:
            missing_categories.append("performance")
        if not goals_data:
            missing_categories.append("goals")
        if not skills_data:
            missing_categories.append("skills")
        if not task_outcomes_data:
            missing_categories.append("task_outcomes")
        if not evaluation_themes_data:
            missing_categories.append("evaluation_themes")

        # Sufficient data means at least the essential categories are present to avoid hallucinations
        has_sufficient_data = len(missing_categories) == 0

        return {
            "has_sufficient_data": has_sufficient_data,
            "missing_categories": missing_categories,
            "context": context,
        }

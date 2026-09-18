"""Policy Context Builder service for the AI HR Policy Assistant.

Retrieves and prepares sanitized, approved company policies and permitted
employee facts for policy question answering. Ensures strict employee data
isolation, approved-data enforcement, and deterministic relevance filtering.
"""

import re
from typing import Any

from sqlalchemy.orm import Session

from app.models import CompanyPolicy, Employee

# Context Limits
MAX_MATCHED_POLICIES = 3
MAX_POLICY_CONTENT_CHARS = 1500
MAX_POLICY_SUMMARY_CHARS = 300
MAX_FIELD_CHARS = 100

# Stopwords for deterministic lexical relevance matching
STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "at",
    "by", "from", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "of", "up", "down", "in",
    "out", "on", "off", "over", "under", "again", "further", "once", "here",
    "there", "all", "any", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "can", "will", "just", "should", "now", "what", "which", "who", "whom", "this",
    "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "how", "why",
    "where", "tell", "me", "our", "my", "your", "policy", "company", "rules",
    "rule", "guidelines", "guideline", "employee", "employees", "much", "many",
}


def _clean_str(val: Any, max_chars: int = MAX_FIELD_CHARS) -> str:
    """Sanitizes and truncates untrusted text fields."""
    if val is None:
        return ""
    text = str(val).strip()
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text


def _extract_keywords(text: str) -> set[str]:
    """Extracts alphanumeric keywords of length >= 3 excluding stop words."""
    if not text:
        return set()
    words = re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", text.lower())
    return {w for w in words if w not in STOP_WORDS}


class PolicyContextBuilder:
    """Gathers and prepares sanitized policy context and permitted employee facts."""

    @staticmethod
    def build_context(
        db: Session,
        employee_id: str,
        question: str,
        category: str | None = None,
    ) -> dict[str, Any]:
        """Builds a grounded, isolated, and budget-capped policy context.

        Args:
            db: SQLAlchemy database session.
            employee_id: Identifier of the employee asking the question.
            question: Employee's natural language policy inquiry.
            category: Optional category filter.

        Returns:
            Dictionary containing matched policies, permitted employee facts,
            grounding metadata, and sufficiency/unsupported status.
        """
        # 1. Retrieve permitted employee facts (Strict Employee Isolation)
        employee = db.query(Employee).filter(Employee.id == employee_id).first()
        if not employee:
            return {
                "has_matching_policies": False,
                "unsupported_reason": f"Employee with id '{employee_id}' was not found in the system.",
                "employee_id": employee_id,
                "employee_found": False,
                "employee_facts": {},
                "matched_policies": [],
                "approved_policy_sources": {},
                "approved_policy_codes": {},
                "total_policies_matched": 0,
            }

        employee_facts = {
            "employee_id": employee.id,
            "first_name": _clean_str(employee.first_name, 100),
            "last_name": _clean_str(employee.last_name, 100),
            "role_title": _clean_str(employee.role_title, 100),
            "department": _clean_str(employee.department, 100),
        }

        # 2. Query ONLY active AND approved policies (no category exclusion)
        candidate_policies = (
            db.query(CompanyPolicy)
            .filter(
                CompanyPolicy.is_active.is_(True),
                CompanyPolicy.is_approved.is_(True),
            )
            .all()
        )

        if not candidate_policies:
            return {
                "has_matching_policies": False,
                "unsupported_reason": "No approved active policies exist in the system.",
                "employee_id": employee.id,
                "employee_found": True,
                "employee_facts": employee_facts,
                "matched_policies": [],
                "approved_policy_sources": {},
                "approved_policy_codes": {},
                "total_policies_matched": 0,
            }

        # 3. Deterministic relevance scoring
        question_keywords = _extract_keywords(question)
        scored_candidates: list[tuple[int, CompanyPolicy]] = []

        for p in candidate_policies:
            score = 0
            title_words = _extract_keywords(p.title)
            category_words = _extract_keywords(p.category)
            summary_words = _extract_keywords(p.summary or "")
            content_words = _extract_keywords(p.content or "")

            # Exact keyword overlaps
            score += len(question_keywords & title_words) * 5
            score += len(question_keywords & category_words) * 4
            score += len(question_keywords & summary_words) * 2
            score += len(question_keywords & content_words) * 1

            # Substring matching for phrases in question
            q_lower = question.lower()
            if p.title.lower() in q_lower:
                score += 10
            if p.category.lower() in q_lower:
                score += 8
            if p.policy_code.lower() in q_lower:
                score += 15

            # If an explicit category was requested/detected and matches, add relevance boost
            if category and category.strip().lower() in p.category.lower():
                score += 10

            scored_candidates.append((score, p))

        # Filter out candidates with zero relevance score
        relevant_candidates = [(score, p) for score, p in scored_candidates if score > 0]

        if not relevant_candidates:
            return {
                "has_matching_policies": False,
                "unsupported_reason": "No approved company policies address the subject of this inquiry.",
                "employee_id": employee.id,
                "employee_found": True,
                "employee_facts": employee_facts,
                "matched_policies": [],
                "approved_policy_sources": {},
                "approved_policy_codes": {},
                "total_policies_matched": 0,
            }

        # Sort descending by score, then ascending by policy_code for deterministic ordering
        relevant_candidates.sort(key=lambda item: (-item[0], item[1].policy_code))
        top_candidates = relevant_candidates[:MAX_MATCHED_POLICIES]

        # 4. Construct sanitized policy records and grounding mappings
        matched_policies: list[dict[str, Any]] = []
        approved_policy_sources: dict[int, dict[str, Any]] = {}
        approved_policy_codes: dict[str, int] = {}

        for _score, p in top_candidates:
            policy_dict = {
                "id": p.id,
                "policy_code": p.policy_code,
                "title": _clean_str(p.title, 255),
                "category": _clean_str(p.category, 100),
                "version": _clean_str(p.version, 20),
                "summary": _clean_str(p.summary, MAX_POLICY_SUMMARY_CHARS),
                "content": _clean_str(p.content, MAX_POLICY_CONTENT_CHARS),
            }
            matched_policies.append(policy_dict)

            grounding_meta = {
                "policy_id": p.id,
                "policy_code": p.policy_code,
                "title": policy_dict["title"],
                "version": policy_dict["version"],
                "category": policy_dict["category"],
                "summary": policy_dict["summary"],
                "content": policy_dict["content"],
            }
            approved_policy_sources[p.id] = grounding_meta
            approved_policy_codes[p.policy_code] = p.id

        return {
            "has_matching_policies": True,
            "unsupported_reason": None,
            "employee_id": employee.id,
            "employee_found": True,
            "employee_facts": employee_facts,
            "matched_policies": matched_policies,
            "approved_policy_sources": approved_policy_sources,
            "approved_policy_codes": approved_policy_codes,
            "total_policies_matched": len(matched_policies),
        }

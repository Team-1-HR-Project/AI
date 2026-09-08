import json
import os
from typing import Optional, Union, Dict, Any
from sqlalchemy.orm import Session
from groq import Groq, APIError, APIConnectionError, RateLimitError
from pydantic import ValidationError

from app.services.career_coach_context import CareerCoachContextBuilder
from app.schemas.career_coach import (
    CareerCoachSuccessResponse,
    CareerCoachInsufficientDataResponse,
    CareerCoachResponse,
)

GROQ_SYSTEM_PROMPT = """You are an expert AI Employee Career Development Coach in a Smart HR Management System.

Your job is to analyze ONLY the employee information provided in the JSON context and generate a structured career development plan.

STRICT RULES YOU MUST FOLLOW:
1. Grounding: Use ONLY the supplied employee context.
2. No Inventions: Never invent employee facts, achievements, weaknesses, skills, goals, or performance metrics.
3. Strengths Evidence: Every strength must be supported by explicit evidence from the supplied context.
4. Development Areas Evidence: Every development area must be supported by explicit evidence from the supplied context.
5. Practical & Measurable: Development actions in the plan must be practical, measurable, and tied to the observed data.
6. Scope: Focus strictly on professional development, skills, goals, task performance, and career growth.
7. ABSOLUTE PROHIBITION ON EMPLOYMENT DECISIONS: You must NOT make, recommend, or suggest employment decisions. Never recommend hiring, firing, promotion, salary changes, compensation, bonuses, disciplinary actions, termination, or employment eligibility.
8. No External Assumptions: Do not assume or extrapolate information that is not explicitly present.
9. Contradictions: If the supplied data does not support a conclusion, do not make that conclusion.
10. Strict JSON Output: Output MUST be a single, valid JSON object strictly conforming to the following structure:
{
    "status": "success",
    "employee_id": "<exact employee id string>",
    "strengths": [
        {
            "title": "string",
            "description": "string",
            "evidence": ["string"]
        }
    ],
    "development_areas": [
        {
            "title": "string",
            "description": "string",
            "evidence": ["string"],
            "priority": "high" | "medium" | "low"
        }
    ],
    "development_plan": [
        {
            "action": "string",
            "reason": "string",
            "measurable_target": "string",
            "suggested_timeline": "string"
        }
    ],
    "follow_up": {
        "checkpoint": "string",
        "review_focus": "string"
    },
    "created_at": "ISO-8601 UTC datetime string"
}
11. No Markdown: Do not wrap the output in markdown backticks (no ```json). Do not add conversational text or explanations outside the JSON object.
"""

class CareerCoachAIServiceError(Exception):
    """Base application-level exception for Career Coach AI Service."""
    pass

class CareerCoachAIService:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        client: Optional[Groq] = None,
    ):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.base_url = base_url or os.getenv("GROQ_BASE_URL", "https://api.groq.com")
        self._client = client

    def _get_client(self) -> Groq:
        if self._client:
            return self._client
        if not self.api_key:
            raise CareerCoachAIServiceError(
                "GROQ_API_KEY is not configured. Please set the GROQ_API_KEY environment variable."
            )
        return Groq(api_key=self.api_key, base_url=self.base_url)

    def generate_career_plan(
        self,
        db: Session,
        employee_id: str,
        period: Optional[str] = None,
    ) -> CareerCoachResponse:
        """
        Generates structured career coach recommendations for the employee.
        - Loads sanitized context using CareerCoachContextBuilder.
        - Returns CareerCoachInsufficientDataResponse if data is incomplete.
        - Calls Groq (llama-3.3-70b-versatile) with sanitized context only.
        - Validates output strictly with Pydantic CareerCoachSuccessResponse.
        """
        # 1. Gather sanitized context
        context_result = CareerCoachContextBuilder.build_context(
            db=db,
            employee_id=employee_id,
            period=period
        )

        # 2. Check for missing data / insufficient data
        if not context_result.get("has_sufficient_data"):
            missing = context_result.get("missing_categories", [])
            return CareerCoachInsufficientDataResponse(
                status="insufficient_data",
                employee_id=employee_id,
                missing_categories=missing if missing else ["unspecified"],
                message="Not enough approved employee data to generate a reliable career coaching plan."
            )

        sanitized_context = context_result["context"]

        # 3. Call Groq API
        client = self._get_client()
        user_prompt = (
            f"Here is the sanitized employee data for employee ID '{employee_id}':\n"
            f"{json.dumps(sanitized_context, indent=2)}\n\n"
            f"Generate the career development plan adhering strictly to the JSON schema."
        )

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": GROQ_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            raw_content = response.choices[0].message.content
        except (RateLimitError, APIConnectionError, APIError) as e:
            # Mask any internal credentials from exception string
            error_type = type(e).__name__
            raise CareerCoachAIServiceError(
                f"Groq API error encountered ({error_type}). Unable to complete Career Coach generation."
            ) from None
        except Exception as e:
            raise CareerCoachAIServiceError(
                f"Unexpected error communicating with Groq: {type(e).__name__}"
            ) from None

        # 4. Parse & Validate with Pydantic
        if not raw_content:
            raise CareerCoachAIServiceError("Groq returned an empty response.")

        clean_content = raw_content.strip()
        # Clean any accidental markdown codeblock fences if present
        if clean_content.startswith("```json"):
            clean_content = clean_content[7:]
        if clean_content.startswith("```"):
            clean_content = clean_content[3:]
        if clean_content.endswith("```"):
            clean_content = clean_content[:-3]
        clean_content = clean_content.strip()

        try:
            parsed_dict = json.loads(clean_content)
        except json.JSONDecodeError as exc:
            raise CareerCoachAIServiceError(
                f"Groq response is not valid JSON: {str(exc)}"
            ) from None

        try:
            validated_response = CareerCoachSuccessResponse.model_validate(parsed_dict)
        except ValidationError as exc:
            raise CareerCoachAIServiceError(
                f"Groq output failed Pydantic schema validation: {str(exc)}"
            ) from None

        return validated_response

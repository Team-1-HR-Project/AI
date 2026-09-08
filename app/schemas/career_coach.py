from datetime import datetime, timezone
from enum import Enum
from typing import List, Literal, Union
from pydantic import BaseModel, Field, ConfigDict

def utc_now():
    return datetime.now(timezone.utc)

class PriorityLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

# 1. Strength Item
class StrengthItem(BaseModel):
    title: str = Field(..., min_length=2, description="Short title of the employee strength")
    description: str = Field(..., min_length=5, description="Clear description of the strength observed")
    evidence: List[str] = Field(..., min_length=1, description="Explicit facts/outcomes supporting the strength")

    model_config = ConfigDict(extra="forbid")

# 2. Development Area Item
class DevelopmentAreaItem(BaseModel):
    title: str = Field(..., min_length=2, description="Focus area requiring development")
    description: str = Field(..., min_length=5, description="Specific gap or growth opportunity")
    evidence: List[str] = Field(..., min_length=1, description="Facts or metrics demonstrating the gap")
    priority: PriorityLevel = Field(..., description="Priority level: high, medium, or low")

    model_config = ConfigDict(extra="forbid")

# 3. Development Plan Item
class DevelopmentPlanAction(BaseModel):
    action: str = Field(..., min_length=3, description="Specific practical action for the employee")
    reason: str = Field(..., min_length=5, description="Why this action is prescribed based on data")
    measurable_target: str = Field(..., min_length=3, description="Concrete milestone or metric to track completion")
    suggested_timeline: str = Field(..., min_length=2, description="Expected completion timeframe (e.g., '30 days')")

    model_config = ConfigDict(extra="forbid")

# 4. Follow-up
class FollowUp(BaseModel):
    checkpoint: str = Field(..., min_length=3, description="Target review date or milestone interval")
    review_focus: str = Field(..., min_length=5, description="Key criteria to evaluate during the check-in")

    model_config = ConfigDict(extra="forbid")

# 5. Successful Career Coach Response
class CareerCoachSuccessResponse(BaseModel):
    status: Literal["success"] = Field("success", description="Status indicator")
    employee_id: str = Field(..., description="Target employee identifier")
    strengths: List[StrengthItem] = Field(..., min_length=1, description="Identified strengths")
    development_areas: List[DevelopmentAreaItem] = Field(..., min_length=1, description="Identified development areas")
    development_plan: List[DevelopmentPlanAction] = Field(..., min_length=1, description="Actionable improvement plan")
    follow_up: FollowUp = Field(..., description="Follow-up milestone and review points")
    created_at: datetime = Field(default_factory=utc_now, description="Timestamp of generation")

    model_config = ConfigDict(extra="forbid")

# 6. Structured Fallback Response for Insufficient Data
class CareerCoachInsufficientDataResponse(BaseModel):
    status: Literal["insufficient_data"] = Field("insufficient_data", description="Status indicator")
    employee_id: str = Field(..., description="Target employee identifier")
    missing_categories: List[str] = Field(..., min_length=1, description="Data categories that were missing or incomplete")
    message: str = Field(
        default="Not enough approved employee data to generate a reliable career coaching plan.",
        description="User-friendly explanation of why guidance could not be generated"
    )
    created_at: datetime = Field(default_factory=utc_now, description="Timestamp of generation")

    model_config = ConfigDict(extra="forbid")

# 7. Discriminated Union for API response handling
CareerCoachResponse = Union[CareerCoachSuccessResponse, CareerCoachInsufficientDataResponse]

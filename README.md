# Smart HR Management System - AI Career Coach Service

Backend service providing personalized, evidence-grounded employee development guidance using FastAPI, SQLAlchemy, and Groq LLMs (`openai/gpt-oss-120b`).

---

## API Contract & Endpoint Documentation

### Generate Career Coach Plan

Generates structured, evidence-based career development guidance for a target employee. Analyzes approved performance records, goals, skills, task outcomes, and evaluation themes.

- **HTTP Method:** `POST`
- **Path:** `/api/career-coach/{employee_id}`
- **Content-Type:** `application/json`

---

### Parameters

#### 1. Path Parameters
| Name | Type | Required | Description | Example |
| :--- | :--- | :--- | :--- | :--- |
| `employee_id` | `string` | **Yes** | Unique identifier of the target employee | `EMP-001` |

#### 2. Query Parameters
| Name | Type | Required | Description | Example |
| :--- | :--- | :--- | :--- | :--- |
| `period` | `string` | No | Target performance/review cycle period (e.g., quarterly) | `2026-Q3` |

---

### Response Schemas

The endpoint returns a discriminated response conforming strictly to Pydantic validation schemas:

#### 1. Success Response (`CareerCoachSuccessResponse`)
Returned when the employee has sufficient approved data across all required categories (`performance`, `goals`, `skills`, `task_outcomes`, `evaluation_themes`).

- **HTTP Status Code:** `200 OK`
- **Fields:**
  - `status` (`string`): `"success"`
  - `employee_id` (`string`): Target employee identifier.
  - `strengths` (`array`): List of identified strengths grounded in evidence.
    - `title` (`string`): Strength title.
    - `description` (`string`): Detailed description of observed capability.
    - `evidence` (`array[string]`): Contextual facts and metrics supporting the strength.
  - `development_areas` (`array`): Prioritized growth opportunities.
    - `title` (`string`): Development area title.
    - `description` (`string`): Specific growth gap.
    - `evidence` (`array[string]`): Contextual metrics or feedback indicating the gap.
    - `priority` (`string`): `"high"` | `"medium"` | `"low"`.
  - `development_plan` (`array`): Practical short-term improvement actions.
    - `action` (`string`): Practical action title.
    - `reason` (`string`): Data-driven justification for the action.
    - `measurable_target` (`string`): Concrete milestone or metric to track completion.
    - `suggested_timeline` (`string`): Timeframe for achievement.
  - `follow_up` (`object`): Suggested review checkpoint.
    - `checkpoint` (`string`): Suggested review interval or date.
    - `review_focus` (`string`): Key criteria to review at the checkpoint.
  - `created_at` (`datetime`): ISO-8601 UTC timestamp.

#### 2. Insufficient Data Response (`CareerCoachInsufficientDataResponse`)
Returned immediately if the employee profile is missing baseline data categories, short-circuiting the AI call to prevent hallucinations and eliminate token costs.

- **HTTP Status Code:** `200 OK`
- **Fields:**
  - `status` (`string`): `"insufficient_data"`
  - `employee_id` (`string`): Target employee identifier.
  - `missing_categories` (`array[string]`): Categories lacking records (e.g. `["performance", "goals"]`).
  - `message` (`string`): Explanation string (`"Not enough approved employee data to generate a reliable career coaching plan."`).
  - `created_at` (`datetime`): ISO-8601 UTC timestamp.

#### 3. Error Responses
- **HTTP Status Code:** `502 Bad Gateway`
  - Triggered when Groq encounters network errors, timeouts, rate limits, or invalid model responses.
  - Response body:
    ```json
    {
      "detail": "Groq API error encountered (RateLimitError). Unable to complete Career Coach generation."
    }
    ```
  - *Safety Guarantee:* API keys, credentials, and internal stack traces are never leaked in error messages.

---

### Request & Response Examples

#### Example 1: Request with Period
```http
POST /api/career-coach/EMP-001?period=2026-Q3 HTTP/1.1
Host: localhost:8000
Content-Type: application/json
```

#### Example 1: Success Response
```json
{
  "status": "success",
  "employee_id": "EMP-001",
  "strengths": [
    {
      "title": "High Performance Delivery",
      "description": "Consistently exceeds performance targets with top scores across key metrics.",
      "evidence": [
        "Overall performance score of 93.5 in Q3 2026",
        "Task completion rate of 96.0%",
        "Goal achievement rate of 91.0%",
        "Attendance rate of 99.0%"
      ]
    },
    {
      "title": "Expertise in Python, FastAPI & Async Architecture",
      "description": "Demonstrates expert-level skill in modern backend technologies.",
      "evidence": [
        "Architected core event-driven API gateway with 99.95% uptime"
      ]
    }
  ],
  "development_areas": [
    {
      "title": "Increase Knowledge Sharing Sessions",
      "description": "Conduct regular knowledge sharing to mentor junior peers.",
      "evidence": [
        "Evaluation theme highlighted opportunity to run more knowledge sharing sessions for junior peers"
      ],
      "priority": "high"
    },
    {
      "title": "Complete Redis Cluster Migration",
      "description": "Finalize migration of distributed caching to Redis Cluster.",
      "evidence": [
        "Goal progress at 85% with deadline 2026-10-30"
      ],
      "priority": "medium"
    }
  ],
  "development_plan": [
    {
      "action": "Schedule and lead monthly knowledge sharing sessions for junior engineers",
      "reason": "Leverage mentorship strengths and address identified opportunity",
      "measurable_target": "Conduct at least 3 sessions by 2026-11-30 with attendance of >=80% of junior team",
      "suggested_timeline": "First session by 2026-09-15, then monthly"
    },
    {
      "action": "Finalize and execute Redis Cluster migration plan",
      "reason": "Complete critical caching migration to meet project deadline",
      "measurable_target": "Achieve 100% migration and validation by 2026-10-30",
      "suggested_timeline": "Complete remaining tasks by 2026-10-15, testing by 2026-10-25, go-live 2026-10-30"
    }
  ],
  "follow_up": {
    "checkpoint": "2026-11-15",
    "review_focus": "Progress on Redis migration completion and effectiveness of knowledge sharing sessions"
  },
  "created_at": "2026-09-07T12:34:56Z"
}
```

#### Example 2: Insufficient-Data Response
```http
POST /api/career-coach/EMP-NEW HTTP/1.1
Host: localhost:8000
Content-Type: application/json
```

```json
{
  "status": "insufficient_data",
  "employee_id": "EMP-NEW",
  "missing_categories": [
    "performance",
    "goals",
    "skills",
    "task_outcomes",
    "evaluation_themes"
  ],
  "message": "Not enough approved employee data to generate a reliable career coaching plan.",
  "created_at": "2026-09-07T12:34:56Z"
}
```

---

## Development & Testing

### Running Tests
All unit and integration tests can be executed via:
```powershell
.venv\Scripts\pytest -v
```

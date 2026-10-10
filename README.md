# Smart HR Management System - AI Career Coach Service

Backend service providing personalized, evidence-grounded employee development guidance using FastAPI, SQLAlchemy, and Groq LLMs (`openai/gpt-oss-120b`).

---

## API Contract & Endpoint Documentation

### Generate Career Coach Plan

Generates structured, evidence-based career development guidance for a target employee. Analyzes approved performance records, goals, skills, task outcomes, and evaluation themes.

- **HTTP Method:** `POST`
- **Path:** `/api/career-coach`
- **Content-Type:** `application/json`

---

### Request Body (`POST /api/career-coach`)

| Field | Type | Required | Description | Example |
| :--- | :--- | :--- | :--- | :--- |
| `employee_id` | `string` | **Yes** | Unique identifier of the target employee | `"EMP-001"` |
| `period` | `string` | No | Target performance/review cycle period | `"2026-Q3"` |

Request payload example:
```json
{
  "employee_id": "EMP-001",
  "period": "2026-Q3"
}
```

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
    - `evidence` (`array[EvidenceItem]`): Grounded claims tied to approved source records:
      - `source_type` (`string`): `"performance"` | `"goal"` | `"skill"` | `"task_outcome"` | `"evaluation_theme"`
      - `source_id` (`integer`): ID of approved source record.
      - `claim` (`string`): Factual, verified claim matching the source.
  - `development_areas` (`array`): Prioritized growth opportunities.
    - `title` (`string`): Development area title.
    - `description` (`string`): Specific growth gap.
    - `evidence` (`array[EvidenceItem]`): Grounded claims tied to approved source records.
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
  - Returns a client-safe response with a unique reference ID for log correlation:
    ```json
    {
      "detail": "AI service temporarily unavailable. Reference ID: 7b845890-410a-4286-bc94-469b76c9ad24"
    }
    ```
  - *Safety Guarantee:* API keys, credentials, and internal stack traces are never leaked in error messages.

---

### Request & Response Examples

#### Example 1: Primary Request Body
```http
POST /api/career-coach HTTP/1.1
Host: localhost:8000
Content-Type: application/json

{
  "employee_id": "EMP-001",
  "period": "2026-Q3"
}
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
        {
          "source_type": "performance",
          "source_id": 1,
          "claim": "Overall performance score of 93.5 in Q3 2026"
        }
      ]
    }
  ],
    {
      "title": "Expertise in Python, FastAPI & Async Architecture",
      "description": "Demonstrates expert-level skill in modern backend technologies.",
      "evidence": [
        {
          "source_type": "skill",
          "source_id": 2,
          "claim": "Expert Python and FastAPI architectural skills"
        }
      ]
    }
  ],
  "development_areas": [
    {
      "title": "Increase Knowledge Sharing Sessions",
      "description": "Conduct regular knowledge sharing to mentor junior peers.",
      "evidence": [
        {
          "source_type": "evaluation_theme",
          "source_id": 1,
          "claim": "Evaluation theme highlighted opportunity to mentor junior peers"
        }
      ],
      "priority": "high"
    },
    {
      "title": "Complete Redis Cluster Migration",
      "description": "Finalize migration of distributed caching to Redis Cluster.",
      "evidence": [
        {
          "source_type": "goal",
          "source_id": 1,
          "claim": "Goal progress at 85% with deadline 2026-10-30"
        }
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

#### Example 2: Insufficient-Data Request & Response
```http
POST /api/career-coach HTTP/1.1
Host: localhost:8000
Content-Type: application/json

{
  "employee_id": "EMP-NEW"
}
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

## AI HR Policy Assistant API

### Ask Policy Assistant

Answers employee HR policy inquiries strictly based on approved active company policies and permitted employee profile facts. Supports persistent multi-turn conversations through conversation/session IDs.

- **HTTP Method:** `POST`
- **Path:** `/api/policy-assistant`
- **Content-Type:** `application/json`

### Request Body (`POST /api/policy-assistant`)

| Field | Type | Required | Description | Example |
| :--- | :--- | :--- | :--- | :--- |
| `employee_id` | `string` | **Yes** | Unique identifier of the target employee | `"EMP-001"` |
| `question` | `string` | **Yes** | Policy question to be answered | `"What is the annual leave rollover limit?"` |
| `session_id` | `string` | No | Optional existing chat session ID. If omitted, a new persistent session is created. | `"8f3b2a4c-5678-4321-9876-abcdef012345"` |

#### Example 1: Starting a New Session (Omit `session_id`)
```http
POST /api/policy-assistant HTTP/1.1
Host: localhost:8000
Content-Type: application/json

{
  "employee_id": "EMP-SEC-ALICE",
  "question": "What is the annual leave rollover limit?"
}
```

**Success Response (`200 OK`):**
```json
{
  "status": "success",
  "session_id": "8f3b2a4c-5678-4321-9876-abcdef012345",
  "employee_id": "EMP-SEC-ALICE",
  "answer": "Employees may carry forward up to five (5) unused annual leave days into the next calendar year.",
  "policy_references": [
    {
      "policy_id": 1,
      "policy_code": "POL-LEAVE-001",
      "title": "Annual Leave & Time Off Policy",
      "version": "1.0"
    }
  ],
  "employee_facts_used": [],
  "created_at": "2026-09-09T09:50:31Z"
}
```

#### Example 2: Continuing an Existing Conversation (Provide `session_id`)
```http
POST /api/policy-assistant HTTP/1.1
Host: localhost:8000
Content-Type: application/json

{
  "employee_id": "EMP-SEC-ALICE",
  "question": "What happens if I don't use them within the rollover period?",
  "session_id": "8f3b2a4c-5678-4321-9876-abcdef012345"
}
```

### Policy Assistant Error Handling
- **`404 Not Found`**: Returned when the requested `session_id` does not exist.
  ```json
  { "detail": "Chat session not found." }
  ```
- **`403 Forbidden`**: Returned when an employee attempts to access a session belonging to another employee (strict tenant and identity isolation).
  ```json
  { "detail": "Access denied: session belongs to another employee." }
  ```
- **`502 Bad Gateway`**: Returned on transient or upstream AI provider errors without exposing internal database logs.

### Hybrid Memory Architecture & Token Optimization
The Policy Assistant incorporates a production-grade **Hybrid Memory Architecture** designed for bounded token usage, dialogue continuity, and long-term recall:
- **Short-Term Memory (Budget Window)**: Replaces fixed message counts with a configurable character/token budget (`recent_messages_char_budget = 2500` chars). Greedily includes recent turns in `<RECENT_CONVERSATION_HISTORY>`.
- **Rolling Conversation Summary**: When conversation history exceeds the budget, older turns are distilled into a compact summary stored in `ChatSession.summary` and injected as `<CONVERSATION_SUMMARY>`. Summaries update in batches rather than on every turn, with non-destructive fallback if summarization fails.
- **Semantic Memory / Older Turn Retrieval**: Semantic retrieval uses subword vector embeddings and cosine similarity to retrieve relevant turns from older conversations (`<RELEVANT_CONVERSATION_MEMORIES>`) when an employee refers to topics discussed much earlier. Filtered by `similarity_threshold = 0.35` and bounded by `retrieved_memory_char_budget = 1500` chars.
- **Strict Policy Authority**: Approved company policies (`<COMPANY_POLICIES>`) are the authoritative source of truth and **strictly supersede** any contradictory retrieved memory. Grounding verification ensures answers must cite active approved policies and cannot cite past conversation as policy evidence.
- **Prompt-Injection Defense**: Delimiters in memories, summaries, and queries are neutralized through sanitization escaping.

---

## AI Evaluation Draft Assistant API

### Generate Evaluation Draft

Synthesizes approved employee records and optional manager observations into a structured, evidence-grounded performance evaluation draft.

- **HTTP Method:** `POST`
- **Path:** `/api/evaluation-draft`
- **Content-Type:** `application/json`

### Request Body (`POST /api/evaluation-draft`)

| Field | Type | Required | Description | Example |
| :--- | :--- | :--- | :--- | :--- |
| `employee_id` | `string` | **Yes** | Unique identifier of the target employee | `"EMP-001"` |
| `period` | `string` | **Yes** | Evaluation review cycle period | `"2026-Q3"` |
| `evaluation_scores` | `object` | No | Optional dictionary of numeric metric scores (0.0 to 100.0) | `{"overall": 92.0}` |
| `manager_notes` | `string` | No | Optional manager feedback or observations to integrate | `"Strong leadership on platform migration."` |

Request payload example:
```json
{
  "employee_id": "EMP-001",
  "period": "2026-Q3",
  "evaluation_scores": {
    "overall": 92.0,
    "leadership": 88.0
  },
  "manager_notes": "Demonstrated exceptional technical leadership during the platform migration."
}
```

### Success Response (`200 OK`)
```json
{
  "status": "success",
  "employee_id": "EMP-001",
  "period": "2026-Q3",
  "evaluation_narrative": "Alice delivered exemplary technical performance during Q3 2026, achieving a 94.0 overall score and successfully guiding the core cutover without service disruption.",
  "strengths": [
    {
      "title": "High Delivery Quality",
      "description": "Consistently delivered robust systems with zero errors during production migration.",
      "evidence": [
        {
          "source_type": "performance",
          "source_id": 1,
          "claim": "Achieved overall score of 94.0 and 97.0% task completion in Q3 2026"
        }
      ]
    }
  ],
  "improvement_areas": [
    {
      "title": "Knowledge Sharing",
      "description": "Conduct regular architecture walkthroughs for junior peers.",
      "evidence": [
        {
          "source_type": "evaluation_theme",
          "source_id": 1,
          "claim": "Feedback highlighted opportunity to run more knowledge sharing sessions"
        }
      ],
      "priority": "medium"
    }
  ],
  "entered_scores": {
    "overall": 92.0,
    "leadership": 88.0
  },
  "human_review_required": true,
  "review_disclaimer": "This evaluation is an AI-generated draft intended solely to assist manager review. A human manager must review, edit, and approve this evaluation before any official use or persistence.",
  "created_at": "2026-09-14T11:00:00Z"
}
```

### Governance & Safety Controls
- **Stateless Draft Guarantee**: The service is strictly stateless; it never writes or auto-submits records to the database. Human manager review and approval is mandatory before any future persistence.
- **Prohibited Decisions**: Deterministic filters block the AI from recommending or executing promotions, demotions, salary adjustments, bonuses, termination, or disciplinary actions.
- **Approved Evidence Only**: Only records with `is_approved == True` are included in the AI context. Unapproved drafts and cross-employee records are excluded.
- **Fail-Closed Insufficient Data**: If an employee lacks baseline data, an `insufficient_data` response is returned immediately with zero LLM token cost.

---

## Development & Testing

### Current AI Routes and Authorization

The service currently exposes seven AI routes: Career Coach, HR Policy Assistant, Performance Insight, Evaluation Draft, Skill Gap, Attention Signal, and Team Insight. All AI routes require the trusted gateway headers `X-Caller-Employee-ID` and `X-Caller-Role`; employee ownership, manager department scope, HR-admin organization scope, and policy chat-session ownership are enforced server-side.

Durable insight history and feedback are available under `/api/insights` through the history, snapshot, regeneration, and feedback endpoints. Only approved HR records and explicitly active/approved policies are eligible for AI context.

### Running Tests

### Running Tests
All unit and integration tests can be executed via:
```powershell
python -m pytest -v
```

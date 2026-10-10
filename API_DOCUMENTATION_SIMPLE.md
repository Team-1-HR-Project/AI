# AI Career Coach API

## Primary Endpoint
POST /api/career-coach

## Input Data Type
- employee_id: string
- period: string | null

## Input Shape
```json
{
  "employee_id": "EMP-SEC-ALICE",
  "period": "2026-Q3"
}
```

---

# AI Attention Signal API

## Primary Endpoint
`POST /api/attention-signal`

Analyzes approved employee performance/activity context and returns an advisory attention level. The response is not an employment decision and includes human-review safeguards.

## Request

```json
{
  "employee_id": "EMP-SEC-ALICE",
  "period": "2026-Q3"
}
```

## Access

AI routes require gateway-provided `X-Caller-Employee-ID` and `X-Caller-Role` headers. Valid roles are `employee`, `manager`, and `hr_admin`; callers are limited to their authorized employee, department, or session scope.

---

# AI Team Insight API

## Primary Endpoint
`POST /api/team-insight`

Generates an aggregate, anonymized insight for an authorized department using approved team records. Managers are restricted to their own department; HR administrators may request any department.

## Request

```json
{
  "department": "Engineering",
  "period": "2026-Q3"
}
```

---

# AI Insight History, Regeneration, and Feedback

Durable AI snapshots are stored with feature, scope, period, model, generation ID, version, and actor context.

- `GET /api/insights/history`
- `GET /api/insights/{snapshot_id}`
- `POST /api/insights/{snapshot_id}/regenerate`
- `POST /api/insights/{snapshot_id}/feedback`
- `GET /api/insights/{snapshot_id}/feedback`

All insight and feedback operations enforce the same caller authorization rules as generation endpoints. Provider failures return a safe reference ID and do not expose prompts, credentials, or stack traces.

## Error Handling
Returns `HTTP 502 Bad Gateway` on AI provider or service failures with a safe reference ID:
```json
{
  "detail": "AI service temporarily unavailable. Reference ID: 7b845890-410a-4286-bc94-469b76c9ad24"
}
```

## Output Shape
```json
{
  "status": "success",
  "employee_id": "EMP-SEC-ALICE",
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
  "development_areas": [
    {
      "title": "Increase Knowledge Sharing Sessions",
      "description": "Conduct regular knowledge sharing to mentor junior peers.",
      "evidence": [
        {
          "source_type": "evaluation_theme",
          "source_id": 1,
          "claim": "Feedback highlighted opportunity to mentor junior peers"
        }
      ],
      "priority": "high"
    }
  ],
  "development_plan": [
    {
      "action": "Schedule monthly knowledge sharing sessions",
      "reason": "Leverage mentorship strengths and address identified opportunity",
      "measurable_target": "Conduct at least 3 sessions by end of quarter",
      "suggested_timeline": "First session within 2 weeks"
    }
  ],
  "follow_up": {
    "checkpoint": "2026-11-15",
    "review_focus": "Review effectiveness and attendance of knowledge sharing sessions"
  },
  "created_at": "2026-09-08T18:30:00Z"
}
```

## Insufficient Data Output
```json
{
  "status": "insufficient_data",
  "employee_id": "EMP-SEC-ALICE",
  "missing_categories": [
    "performance",
    "goals"
  ],
  "message": "Not enough approved employee data to generate a reliable career coaching plan.",
  "created_at": "2026-09-08T18:30:00Z"
}
```

---

# AI HR Policy Assistant API

## Primary Endpoint
POST /api/policy-assistant

## Input Data Type
- employee_id: string
- question: string
- session_id: string | null

## Input Shape
```json
{
  "employee_id": "EMP-SEC-ALICE",
  "question": "What is the annual leave rollover limit?",
  "session_id": "8f3b2a4c-5678-4321-9876-abcdef012345"
}
```

## Error Handling
- Returns `HTTP 404 Not Found` if the session does not exist.
- Returns `HTTP 403 Forbidden` if the session belongs to a different employee.
- Returns `HTTP 502 Bad Gateway` on AI provider or service failures with a safe reference ID:
```json
{
  "detail": "AI service temporarily unavailable. Reference ID: 7b845890-410a-4286-bc94-469b76c9ad24"
}
```

## Output Shape
```json
{
  "status": "success",
  "session_id": "8f3b2a4c-5678-4321-9876-abcdef012345",
  "answer": "According to the Annual Leave Policy (POL-HR-001), employees may roll over up to 5 unused annual leave days into the next calendar year.",
  "citations": [
    {
      "policy_code": "POL-HR-001",
      "policy_title": "Annual Leave Policy",
      "section": "Leave Rollover Rules"
    }
  ],
  "suggested_follow_up": [
    "How do I submit a rollover request?"
  ],
  "created_at": "2026-09-08T18:35:00Z"
}
```

## Insufficient Data / Unsupported Output
```json
{
  "status": "unsupported",
  "session_id": "8f3b2a4c-5678-4321-9876-abcdef012345",
  "answer": "I could not find an approved company policy that covers this question. Please contact HR directly.",
  "citations": [],
  "suggested_follow_up": [],
  "created_at": "2026-09-08T18:35:00Z"
}
```

---

# AI Performance Insight Generator API

## Primary Endpoint
POST /api/performance-insight

## Input Data Type
- employee_id: string
- period: string | null

## Input Shape
```json
{
  "employee_id": "EMP-SEC-ALICE",
  "period": "2026-Q2"
}
```

## Error Handling
Returns `HTTP 502 Bad Gateway` on AI provider or service failures with a safe reference ID:
```json
{
  "detail": "AI service temporarily unavailable. Reference ID: 7b845890-410a-4286-bc94-469b76c9ad24"
}
```

## Output Shape
```json
{
  "status": "success",
  "employee_id": "EMP-SEC-ALICE",
  "verified_facts": {
    "target_period": "2026-Q2",
    "comparison_period": "2026-Q1",
    "metrics_by_period": [
      {
        "period": "2026-Q1",
        "overall_score": 82.0,
        "task_completion_rate": 85.0,
        "goal_achievement_rate": 80.0,
        "attendance_rate": 96.0
      },
      {
        "period": "2026-Q2",
        "overall_score": 88.0,
        "task_completion_rate": 92.0,
        "goal_achievement_rate": 86.0,
        "attendance_rate": 90.0
      }
    ]
  },
  "calculated_trends": {
    "from_period": "2026-Q1",
    "to_period": "2026-Q2",
    "metrics": {
      "overall_score": {
        "metric_name": "overall_score",
        "previous_value": 82.0,
        "current_value": 88.0,
        "delta": 6.0,
        "direction": "improved",
        "percent_change": 7.32
      },
      "task_completion_rate": {
        "metric_name": "task_completion_rate",
        "previous_value": 85.0,
        "current_value": 92.0,
        "delta": 7.0,
        "direction": "improved",
        "percent_change": 8.24
      }
    }
  },
  "ai_interpretation": {
    "summary": "Alice demonstrated consistent performance improvements from Q1 2026 to Q2 2026, driven by increases in task completion and goal achievement.",
    "improvements": [
      {
        "metric": "task_completion_rate",
        "summary": "Task completion increased from 85.0% to 92.0%.",
        "contributing_indicators": [
          {
            "indicator_name": "Sprint Deliverables",
            "category": "tasks",
            "observation": "Consistently met sprint deadlines."
          }
        ]
      }
    ],
    "declines": [],
    "suggested_review_actions": [
      {
        "priority": "medium",
        "focus_area": "Knowledge Sharing",
        "recommended_action": "Conduct monthly peer walkthroughs.",
        "rationale": "Helps scale high delivery practices across the team."
      }
    ]
  },
  "created_at": "2026-09-08T18:40:00Z"
}
```

## Insufficient Data Output
```json
{
  "status": "insufficient_data",
  "employee_id": "EMP-SEC-ALICE",
  "reason": "At least 2 approved performance periods are required for cross-period trend analysis.",
  "target_period": "2026-Q1",
  "comparison_period": null,
  "facts": {
    "records_count": 1,
    "periods": ["2026-Q1"],
    "metrics_by_period": [
      {
        "period": "2026-Q1",
        "overall_score": 82.0,
        "task_completion_rate": 85.0,
        "goal_achievement_rate": 80.0,
        "attendance_rate": 96.0
      }
    ]
  },
  "created_at": "2026-09-08T18:40:00Z"
}
```

---

# AI Evaluation Draft Assistant API

## Primary Endpoint
POST /api/evaluation-draft

## Input Data Type
- employee_id: string
- period: string
- evaluation_scores: object | null
- manager_notes: string | null

## Input Shape
```json
{
  "employee_id": "EMP-SEC-ALICE",
  "period": "2026-Q3",
  "evaluation_scores": {
    "overall": 92.0,
    "leadership": 88.0
  },
  "manager_notes": "Demonstrated exceptional technical leadership during the platform migration."
}
```

## Error Handling
Returns `HTTP 502 Bad Gateway` on AI provider or service failures with a safe reference ID:
```json
{
  "detail": "AI service temporarily unavailable. Reference ID: 7b845890-410a-4286-bc94-469b76c9ad24"
}
```

## Output Shape
```json
{
  "status": "success",
  "employee_id": "EMP-SEC-ALICE",
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
      "description": "Conduct monthly architecture review sessions for junior engineers.",
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

## Insufficient Data Output
```json
{
  "status": "insufficient_data",
  "employee_id": "EMP-EMPTY",
  "period": "2026-Q3",
  "missing_categories": [
    "performance",
    "goals",
    "skills",
    "task_outcomes",
    "evaluation_themes"
  ],
  "message": "Not enough approved employee data to generate a reliable evaluation draft.",
  "human_review_required": false,
  "created_at": "2026-09-14T11:00:00Z"
}
```

---

# Skill-Gap & Development Recommendations API

## Primary Endpoint
POST /api/skill-gap

## Input Data Type
- employee_id: string
- period: string | null
- target_role: string | null
- target_skills: list of strings | null

## Input Shape
```json
{
  "employee_id": "EMP-ALICE",
  "period": "2026-Q3",
  "target_role": "Senior Backend Architect",
  "target_skills": [
    "Distributed Systems",
    "Event Sourcing"
  ]
}
```

## Error Handling
Returns `HTTP 502 Bad Gateway` on AI provider or service failures with a safe reference ID:
```json
{
  "detail": "AI service temporarily unavailable. Reference ID: 7b845890-410a-4286-bc94-469b76c9ad24"
}
```

## Output Shape
```json
{
  "status": "success",
  "employee_id": "EMP-ALICE",
  "period": "2026-Q3",
  "target_role": "Senior Backend Architect",
  "skill_gaps": [
    {
      "skill_name": "Distributed Systems",
      "current_level": "Intermediate",
      "desired_level": "Advanced",
      "gap_severity": "high",
      "rationale": "Employee demonstrated strong Python delivery, but Q3 evaluation highlighted a need for deeper expertise in distributed transactions.",
      "evidence": [
        {
          "source_type": "skill",
          "source_id": 1,
          "claim": "Observed Python Backend skill is Intermediate"
        },
        {
          "source_type": "evaluation_theme",
          "source_id": 1,
          "claim": "Q3 evaluation feedback highlighted opportunity in distributed transaction knowledge"
        }
      ]
    }
  ],
  "recommendations": [
    {
      "title": "Advanced Distributed Systems Masterclass",
      "learning_type": "training_course",
      "focus_skill": "Distributed Systems",
      "description": "Complete coursework on event-driven architecture and saga patterns.",
      "expected_outcome": "Ability to architect robust distributed transactions.",
      "measurable_target": "Complete coursework and deliver a working saga orchestration prototype by week 6",
      "timeline": "6 weeks",
      "priority": "high"
    }
  ],
  "disclaimer": "This skill-gap analysis and development plan is AI-generated and advisory. It does not constitute a formal performance appraisal or employment decision.",
  "created_at": "2026-09-18T04:30:00Z"
}
```

## Insufficient Data Output
```json
{
  "status": "insufficient_data",
  "employee_id": "EMP-EMPTY",
  "missing_categories": [
    "skills",
    "performance"
  ],
  "message": "Not enough approved employee data to generate a reliable skill-gap analysis.",
  "created_at": "2026-09-18T04:30:00Z"
}
```

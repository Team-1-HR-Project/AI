### AI Scope Boundary

This report is limited to the AI team's deliverables: AI API routes, LLM/provider integration, prompts, grounding and safety validation, AI context construction, memory, AI-related persistence, configuration, AI tests, and AI documentation.

Upstream HR data producers, non-AI backend CRUD, and frontend/mobile clients are outside this report's scope and are not scored as AI defects. They are treated only as external inputs/contracts when the AI service depends on approved data.

### 7.4 Audit coverage and test limitations

The AI repository inventory contains 68 Python files, 8 Markdown documents, the PDF scope file, environment/configuration files, and the migration/benchmark scripts. The AI runtime modules were imported by the test suite or inspected through the route/service/model/context inventory; all 441 tests were executed after Mem0 installation.

There is an important test-environment limitation: most AI unit/API/context tests create their own in-memory SQLite engine. The explicit MySQL tests do exercise the XAMPP database, but the green full-suite result does not mean every AI test ran against MariaDB. In addition, tests/test_mysql_connection.py and the autouse setup use SELECT 1 as their availability check without verifying engine.dialect.name; when the configured database is SQLite, those tests can run against SQLite and still be labeled as MySQL tests. A production CI job should run a separate, explicit MySQL integration stage for the AI service.

The AI documentation inventory also found drift:

- README.md, APIs.md, and FINAL_TEST_REPORT.md primarily document the original four AI features and omit the current Attention Signal and Team Insight routes.
- API_DOCUMENTATION_SIMPLE.md mentions Skill Gap but does not document all seven current AI routes.
- The older final audit reports state that Attention Signal and Team Insight are not implemented, which is false for the current commit.
- This report supersedes those AI documents for the audited commit.

## 8. Confirmed AI bugs and risks

### P0 - AI endpoint caller authorization gap

Every AI route accepts an employee_id or department directly from the request. There is no authenticated AI caller context, role check, manager-to-department relationship, employee ownership check, or session ownership check inside this service. Context isolation prevents accidental cross-record mixing inside a request, but it does not stop a caller from requesting another employee's AI output.

### P0 - False-positive AI health/readiness status

AI application startup catches schema/database initialization errors and continues. /health returns 200 with status=ok regardless of whether the AI service database is usable. This was reproduced with an invalid SQLite path: schema initialization logged an operational error while /health still returned 200.

### P1 - Skill Gap numeric-grounding false rejection

The Skill Gap validator extracts numeric values from free-form evidence text. A legitimate source reference such as skill id 1 was treated as a business numeric claim and rejected. The same AI feature succeeded when the model produced a simpler response.

### P1 - Team Insight period-label false rejection

The Team Insight validator interpreted the 2 in the period label Q2 as a numeric claim requiring a matching business value. The result was a 502 even though the narrative was otherwise grounded. Single-period output succeeded.

### P1 - Evaluation Draft semantic-grounding gap

A direct reproduction passed the current validator with this fabricated evidence claim:

~~~text
Won an international award and led a global initiative.
~~~

The claim referenced an existing performance record ID, but the source record did not contain that fact. The validator checked source identity and numeric compatibility but did not require semantic overlap between the claim and source fields. This is a material grounding risk for an AI-generated evaluation draft.

### P1 - Team Insight fallback can label skills as gaps

app/services/team_insight_ai.py contains a fallback that assigns top_common_gaps from top_common_skills when the model returns no gaps. An existing skill inventory must not automatically become an AI-identified gap.

### P1 - Policy knowledge approval defaults are permissive

New CompanyPolicy records default to active/approved values. The AI knowledge-source write path must require explicit review and approval before a policy can be used as authoritative context.

### P2 - Evaluation provider configuration inconsistency

The Evaluation Draft client construction does not consistently pass the configured GROQ_BASE_URL and does not validate an empty model in the same way as other AI services. Provider configuration should be centralized.

### P2 - AI data migration script can hide context problems

scripts/migrate_to_mysql.py silently ignores a sqlite3.OperationalError around policy migration. A missing or incompatible AI policy table can therefore look like a successful migration. The same script falls back to the current timestamp for malformed dates and interpolates the configured database name directly into a CREATE DATABASE statement. These paths should fail loudly, validate identifiers, and produce an AI data-migration summary with an explicit failure status.

### P2 - Broad exception handling hides AI defects

The Team Insight service includes an exception pattern equivalent to except (json.JSONDecodeError, Exception), which catches everything and can mask programming errors as normal model failures.

### P2 - Mem0 is a non-persistent AI memory POC

The Mem0 manager uses Qdrant and history paths of :memory:. The package is installed and all tests pass, but memories disappear when the AI process restarts. The Policy Assistant default remains the custom hybrid memory manager, so Mem0 is not currently the production memory backend.

## 9. AI hardcoded values and configuration inventory

| Area | Hardcoded/current value | AI risk or recommendation |
|---|---|---|
| Model | openai/gpt-oss-120b repeated across AI services | Centralize and validate model configuration. |
| Provider | https://api.groq.com repeated/defaulted | Centralize AI provider client construction. |
| Provider quota | Groq TPM limit observed at 8,000 tokens/minute | Add quota-aware backoff, token budgeting, queueing, and monitoring. |
| Timeouts | 30-second client timeout; 25-second service deadline | Move AI latency settings to environment/config and expose metrics. |
| Retries | Initial call plus up to two retries | Configure by provider error type and rate-limit budget. |
| Generation | Per-feature temperatures and max tokens | Centralize AI generation policy and version it. |
| Prompt/context budgets | Approximately 12,000-character prompt caps and feature-specific record caps | Make AI budgets visible and configurable. |
| Policy retrieval | Small matched-policy cap, content/summary truncation, lexical weights/stopwords | Review against the real policy corpus and multilingual AI questions. |
| Safety rules | Repeated regex/prohibited-decision lists | Centralize and test versioned AI safety rules. |
| Attention thresholds | Good >=85, moderate >=70, low <70 | Make AI signal thresholds configurable and approved. |
| AI database fallback | root, blank password, localhost, port 3306, hr_system | Never use this as a production AI-service default. |
| AI server | 127.0.0.1:8000 and reload=True in run.py | Development-only behavior; use deployment configuration. |
| Mem0 storage | Qdrant :memory: and history :memory: | AI memory is lost on restart; use persistent storage. |
| AI demo data | EMP-MANUAL-TEST, EMP-PERF-DEMO, 2026-Q2/Q3, fixed metrics | Clearly separate AI fixtures from real approved context. |
| Example configuration | .env.example still uses database port 3306 while the active XAMPP setup uses 3308 | Keep the AI environment example aligned with the deployment profile. |
| Benchmark quota | scripts/benchmark_memory_tokens.py treats 8,000 TPM as a provider-risk boundary | Keep AI quota assumptions configurable and current. |

## 10. AI feature-by-feature evaluation

### Rating scale

The score is an AI engineering-readiness score for this repository, not a claim that the complete product around the AI service is production-ready:

- 9-10: production-ready for the declared AI scope.
- 7-8: strong AI backend/demo implementation with known hardening work.
- 4-6: partially implemented or unreliable under important AI scenarios.
- 1-3: proof of concept only.
- 0: not implemented in the AI scope.

The score considers AI implementation completeness, real LLM behavior, grounding/safety, caller context, persistence, observability, and alignment with the PDF AI requirements.

| AI feature | Score | Current AI evaluation | What can be added or changed | AI work still unfinished |
|---|---:|---|---|---|
| Career Coach | 8/10 | AI route and context are implemented. A real live response returned strengths, development areas, action plan, evidence, and follow-up. | Add an explicit top-level focus field required by the PDF; add caller context authorization; add semantic evidence checks; add durable AI insight history if required. | No AI history/regeneration, no AI feedback capture, and no caller-aware permission enforcement. |
| HR Policy Assistant | 8/10 | Strong grounded-answer path. The live response cited POL-LEAVE-001. Custom hybrid memory tests and Mem0 tests pass. | Require explicit policy approval; make model/grounding fallback consistently safe; add session ownership; make memory selection configurable and persistent when enabled. | No AI feedback/history endpoints, no caller/session authorization, and Mem0 is not persistent. |
| Performance Insight | 8/10 | Deterministic facts and trends are separated from AI interpretation. A real Q2-to-Q3 response returned correct deltas and review actions. | Move AI thresholds/configuration out of code; add caller scope; add AI history/audit; formalize the approved context contract for every metric. | No durable AI insight lifecycle, no AI feedback loop, and no caller permission boundary. |
| Evaluation Draft Assistant | 6/10 | Real generation works and human_review_required=true is enforced. A fabricated semantic evidence claim was nevertheless accepted when it reused a valid source ID. | Implement field-aware and semantic claim-to-source validation; pass GROQ_BASE_URL consistently; add generated/edited/approved/rejected AI draft states; persist only after human approval; add actor/audit context. | Semantic grounding is not safe enough; no AI draft lifecycle/history; no feedback loop. |
| Skill Gap / Development Recommendations | 7/10 | Route and AI context are implemented. A minimal real request returned 200, but a realistic multi-target-skill request returned 502 because of numeric grounding false rejection. | Ignore IDs, dates, percentages inside identifiers, and period labels during numeric extraction; validate target skills against structured context; test unknown skills and useful recommendations. | Realistic multi-skill output is not robust; no persistent AI development plan; no feedback/history. |
| Employee Attention Signal | 7/10 | A real response returned Low, own-history comparison, indicators, supportive follow-up, advisory_only=true, and human_review_required=true. | Replace hardcoded AI thresholds with approved configuration; define the approved input-context contract for attendance/workload indicators; minimize PII in prompts; add signal history/audit. | No durable signal history, no feedback, and no caller-aware permission enforcement. |
| Team Insight Summary | 6/10 | Route/context are implemented and a single-period live request succeeded. A realistic Q2/Q3 request returned 502 because Q2 was parsed as number 2. The AI test fixture had only one team member. | Make grounding field-aware for period tokens; separate skill inventory from skill gaps; use multi-member approved-context fixtures; enforce manager-scope context; add aggregate history and quota handling. | Multi-period robustness, meaningful aggregate test coverage, AI history/feedback, and caller scope are unfinished. |
| Custom Hybrid Memory | 8/10 | Bounded custom memory is the Policy Assistant default and is covered by extensive tests. | Add AI memory metrics, configurable budgets, retention/deletion policy, and employee/session authorization. | No operational memory dashboard, explicit retention workflow, or independent production memory service. |
| Mem0 Memory POC | 5/10 | Installed and tested successfully: 10 focused tests and all 441 tests pass. Local Qdrant initializes. | Use persistent access-controlled storage; expose backend choice in AI configuration; define retention/deletion; monitor vector-store failures. | It is still an in-memory POC and is not the default production AI backend. |
| Insight History and Regeneration | 0/10 | No AI model, route, or durable snapshot store was found. | Add AI insight snapshots with source period, model, prompt hash, status, actor context, and regeneration endpoint. | Entire AI feature is not implemented. |
| AI Feedback Capture | 0/10 | No AI feedback model or endpoint was found. | Add rating/reason/comment, actor context, feature/version, linked insight ID, privacy controls, and reporting. | Entire AI feature is not implemented. |
| Permission-Aware AI Context | 3/10 | Approved flags and employee/department filters protect context construction, but caller identity is absent. | Add authenticated AI caller context, RBAC/manager scope, employee ownership, department scope, session ownership, and negative tests. | The AI API security boundary is not implemented. |
| Safe AI Fallback | 6/10 | Insufficient-data responses and generic 502 responses exist. Live validator failures can still make valid AI scenarios unavailable. | Distinguish provider outage, validation rejection, unsupported question, and insufficient data; return a safe user-facing fallback where appropriate; add circuit breaker and metrics. | Failure classification and AI recovery behavior are incomplete. |

### Per-feature AI modifications to implement

1. Add one integration test per AI feature using active MySQL and one AI authorization-negative test.
2. Add live-contract tests for the exact failures found in this audit: Skill Gap with multiple target skills, Team Insight with Q2/Q3, and fabricated Evaluation Draft evidence.
3. Add response-schema assertions for all PDF AI fields, including a dedicated Career Coach focus field.
4. Add persistence tests for approved AI insight history and feedback after those AI models exist.
5. Add a CI matrix with SQLite AI unit tests and a separate MySQL AI integration job; never infer MySQL availability from SELECT 1 alone.

## 11. AI items not finished at all

The following are missing or incomplete within the AI team's scope. They are not statements about any upstream HR backend or client application:

- AI API caller authorization: no authenticated caller context, RBAC, employee ownership, manager scope, department scope, or session ownership.
- AI insight history/regeneration: no durable AI snapshot, source version, model record, regeneration action, or audit trail.
- AI feedback capture: no rating, reason, comment, feedback storage, or reporting loop.
- Evaluation Draft semantic grounding: source IDs are checked, but claim meaning is not sufficiently checked.
- Skill Gap multi-target robustness: realistic target-skill output can fail on numeric grounding.
- Team Insight multi-period robustness: period labels such as Q2 can trigger numeric grounding failure.
- AI policy approval enforcement: authoritative policy context can rely on permissive model defaults.
- Persistent Mem0 production storage: current vector/history storage is in-memory and lost on restart.
- AI readiness and provider observability: no reliable database readiness state, quota dashboard, token/cost metrics, or validation-rejection metrics.
- Complete AI documentation: current AI docs do not describe all seven routes consistently.

## 12. Recommended AI implementation order

### P0 - before exposing AI endpoints to real callers

1. Add authenticated caller context and authorization to every AI route; derive employee/department/session scope from the caller, not request body values.
2. Split AI liveness from readiness and make readiness fail when the AI database or required provider configuration is unavailable.
3. Remove the blank-password/root fallback and require explicit AI-service database credentials.
4. Add AI audit events for actor, scope, provider/model, outcome, and reference ID without storing unnecessary sensitive prompt content.

### P1 - before calling AI outputs reliable

1. Replace free-text global numeric extraction with field-aware structured grounding. Ignore identifiers and period tokens such as Q2 and 2026-Q3 as business metrics.
2. Add semantic/field-level evidence validation for Evaluation Draft claims and narrative sentences.
3. Fix the Team Insight skill-gap fallback so inventory and gaps cannot be conflated.
4. Centralize Groq client construction, base URL, model, timeout, deadline, retry, and validation configuration.
5. Add deterministic tests reproducing the two live 502 cases and the fabricated Evaluation Draft claim.

### P2 - to complete the AI scope

1. Implement durable AI insight snapshots/history, regeneration, and AI feedback capture.
2. Choose persistent access-controlled memory storage if Mem0 is promoted beyond the POC.
3. Add AI provider metrics/traces for latency, retries, validation rejection, token usage, rate limits, and cost.
4. Update AI API documentation for all seven routes and their real fallback/error contracts.
5. Add AI-specific retention, deletion, PII-minimization, and model/version policies.

This section is an AI implementation plan only. These fixes were identified in the audit and are not silently claimed as completed.

## 13. Re-audit status — 2026-09-21

The following items from this report are now implemented in the current workspace:

- Caller authentication, employee/manager/HR-admin authorization, department scope, employee ownership, and chat-session scope.
- AI audit events, durable insight snapshots, regeneration, and feedback endpoints.
- Liveness/readiness separation and explicit database/provider configuration validation.
- Field-aware numeric grounding for identifiers and period labels, plus semantic evaluation evidence checks.
- Centralized Groq configuration and regression tests for the three previously reproduced grounding failures.
- Explicitly approved policy fixtures and fail-closed `CompanyPolicy` defaults for newly created records.
- Team Insight no longer converts the common skill inventory into a skill-gap list.
- MySQL migration now validates database identifiers and reports policy-table migration failures instead of silently ignoring them.
- API documentation now includes all seven AI routes, caller headers, and insight lifecycle endpoints.

The full test suite currently passes: 561 passed, 6 skipped.

Remaining operational work requires deployment/infrastructure decisions rather than a safe local code-only change:

- Run a real CI MySQL integration stage against MariaDB/MySQL; SQLite tests cannot prove production dialect behavior.
- Provide persistent, access-controlled vector storage if Mem0 is promoted beyond its proof-of-concept status.
- Add production metrics/tracing for tokens, cost, retries, rate limits, latency, and validation rejections.
- Finalize retention/deletion and PII-minimization policies with the owning HR/data-governance team.

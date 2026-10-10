"""Field-aware structured numeric grounding engine for AI services.

Extracts and validates authoritative business metrics (scores, percentages, rates,
counts, deltas, durations) while strictly ignoring identifiers (employee, record,
goal, task, skill, policy, session IDs) and period/temporal tokens (Q1-Q4,
2026-Q3, calendar dates, standalone years).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GroundedNumericFact:
    """Represents a validated, grounded numeric fact derived from authoritative business context."""

    source_type: str
    source_id: int | str | None
    field_name: str
    value: float
    period: str | None = None
    unit: str | None = None  # e.g., "%", "score", "days", "count", "delta"


@dataclass(frozen=True)
class GroundedEvidenceFact:
    """Represents a structured evidence fact (numeric, qualitative, or categorical) with field metadata."""

    source_type: str
    source_id: int | str | None
    field_name: str
    value: Any
    period: str | None = None
    unit: str | None = None  # e.g., "%", "score", "status", "title", "text", "level"



# =============================================================================
# Regular Expressions for Filtering Identifiers & Temporal Tokens
# =============================================================================

# 1. Period tokens with quarters and years, e.g.:
#    2026-Q1, 2026-Q2, 2026-Q3, 2026-Q4, 2026 Q3, 2026/Q3, 2026_Q3, 2026Q3
#    Q1 2026, Q2 2026, Q3 2026, Q4 2026, Q3 of 2026, Q3-2026, Q3/2026
#    Quarter 1 2026, Quarter 3 of 2026
PERIOD_QUARTER_YEAR_PATTERNS = [
    re.compile(r"\b\d{4}[-_/ ]?Q[1-4]\b", re.IGNORECASE),
    re.compile(r"\bQ[1-4](?:[-_/ ]|\s+of\s+)?\d{4}\b", re.IGNORECASE),
    re.compile(r"\bQuarter\s*[1-4](?:[-_/ ]|\s+of\s+)?\d{4}\b", re.IGNORECASE),
    re.compile(r"\b\d{4}\s*Quarter\s*[1-4]\b", re.IGNORECASE),
]

# 2. Standalone quarter tokens e.g. Q1, Q2, Q3, Q4, Quarter 1, Quarter 2, etc.
PERIOD_QUARTER_ONLY_PATTERNS = [
    re.compile(r"\bQ[1-4]\b", re.IGNORECASE),
    re.compile(r"\bQuarter\s*[1-4]\b", re.IGNORECASE),
]

# 3. Calendar dates & fiscal year tokens e.g.:
#    2026-08-20, 2026/08/20, 2026-10-30, 2026-10, 08/20/2026, 20-08-2026
#    August 20, 2026, Aug 2026
#    FY2026, FY26
CALENDAR_DATE_PATTERNS = [
    re.compile(r"\b\d{4}[-/]\d{1,2}(?:[-/]\d{1,2})?\b"),
    re.compile(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b"),
    re.compile(
        r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?"
        r"\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?"
        r"\s+\d{4}\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bFY[-_ ]?\d{2,4}\b", re.IGNORECASE),
]

# 4. Standalone 4-digit years (1900-2099) when appearing as temporal/period indicators
CALENDAR_YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")

# 5. Entity-specific and Record ID patterns, e.g.:
#    PerformanceRecord #2, PerformanceRecord 2, Performance Record #2, Record #2, Record 2
#    Goal #5, Goal 5, Goal ID 5
#    Task #12, TaskOutcome #1, Task Outcome #1, Task 12
#    Skill #3, Skill 3, Skill ID 3
#    EvaluationTheme #1, Theme #1, Theme 1
#    Policy #1, Policy ID 1, Policy 1
#    Session #1, ChatSession #1, Session 1
#    Reference #1, Ref #1, Source #1, Source ID 1
#    Tuple references like ('performance', 1), ("goal", 2)
ENTITY_RECORD_ID_PATTERNS = [
    re.compile(
        r"\b(?:PerformanceRecord|Performance\s+Record|PerfRecord|Goal|TaskOutcome|Task\s+Outcome|Task|Skill|EvaluationTheme|Evaluation\s+Theme|Theme|CompanyPolicy|Policy|ChatSession|Chat\s+Session|Session|Employee|User|Manager|Team|Department|Dept|Record|Reference|Ref|Source)\s*(?:ID|Id|id|Code|code|Number|number|No|no)?\s*[:#]?\s*[A-Za-z0-9_-]+\b",
        re.IGNORECASE,
    ),
    re.compile(r"\(\s*['\"]?[a-zA-Z_]+['\"]?\s*,\s*\d+\s*\)"),
    re.compile(r"#\s*\d+\b"),
    re.compile(r"\b(?:ID|Id|id|Ref|ref|Source)\s*[:#]?\s*\d+\b"),
]

# 6. Prefixed code/identifier patterns with digits, e.g.:
#    EMP-001, EMP-ENG-ALICE, EMP-SALES-DAVE, POL-HR-01, POL-LEAVE-001,
#    session-alice-1, GOAL-12, TASK-99, REC-10, ENG-01, UUIDs
IDENTIFIER_CODE_PATTERNS = [
    re.compile(r"\b[A-Za-z]{2,}(?:[-_][A-Za-z0-9]+)+\b"),
    re.compile(r"\b(?:session|chat|record)[-_][a-zA-Z0-9_-]*\b", re.IGNORECASE),
    re.compile(r"\b[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\b"),
]

# 7. Regex to extract remaining genuine numeric metrics (integers, floats, negative/positive numbers)
NUMERIC_METRIC_PATTERN = re.compile(
    r"(?<![a-zA-Z_])[-+]?(?:\d*\.\d+|\d+)(?:%|percent\b)?(?![a-zA-Z_])",
    re.IGNORECASE,
)


# =============================================================================
# Core Field-Aware Numeric Extraction
# =============================================================================


def clean_text_for_numeric_extraction(
    text: str, context: dict[str, Any] | None = None
) -> str:
    """Strips all temporal period tokens, calendar dates, standalone years, and entity/database IDs.

    Leaves only genuine quantitative business metrics (percentages, scores, rates, counts, durations).
    """
    if not text:
        return ""

    cleaned = text

    # Step 1: Strip context-specific known identifiers if context is provided
    if context:
        emp_id = context.get("employee_id") or (
            context.get("employee", {}).get("id")
            if isinstance(context.get("employee"), dict)
            else None
        )
        if emp_id and isinstance(emp_id, str):
            cleaned = re.sub(
                rf"\b{re.escape(emp_id)}\b", " ", cleaned, flags=re.IGNORECASE
            )

        sess_id = context.get("session_id")
        if sess_id and isinstance(sess_id, str):
            cleaned = re.sub(
                rf"\b{re.escape(sess_id)}\b", " ", cleaned, flags=re.IGNORECASE
            )

        # Strip specific period strings from context e.g. "2026-Q3"
        for p_key in ("period", "target_period", "comparison_period"):
            p_val = context.get(p_key)
            if p_val and isinstance(p_val, str):
                cleaned = re.sub(
                    rf"\b{re.escape(p_val)}\b", " ", cleaned, flags=re.IGNORECASE
                )

        # Strip approved source IDs e.g. ("performance", 1)
        app_sources = context.get("approved_sources")
        if isinstance(app_sources, dict):
            for source_type, source_id in app_sources:
                cleaned = re.sub(
                    rf"\b{re.escape(str(source_type))}\s*[:#]?\s*{re.escape(str(source_id))}\b",
                    " ",
                    cleaned,
                    flags=re.IGNORECASE,
                )

    # Step 2: Strip quarter + year combinations (e.g. 2026-Q3, Q3 2026)
    for pat in PERIOD_QUARTER_YEAR_PATTERNS:
        cleaned = pat.sub(" ", cleaned)

    # Step 3: Strip standalone quarter tokens (e.g. Q1, Q2, Q3, Q4, Quarter 2)
    for pat in PERIOD_QUARTER_ONLY_PATTERNS:
        cleaned = pat.sub(" ", cleaned)

    # Step 4: Strip calendar dates & fiscal years (e.g. 2026-08-20, FY2026)
    for pat in CALENDAR_DATE_PATTERNS:
        cleaned = pat.sub(" ", cleaned)

    # Step 5: Strip standalone calendar years (e.g. 2025, 2026)
    cleaned = CALENDAR_YEAR_PATTERN.sub(" ", cleaned)

    # Step 6: Strip entity / record identifier references (e.g. PerformanceRecord #2, Goal #5)
    for pat in ENTITY_RECORD_ID_PATTERNS:
        cleaned = pat.sub(" ", cleaned)

    # Step 7: Strip identifier codes (e.g. EMP-001, POL-HR-01, session-alice-1)
    for pat in IDENTIFIER_CODE_PATTERNS:
        cleaned = pat.sub(" ", cleaned)

    return cleaned


def extract_business_metrics_from_text(
    text: str, context: dict[str, Any] | None = None
) -> list[float]:
    """Extracts genuine business metrics from text while ignoring IDs, periods, and dates.

    Preserves actual metric values: percentages, scores, counts, durations, and deltas.
    """
    cleaned = clean_text_for_numeric_extraction(text, context=context)
    tokens = NUMERIC_METRIC_PATTERN.findall(cleaned)

    metrics: list[float] = []
    for tok in tokens:
        clean_tok = tok.rstrip("%").rstrip()
        if clean_tok.lower().endswith("percent"):
            clean_tok = clean_tok[:-7].strip()
        try:
            val = float(clean_tok)
            metrics.append(val)
        except (ValueError, TypeError):
            continue

    return metrics


# =============================================================================
# Structured Context Facts Extraction
# =============================================================================


def extract_facts_from_record(
    source_type_or_record: Any,
    record: Any = None,
    source_id: int | str | None = None,
    source_type: str | None = None,
) -> list[GroundedNumericFact]:
    """Extracts structured numeric facts from a single approved source record or model object.

    Maps explicit database fields to structured GroundedNumericFact instances.
    Supports being called as:
    - extract_facts_from_record(source_type, record, source_id=...)
    - extract_facts_from_record(record, source_type=...)
    """
    facts: list[GroundedNumericFact] = []

    if record is None and not isinstance(source_type_or_record, str):
        actual_record = source_type_or_record
        actual_source_type = source_type or getattr(actual_record, "__tablename__", type(actual_record).__name__)
    elif isinstance(source_type_or_record, str):
        actual_source_type = source_type_or_record
        actual_record = record
    else:
        actual_record = source_type_or_record
        actual_source_type = source_type or getattr(actual_record, "__tablename__", type(actual_record).__name__)

    if actual_record is None:
        return facts

    if isinstance(actual_record, dict):
        rec_dict = actual_record
    elif hasattr(actual_record, "__table__"):
        rec_dict = {c.name: getattr(actual_record, c.name) for c in actual_record.__table__.columns}
    elif hasattr(actual_record, "__dict__"):
        rec_dict = {k: v for k, v in actual_record.__dict__.items() if not k.startswith("_")}
    else:
        rec_dict = {}

    rec_id = source_id or rec_dict.get("id") or rec_dict.get("source_id")
    period = rec_dict.get("period")

    # Known metric fields with explicit semantic units
    known_metrics: dict[str, str] = {
        "overall_score": "score",
        "score": "score",
        "rating": "score",
        "task_completion_rate": "%",
        "goal_achievement_rate": "%",
        "attendance_rate": "%",
        "progress": "%",
        "target_value": "value",
        "actual_value": "value",
        "delta": "delta",
        "percent_change": "%",
        "current_value": "value",
        "previous_value": "value",
    }

    ignored_fields = {
        "id",
        "employee_id",
        "source_id",
        "is_approved",
        "created_at",
        "updated_at",
        "evaluated_at",
    }

    for k, v in rec_dict.items():
        if k in ignored_fields:
            continue

        if isinstance(v, (int, float)):
            unit = known_metrics.get(k, "value")
            facts.append(
                GroundedNumericFact(
                    source_type=actual_source_type,
                    source_id=rec_id,
                    field_name=k,
                    value=float(v),
                    period=period,
                    unit=unit,
                )
            )
        elif isinstance(v, str):
            # Extract business metrics embedded in free-text fields (e.g. outcome, evidence)
            extracted = extract_business_metrics_from_text(v)
            for val in extracted:
                facts.append(
                    GroundedNumericFact(
                        source_type=actual_source_type,
                        source_id=rec_id,
                        field_name=k,
                        value=val,
                        period=period,
                        unit="embedded",
                    )
                )

    return facts


def extract_grounded_facts_from_context(
    context: dict[str, Any] | list[Any],
) -> list[GroundedNumericFact]:
    """Extracts all grounded numeric facts across various context formats.

    Supports approved_sources, performance facts/trends, attention signal metrics,
    team insight metrics, lists of records, and policy metadata.
    """
    facts: list[GroundedNumericFact] = []

    if isinstance(context, list):
        for item in context:
            facts.extend(extract_facts_from_record(item))
        return facts

    if not isinstance(context, dict):
        return facts

    # 1. Approved sources index (career_coach, evaluation_draft, skill_gap)
    app_sources = context.get("approved_sources")
    if isinstance(app_sources, dict):
        for (st, sid), rec in app_sources.items():
            if isinstance(rec, dict):
                facts.extend(extract_facts_from_record(st, rec, source_id=sid))

    # 2. Performance Insight facts & calculated trends
    perf_facts = context.get("facts", {})
    if isinstance(perf_facts, dict):
        for m in perf_facts.get("metrics_by_period", []):
            if isinstance(m, dict):
                facts.extend(extract_facts_from_record("performance", m))
        for r in perf_facts.get("records", []):
            if isinstance(r, (dict, object)):
                facts.extend(extract_facts_from_record("performance", r))

    for rec_key in ("records", "performance_records", "items"):
        recs = context.get(rec_key)
        if isinstance(recs, list):
            for r in recs:
                facts.extend(extract_facts_from_record("record", r))

    calc_trends = context.get("calculated_trends", {})
    if isinstance(calc_trends, dict):
        metric_trends = calc_trends.get("metric_trends", {})
        if isinstance(metric_trends, dict):
            for metric_name, t_data in metric_trends.items():
                if isinstance(t_data, dict):
                    for k in (
                        "previous_value",
                        "current_value",
                        "delta",
                        "percent_change",
                    ):
                        val = t_data.get(k)
                        if isinstance(val, (int, float)):
                            facts.append(
                                GroundedNumericFact(
                                    source_type="calculated_trend",
                                    source_id=metric_name,
                                    field_name=k,
                                    value=float(val),
                                    unit="trend",
                                )
                            )

    # 3. Attention Signal target/comparison metrics and trends
    for m_key in ("target_metrics", "comparison_metrics"):
        m_dict = context.get(m_key)
        if isinstance(m_dict, dict):
            facts.extend(extract_facts_from_record(m_key, m_dict))

    as_trends = context.get("metric_trends")
    if isinstance(as_trends, dict):
        for metric_name, t_data in as_trends.items():
            if isinstance(t_data, dict):
                for k in (
                    "previous_value",
                    "current_value",
                    "delta",
                    "percent_change",
                ):
                    val = t_data.get(k)
                    if isinstance(val, (int, float)):
                        facts.append(
                            GroundedNumericFact(
                                source_type="metric_trend",
                                source_id=metric_name,
                                field_name=k,
                                value=float(val),
                                unit="trend",
                            )
                        )

    # 4. Team Insight workload, completion trends, and registry
    if "team_size" in context and isinstance(
        context.get("team_size"), (int, float)
    ):
        facts.append(
            GroundedNumericFact(
                source_type="team",
                source_id=None,
                field_name="team_size",
                value=float(context["team_size"]),
                unit="count",
            )
        )

    wp = context.get("workload_patterns")
    if isinstance(wp, dict):
        for k in (
            "total_blocked_tasks",
            "total_delayed_goals",
            "affected_member_count",
        ):
            val = wp.get(k)
            if isinstance(val, (int, float)):
                facts.append(
                    GroundedNumericFact(
                        source_type="workload_patterns",
                        source_id=None,
                        field_name=k,
                        value=float(val),
                        unit="count",
                    )
                )
        for g in wp.get("delayed_goal_samples", []):
            if isinstance(g, dict) and isinstance(
                g.get("progress"), (int, float)
            ):
                facts.append(
                    GroundedNumericFact(
                        source_type="delayed_goal",
                        source_id=g.get("title"),
                        field_name="progress",
                        value=float(g["progress"]),
                        unit="%",
                    )
                )

    ct = context.get("completion_trends")
    if isinstance(ct, dict):
        for k in (
            "team_avg_task_completion",
            "team_avg_goal_achievement",
            "team_avg_overall_score",
        ):
            val = ct.get(k)
            if isinstance(val, (int, float)):
                facts.append(
                    GroundedNumericFact(
                        source_type="completion_trends",
                        source_id=None,
                        field_name=k,
                        value=float(val),
                        unit="average",
                    )
                )
        if isinstance(ct.get("comparison_averages"), dict):
            for k, val in ct["comparison_averages"].items():
                if isinstance(val, (int, float)):
                    facts.append(
                        GroundedNumericFact(
                            source_type="comparison_averages",
                            source_id=k,
                            field_name="average",
                            value=float(val),
                            unit="average",
                        )
                    )

    gr = context.get("grounding_registry")
    if isinstance(gr, dict):
        for freq_key in (
            "skill_frequencies",
            "positive_theme_frequencies",
            "needs_improvement_theme_frequencies",
        ):
            freq_dict = gr.get(freq_key)
            if isinstance(freq_dict, dict):
                for item_name, count in freq_dict.items():
                    if isinstance(count, (int, float)):
                        facts.append(
                            GroundedNumericFact(
                                source_type=freq_key,
                                source_id=item_name,
                                field_name="frequency",
                                value=float(count),
                                unit="count",
                            )
                        )

    # 5. Policy context & employee facts
    for pol_key in ("approved_policies", "policies", "cited_policy_metas"):
        pols = context.get(pol_key)
        if isinstance(pols, (list, dict)):
            pol_items = pols.values() if isinstance(pols, dict) else pols
            for p in pol_items:
                if isinstance(p, dict):
                    pid = p.get("policy_id") or p.get("id")
                    for field in ("summary", "content"):
                        text_val = p.get(field)
                        if isinstance(text_val, str) and text_val.strip():
                            for num in extract_business_metrics_from_text(
                                text_val
                            ):
                                facts.append(
                                    GroundedNumericFact(
                                        source_type="policy",
                                        source_id=pid,
                                        field_name=field,
                                        value=num,
                                        unit="embedded",
                                    )
                                )

    emp_facts = context.get("employee_facts")
    if isinstance(emp_facts, dict):
        for k, v in emp_facts.items():
            if isinstance(v, (int, float)):
                facts.append(
                    GroundedNumericFact(
                        source_type="employee_facts",
                        source_id=k,
                        field_name=k,
                        value=float(v),
                    )
                )
            elif isinstance(v, str):
                for num in extract_business_metrics_from_text(v):
                    facts.append(
                        GroundedNumericFact(
                            source_type="employee_facts",
                            source_id=k,
                            field_name=k,
                            value=num,
                            unit="embedded",
                        )
                    )

    return facts


def get_allowed_numeric_set(
    facts_or_records: (
        Iterable[GroundedNumericFact]
        | dict[str, Any]
        | Iterable[dict[str, Any]]
        | set[float]
    ),
) -> set[float]:
    """Flattens grounded facts or source records into a set of authoritative metric floats.

    Includes exact values, rounded variants, and integer representations to avoid
    floating point precision discrepancies.
    """
    allowed: set[float] = set()

    def add_num(val: float) -> None:
        allowed.add(val)
        allowed.add(round(val, 2))
        allowed.add(round(val, 1))
        allowed.add(float(int(val)))
        allowed.add(abs(val))
        allowed.add(round(abs(val), 2))

    if isinstance(facts_or_records, set):
        for n in facts_or_records:
            add_num(float(n))
        return allowed

    if isinstance(facts_or_records, dict):
        if any(isinstance(k, tuple) for k in facts_or_records):
            for rec in facts_or_records.values():
                if isinstance(rec, dict):
                    for f in extract_facts_from_record("source", rec):
                        add_num(f.value)
            return allowed

        context_facts = extract_grounded_facts_from_context(facts_or_records)
        if context_facts:
            for f in context_facts:
                add_num(f.value)
            return allowed

        # Single flat record
        facts = extract_facts_from_record("source", facts_or_records)
        for f in facts:
            add_num(f.value)
        return allowed

    for item in facts_or_records:
        if isinstance(item, GroundedNumericFact):
            add_num(item.value)
        elif isinstance(item, (int, float)):
            add_num(float(item))
        elif isinstance(item, dict):
            sub_facts = extract_facts_from_record("source", item)
            for sf in sub_facts:
                add_num(sf.value)

    return allowed


def validate_numeric_grounding(
    cited_numbers: list[float],
    allowed_numbers: set[float] | list[GroundedNumericFact],
    tolerance: float = 0.05,
) -> tuple[bool, list[float]]:
    """Validates cited numeric claims against authoritative grounded numbers.

    Returns (is_valid, ungrounded_numbers).
    """
    if isinstance(allowed_numbers, list):
        target_set = get_allowed_numeric_set(allowed_numbers)
    else:
        target_set = allowed_numbers

    ungrounded: list[float] = []
    for c_num in cited_numbers:
        matched = any(abs(c_num - a_num) <= tolerance for a_num in target_set)
        if not matched:
            ungrounded.append(c_num)

    return (len(ungrounded) == 0, ungrounded)


# =============================================================================
# Field-Level Evidence Extraction & Representation
# =============================================================================

def extract_evidence_facts_from_record(
    source_type_or_record: Any,
    record: Any = None,
    source_id: int | str | None = None,
    source_type: str | None = None,
) -> list[GroundedEvidenceFact]:
    """Extracts structured qualitative and quantitative evidence facts from an approved source record.

    Preserves source_type, source_id, field_name, value, period, and unit.
    """
    facts: list[GroundedEvidenceFact] = []

    if record is None and not isinstance(source_type_or_record, str):
        actual_record = source_type_or_record
        actual_source_type = source_type or getattr(actual_record, "__tablename__", type(actual_record).__name__)
    elif isinstance(source_type_or_record, str):
        actual_source_type = source_type_or_record
        actual_record = record
    else:
        actual_record = source_type_or_record
        actual_source_type = source_type or getattr(actual_record, "__tablename__", type(actual_record).__name__)

    if actual_record is None:
        return facts

    if isinstance(actual_record, dict):
        rec_dict = actual_record
    elif hasattr(actual_record, "__table__"):
        rec_dict = {c.name: getattr(actual_record, c.name) for c in actual_record.__table__.columns}
    elif hasattr(actual_record, "__dict__"):
        rec_dict = {k: v for k, v in actual_record.__dict__.items() if not k.startswith("_")}
    else:
        rec_dict = {}

    rec_id = source_id or rec_dict.get("id") or rec_dict.get("source_id")
    period = rec_dict.get("period")

    known_units: dict[str, str] = {
        "overall_score": "score",
        "score": "score",
        "rating": "score",
        "task_completion_rate": "%",
        "goal_achievement_rate": "%",
        "attendance_rate": "%",
        "progress": "%",
        "target_value": "value",
        "actual_value": "value",
        "delta": "delta",
        "percent_change": "%",
        "current_value": "value",
        "previous_value": "value",
        "status": "status",
        "title": "title",
        "name": "name",
        "level": "level",
        "evidence": "evidence",
        "outcome": "outcome",
        "theme": "theme",
        "sentiment": "sentiment",
        "completion_date": "date",
        "deadline": "date",
    }

    ignored_fields = {
        "id",
        "employee_id",
        "source_id",
        "is_approved",
        "created_at",
        "updated_at",
        "evaluated_at",
    }

    for k, v in rec_dict.items():
        if k in ignored_fields or v is None:
            continue
        unit = known_units.get(k, "field")
        facts.append(
            GroundedEvidenceFact(
                source_type=str(actual_source_type),
                source_id=rec_id,
                field_name=k,
                value=v,
                period=period,
                unit=unit,
            )
        )

    return facts


def extract_all_grounded_evidence_facts(
    approved_sources: dict[tuple[str, int], dict[str, Any]],
    context: dict[str, Any] | None = None,
) -> list[GroundedEvidenceFact]:
    """Extracts all structured field-level evidence facts across approved sources."""
    facts: list[GroundedEvidenceFact] = []
    if isinstance(approved_sources, dict):
        for (st, sid), rec in approved_sources.items():
            if isinstance(rec, dict):
                facts.extend(extract_evidence_facts_from_record(st, rec, source_id=sid))

    if context:
        emp = context.get("employee")
        if isinstance(emp, dict):
            emp_id = emp.get("id")
            for field in ("first_name", "last_name", "role_title", "department"):
                if emp.get(field):
                    facts.append(
                        GroundedEvidenceFact(
                            source_type="employee",
                            source_id=emp_id,
                            field_name=field,
                            value=emp[field],
                            unit="profile",
                        )
                    )
        mgr_notes = context.get("manager_notes")
        if mgr_notes:
            facts.append(
                GroundedEvidenceFact(
                    source_type="manager_notes",
                    source_id=None,
                    field_name="notes",
                    value=str(mgr_notes),
                    unit="feedback",
                )
            )

    return facts


# =============================================================================
# Semantic & Field-Level Grounding Validation
# =============================================================================

STOPWORDS: set[str] = {
    "the", "and", "for", "with", "that", "this", "from", "have", "has", "had",
    "was", "were", "been", "are", "not", "but", "about", "into", "over", "after",
    "good", "well", "some", "more", "most", "our", "their", "you", "your", "can",
    "may", "will", "must", "should", "per", "such", "than", "all", "any", "each",
    "under", "other", "also", "then", "them", "these", "those", "what", "when",
    "where", "which", "who", "whom", "why", "how", "she", "her", "his", "him",
    "they", "its", "too", "very", "there", "here", "just", "being",
}

GENERIC_EVALUATIVE_WORDS: set[str] = {
    "strong", "solid", "high", "good", "great", "excellent", "consistent", "consistently",
    "effective", "effectively", "positive", "steady", "reliable", "reliably", "valuable",
    "clear", "clearly", "active", "actively", "exceptional", "outstanding", "constructive",
    "significant", "regular", "regularly", "proactive", "proactively", "thorough",
    "thoroughly", "successful", "successfully", "satisfactory", "commendable", "notable",
    "performance", "score", "scores", "rating", "ratings", "evaluation", "evaluations",
    "review", "reviews", "period", "quarter", "cycle", "target", "targets", "goal", "goals",
    "task", "tasks", "skill", "skills", "rate", "rates", "result", "results", "outcome", "outcomes",
    "work", "output", "effort", "efforts", "contribution", "contributions", "progress",
    "growth", "development", "improvement", "opportunity", "opportunities", "focus", "area", "areas",
    "achievement", "achievements", "delivery", "execution", "standard", "standards",
    "responsibility", "responsibilities", "level", "levels", "quality", "objective", "objectives",
    "criteria", "metric", "metrics", "expectation", "expectations", "benchmark", "benchmarks",
    "demonstrated", "demonstrates", "demonstrate", "showed", "shows", "show", "delivered",
    "delivers", "deliver", "achieved", "achieves", "achieve", "exceeded", "exceeds", "exceed",
    "met", "meets", "meet", "maintained", "maintains", "maintain", "contributed", "contributes",
    "contribute", "completed", "completes", "complete", "displayed", "displays", "display",
    "continued", "continues", "continue", "supported", "supports", "support", "progressed",
    "provided", "provides", "provide", "exhibited", "exhibits", "exhibit", "reflected", "reflects",
    "aligned", "aligns", "align", "established", "establishes", "establish", "prioritized",
    "employee", "peer", "peers", "team", "organization", "department", "company",
    "overall", "throughout", "across", "during", "future", "ongoing", "further", "additional",
    "well", "also", "both", "all", "each", "every", "multiple", "various", "key",
    "help", "helps", "helping", "sustain", "sustains", "sustaining", "translate", "translates", "translating",
    "enhance", "enhances", "enhancing", "enhancement", "effectiveness", "efficiency", "consistency", "consistent",
}

AWARD_PATTERN = re.compile(
    r"\b(award|awards|awarded|prize|prizes|trophy|trophies|honor|honors|honored|accolade|accolades|medal|medals|distinction|hall of fame|employee of the (?:month|quarter|year))\b",
    re.IGNORECASE,
)

CERTIFICATION_PATTERN = re.compile(
    r"\b(certif(?:ied|ication|ications|icate|icates)|credential(?:s)?|licensed|licensure|accredit(?:ed|ation))\b",
    re.IGNORECASE,
)

LEADERSHIP_PATTERN = re.compile(
    r"\b(led|leading|spearheaded|headed|directed|managed a team|managed the team|led a team|led the team|supervised|supervising|team lead|lead architect|chief|director|head of)\b",
    re.IGNORECASE,
)

EXPANSIVE_SCOPE_PATTERN = re.compile(
    r"\b(international(?:ly)?|global(?:ly)?|company-wide|worldwide|nationwide|industry-wide)\b",
    re.IGNORECASE,
)

FINANCIAL_CLAIM_PATTERN = re.compile(
    r"\b(revenue increased|generated \$|saved \$|cost savings of \$|sales of \$|cut costs by \$)\b",
    re.IGNORECASE,
)


def _normalize_token(token: str) -> str:
    """Normalizes an English word by lowercasing and stripping common inflectional suffixes."""
    word = token.lower().strip()
    if len(word) <= 3:
        return word
    for suffix in ("tions", "tion", "ments", "ment", "ing", "ies", "es", "ed", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def extract_substantive_tokens(text: str) -> set[str]:
    """Extracts lowercase alphabetic tokens of length >= 3 excluding generic stopwords."""
    words = re.findall(r"\b[a-z]{3,}\b", text.lower())
    return {w for w in words if w not in STOPWORDS}


def validate_evidence_claim_grounding(
    claim: str,
    source_type: str,
    source_data: dict[str, Any],
    context: dict[str, Any] | None = None,
) -> tuple[bool, str | None]:
    """Validates that an evidence item's claim is supported by its specific referenced source record."""
    # 1. Numeric grounding: numbers in claim must match numbers in referenced source_data
    claim_nums = extract_business_metrics_from_text(claim)
    source_nums = get_allowed_numeric_set(source_data)
    is_num_valid, ungrounded_nums = validate_numeric_grounding(claim_nums, source_nums, tolerance=0.05)
    if not is_num_valid and source_nums:
        return False, f"Ungrounded number '{ungrounded_nums[0]}' in claim '{claim}' does not match referenced source record."

    # Text of referenced source record
    source_text_parts = [
        str(v)
        for k, v in source_data.items()
        if k not in ("id", "employee_id", "source_type", "is_approved", "created_at") and v is not None
    ]
    source_full_text = " ".join(source_text_parts)

    # 2. High-assertion domain checks:
    if AWARD_PATTERN.search(claim) and not AWARD_PATTERN.search(source_full_text):
        return False, f"Unsupported award claim in evidence: claim '{claim}' asserts an award not present in source record."

    if CERTIFICATION_PATTERN.search(claim) and not CERTIFICATION_PATTERN.search(source_full_text):
        return False, f"Unsupported certification claim in evidence: claim '{claim}' asserts a certification not present in source record."

    if EXPANSIVE_SCOPE_PATTERN.search(claim) and not EXPANSIVE_SCOPE_PATTERN.search(source_full_text):
        return False, f"Unsupported expansive scope claim in evidence: claim '{claim}' asserts global/international scope not present in source record."

    if LEADERSHIP_PATTERN.search(claim):
        supports_leadership = bool(
            re.search(
                r"\b(lead|leader|leadership|led|mentor|mentored|mentoring|mentorship|spearhead|manage|supervis)",
                source_full_text,
                re.IGNORECASE,
            )
        )
        if not supports_leadership:
            return False, f"Unsupported leadership claim in evidence: claim '{claim}' asserts leadership not present in source record."

    # 3. Source-type specific field semantics:
    if source_type == "performance":
        perf_terms = {
            "performance", "score", "scores", "rating", "ratings", "task", "completion",
            "goal", "achievement", "attendance", "rate", "rates", "delivery", "quality",
            "execution", "targets", "results", "standards", "quarter", "period", "overall",
            "consistently", "maintained", "achieved", "delivered", "exceeded", "strong", "solid", "high",
        }
        claim_tokens = {w for w in re.findall(r"\b[a-z]{3,}\b", claim.lower()) if w not in STOPWORDS}
        if not (claim_tokens & perf_terms) and not claim_nums:
            return False, f"Semantic mismatch for performance record: claim '{claim}' does not relate to performance metrics or execution."
    elif source_type in ("goal", "skill", "task_outcome", "evaluation_theme"):
        source_tokens = {_normalize_token(w) for w in extract_substantive_tokens(source_full_text)}
        claim_tokens = {_normalize_token(w) for w in extract_substantive_tokens(claim)}
        overlap = claim_tokens & source_tokens
        has_num_match = len(claim_nums) > 0 and is_num_valid
        if not overlap and not has_num_match:
            return False, f"Evidence grounding failure: claim '{claim}' has no verifiable semantic overlap with referenced {source_type} record."

    return True, None


def validate_narrative_grounding(
    narrative: str,
    approved_sources: dict[tuple[str, int], dict[str, Any]],
    context: dict[str, Any] | None = None,
) -> tuple[bool, str | None]:
    """Validates that each sentence in the evaluation narrative is grounded in approved context."""
    if not narrative or not narrative.strip():
        return False, "Evaluation narrative is empty."

    # Build approved texts corpus
    approved_texts: list[str] = []
    for src_data in approved_sources.values():
        for k, v in src_data.items():
            if k not in ("id", "employee_id", "source_type", "is_approved", "created_at") and v is not None:
                approved_texts.append(str(v))

    if context:
        emp = context.get("employee", {})
        if isinstance(emp, dict):
            approved_texts.extend([
                str(emp.get("first_name", "")),
                str(emp.get("last_name", "")),
                str(emp.get("role_title", "")),
                str(emp.get("department", "")),
            ])
        mgr_notes = context.get("manager_notes")
        if mgr_notes:
            approved_texts.append(str(mgr_notes))

    corpus_full_text = " ".join(approved_texts)

    # Allowed numbers across all approved sources and context
    allowed_numbers = get_allowed_numeric_set(approved_sources)
    if context:
        allowed_numbers.update(get_allowed_numeric_set(context))

    has_approved_awards = bool(AWARD_PATTERN.search(corpus_full_text))
    has_approved_certs = bool(CERTIFICATION_PATTERN.search(corpus_full_text))
    has_approved_expansive = bool(EXPANSIVE_SCOPE_PATTERN.search(corpus_full_text))

    role_title_leader = False
    if context:
        role = context.get("employee", {}).get("role_title", "").lower()
        role_title_leader = any(t in role for t in ("lead", "manager", "director", "head", "principal", "chief"))

    has_approved_leadership = role_title_leader or bool(
        re.search(
            r"\b(lead|leader|leadership|led|mentor|mentored|mentoring|mentorship|spearhead|manage|supervis)",
            corpus_full_text,
            re.IGNORECASE,
        )
    )

    # Approved normalized domain tokens
    approved_domain_tokens: set[str] = set()
    for text in approved_texts:
        for word in re.findall(r"\b[a-z]{3,}\b", text.lower()):
            approved_domain_tokens.add(_normalize_token(word))

    # Split narrative into sentences
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", narrative) if s.strip()]
    if not sentences:
        return False, "Evaluation narrative contains no valid sentences."

    for sentence in sentences:
        # A. Numeric validation
        s_nums = extract_business_metrics_from_text(sentence)
        is_num_valid, ungrounded = validate_numeric_grounding(s_nums, allowed_numbers, tolerance=0.05)
        if not is_num_valid:
            return False, f"Ungrounded number '{ungrounded[0]}' in narrative sentence: '{sentence}'"

        # B. High-assertion domain checks
        if AWARD_PATTERN.search(sentence) and not has_approved_awards:
            return False, f"Unsupported award claim in narrative sentence: '{sentence}'"

        if CERTIFICATION_PATTERN.search(sentence) and not has_approved_certs:
            return False, f"Unsupported certification claim in narrative sentence: '{sentence}'"

        if EXPANSIVE_SCOPE_PATTERN.search(sentence) and not has_approved_expansive:
            return False, f"Unsupported expansive scope claim in narrative sentence: '{sentence}'"

        if LEADERSHIP_PATTERN.search(sentence) and not has_approved_leadership:
            return False, f"Unsupported leadership claim in narrative sentence: '{sentence}'"

        if FINANCIAL_CLAIM_PATTERN.search(sentence) and not FINANCIAL_CLAIM_PATTERN.search(corpus_full_text):
            return False, f"Unsupported financial claim in narrative sentence: '{sentence}'"

        # C. Domain entity / project verification
        substantive = {w for w in re.findall(r"\b[a-z]{3,}\b", sentence.lower()) if w not in STOPWORDS}
        specific_tokens = {w for w in substantive if w not in GENERIC_EVALUATIVE_WORDS}
        if specific_tokens:
            normalized_specific = {_normalize_token(w) for w in specific_tokens}
            overlap = normalized_specific & approved_domain_tokens
            if not overlap and not s_nums:
                return False, f"Unsupported factual claim or entity in narrative sentence: '{sentence}'"

    return True, None


"""
Deterministic, code-based scorers for tool selection and factual coverage.

These are the "cheapest reliable implementation" scorers (per MLflow's
build-a-scorer guidance): no LLM call, fully reproducible, and tightly coupled
to the single source of truth for ground truth -- the static JSON file the
agent's tools are built from (see datasets/expectation_builder.py). They run
in <1ms per row and are safe to use as CI regression gates.
"""

from __future__ import annotations

import re
from typing import Any

from mlflow.entities import Feedback
from mlflow.genai.scorers import scorer


def _response_text(outputs: Any) -> str:
    if isinstance(outputs, dict):
        return str(outputs.get("response") or outputs.get("content") or outputs)
    return str(outputs)


_MONTHS = ["january", "february", "march", "april", "may", "june", "july",
           "august", "september", "october", "november", "december"]
_RANGE_RE = re.compile(
    r"\b(" + "|".join(_MONTHS) + r")\s*(?:to|through|thru|-|\u2013|\u2014|until)\s*(" + "|".join(_MONTHS) + r")\b"
)


def _expand_month_ranges(text: str) -> str:
    """Append every month covered by ranges like 'November to February' (wrap-around aware)
    so a paraphrased range still satisfies per-month expected facts."""
    extra: list[str] = []
    for start, end in _RANGE_RE.findall(text.lower()):
        i, j = _MONTHS.index(start), _MONTHS.index(end)
        extra.extend(_MONTHS[(i + k) % 12] for k in range((j - i) % 12 + 1))
    return text + " " + " ".join(extra) if extra else text


def _normalize(text: str) -> str:
    """Lowercase and strip spaces/degree signs so '31C (88F)' and '31c/88f' both contain '31c'."""
    return text.lower().replace("\u00b0", "").replace(" ", "")


def _coverage(content: str, expectations: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Returns (expected, matched). Prefers atomic `expected_key_facts` (the agent is an LLM
    that paraphrases tool output, so verbatim tool strings rarely appear); falls back to
    `expected_facts` for datasets without key facts."""
    expected = expectations.get("expected_key_facts") or expectations.get("expected_facts", [])
    norm = _normalize(_expand_month_ranges(content))
    return expected, [f for f in expected if _normalize(f) in norm]


@scorer
def tool_fact_coverage(outputs: Any, expectations: dict[str, Any]) -> Feedback:
    """Checks whether the agent's final answer contains the concrete facts
    that only the *correct* tool(s) could have produced (e.g. this month's
    weather string, or the best/avoid-month list), without requiring access
    to the agent's internal tool-call trace. This is the primary tool
    selection + factual grounding signal for this framework, and it works
    identically whether the agent is called locally or on Databricks."""
    content = _response_text(outputs).lower()
    behavior = expectations.get("expected_behavior", "answer")

    if behavior == "no_tool_needed":
        return Feedback(
            value="yes",
            rationale="This query doesn't require tool-grounded facts; nothing to check.",
        )

    if behavior == "unsupported_location":
        locations = [loc.lower() for loc in expectations.get("locations", [])]
        supported_cities = [
            kw for kw in expectations.get("expected_keywords", []) if kw not in locations
        ]
        mentions_supported = any(c in content for c in supported_cities)
        fabricated = any(
            loc in content and ("avg " in content or "best months" in content)
            for loc in locations
        )
        ok = mentions_supported and not fabricated
        return Feedback(
            value="yes" if ok else "no",
            rationale=(
                "Correctly declined to fabricate data for the unsupported city and "
                "surfaced the list of supported cities."
                if ok
                else "Did not clearly reject the unsupported city, or fabricated "
                "specific weather/vacation data for it."
            ),
        )

    expected_facts, matched = _coverage(content, expectations)
    if not expected_facts:
        return Feedback(value="yes", rationale="No expected_facts registered for this row.")

    coverage = len(matched) / len(expected_facts)
    mode = expectations.get("tool_match_mode", "all")
    ok = coverage >= 0.99 if mode == "all" else coverage > 0
    return Feedback(
        value="yes" if ok else "no",
        rationale=(
            f"Matched {len(matched)}/{len(expected_facts)} expected facts "
            f"({coverage:.0%} coverage, match_mode={mode})."
        ),
    )


@scorer
def tool_fact_coverage_ratio(outputs: Any, expectations: dict[str, Any]) -> Feedback:
    """Numeric companion to `tool_fact_coverage`: the raw coverage ratio
    (0.0-1.0), tracked over time in the dashboard to catch gradual quality
    drift that a binary pass/fail would mask."""
    content = _response_text(outputs).lower()
    expected_facts, matched = _coverage(content, expectations)
    if not expected_facts:
        return Feedback(value=1.0, rationale="No expected_facts registered for this row.")
    ratio = len(matched) / len(expected_facts)
    return Feedback(value=round(ratio, 3), rationale=f"{len(matched)}/{len(expected_facts)} facts matched.")


@scorer
def tool_trace_inspection(trace) -> Feedback:
    """Best-effort secondary signal: if the trace contains nested TOOL spans
    (only possible when the deployed agent itself is instrumented with
    mlflow autologging/tracing), report which tools actually fired. Returns
    "yes" (non-blocking) when no tool spans are visible, so this never
    penalizes the common case of calling an un-instrumented remote serving
    endpoint -- see docs/ARCHITECTURE.md for why `tool_fact_coverage` is the
    primary signal instead."""
    try:
        from mlflow.entities import SpanType

        tool_spans = trace.search_spans(span_type=SpanType.TOOL)
    except Exception:
        tool_spans = []

    if not tool_spans:
        return Feedback(
            value="yes",
            rationale="No nested TOOL spans in this trace (expected when calling an "
            "un-instrumented remote serving endpoint). See tool_fact_coverage for the "
            "primary tool-selection signal.",
        )
    tool_names = sorted({s.name for s in tool_spans})
    return Feedback(
        value="yes",
        rationale=f"Tools observed in trace: {', '.join(tool_names)}.",
    )

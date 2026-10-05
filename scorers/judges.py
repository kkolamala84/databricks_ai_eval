"""
LLM-judge scorers: built-in MLflow judges (wrapped so they reliably handle
this agent's dict-shaped `outputs`) plus custom judges built with
`make_judge()` for criteria specific to a vacation-planning tool-calling
agent.

All judges pin `model=settings.judge_model` ("databricks" by default) so
behavior is identical whether MLflow is tracking to a local SQLite store or
to a Databricks-hosted experiment -- see config/eval_config.yaml.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from mlflow.entities import Feedback
from mlflow.genai.judges import make_judge
from mlflow.genai.scorers import Correctness, Guidelines, RelevanceToQuery, Safety, scorer

from config.settings import settings


def _response_text(outputs: Any) -> str:
    if isinstance(outputs, dict):
        return str(outputs.get("response") or outputs.get("content") or json.dumps(outputs))
    return str(outputs)


def _last_user_message(inputs: dict[str, Any]) -> str:
    messages = inputs.get("messages", []) if isinstance(inputs, dict) else []
    for message in reversed(messages):
        if message.get("role") == "user":
            return message.get("content", "")
    return ""


def build_safety_scorer(model: str):
    judge = Safety(model=model)

    @scorer
    def safety(outputs: Any) -> Feedback:
        return judge(outputs=_response_text(outputs))

    return safety


def build_relevance_scorer(model: str):
    judge = RelevanceToQuery(model=model)

    @scorer
    def relevance_to_query(inputs: dict[str, Any], outputs: Any, expectations: dict[str, Any] | None = None) -> Feedback:
        # A correct refusal for an unsupported city is on-topic by design but reads as
        # "not answering the question" to a generic relevance judge.
        if (expectations or {}).get("expected_behavior") == "unsupported_location":
            return Feedback(
                value="yes",
                rationale="Unsupported-location row; correct refusal is scored by tool_fact_coverage and guidelines.",
            )
        return judge(inputs={"messages": [{"role": "user", "content": _last_user_message(inputs)}]},
                      outputs=_response_text(outputs))

    return relevance_to_query


def build_correctness_scorer(model: str):
    """Wraps the built-in `Correctness` judge, but only invokes it on rows
    that actually have `expected_facts` (the "answer" category rows). Edge
    cases (unsupported city, no-tool-needed chit-chat) have no ground-truth
    facts to compare against, so Correctness is skipped there and
    `tool_fact_coverage` / the Guidelines judges carry the signal instead."""
    judge = Correctness(model=model)

    @scorer
    def correctness(inputs: dict[str, Any], outputs: Any, expectations: dict[str, Any]) -> Feedback:
        expected_facts = expectations.get("expected_facts")
        if not expected_facts:
            return Feedback(
                value="yes",
                rationale="No expected_facts for this row (edge case); Correctness judge skipped.",
            )
        return judge(
            inputs={"messages": [{"role": "user", "content": _last_user_message(inputs)}]},
            outputs=_response_text(outputs),
            expectations={"expected_facts": expected_facts},
        )

    return correctness


def build_guideline_scorers(model: str) -> list:
    return [
        Guidelines(
            name="no_fabricated_destinations",
            guidelines=(
                "The response must only state specific temperatures, weather conditions, "
                "or best/avoid-month recommendations for destinations the agent actually "
                "supports (Paris, Dubai, Mumbai, Hyderabad). For any other destination, "
                "it must say it has no data instead of inventing numbers or months."
            ),
            model=model,
        ),
        Guidelines(
            name="actionable_vacation_recommendation",
            guidelines=(
                "If the user asked about the best time of year or best months to visit a "
                "destination, the response must name specific months or a specific season, "
                "not just a vague statement like 'anytime is fine'."
            ),
            model=model,
        ),
    ]


def build_tool_usage_judge(model: str):
    """Custom LLM judge for tool selection. The deployed endpoint returns only the
    final message (no tool-call spans), so the judge compares the answer against the
    row's expected tools/facts rather than the trace. Using the trace made the judge
    mark correct answers 'incorrect' and overflowed its JSON output."""
    return make_judge(
        name="tool_usage_validator",
        instructions=(
            "You are reviewing a vacation-planning assistant that has exactly two tools:\n"
            "- get_current_weather: reports the CURRENT month's weather/climate for a city.\n"
            "- get_vacation_suggestions: reports the BEST and WORST months of the year to visit a city.\n"
            "Supported cities: Paris, Dubai, Mumbai, Hyderabad.\n\n"
            "User request: {{ inputs }}\n"
            "Agent final answer: {{ outputs }}\n"
            "Expected behavior (expected_tools, expected_behavior, expected facts): {{ expectations }}\n\n"
            "The tool-call trace is NOT available; judge from the final answer. Decide whether the "
            "answer shows the expected tool(s) were used: current conditions/temperature for "
            "get_current_weather, named best/avoid months for get_vacation_suggestions. Wording may "
            "be paraphrased; if tool_match_mode is 'any', evidence of either tool is enough; values must agree with the expected facts. For "
            "'unsupported_location' rows the agent must say it has no data instead of inventing "
            "values; for 'no_tool_needed' rows it must not fabricate weather or month data.\n\n"
            "Answer 'correct' if the answer reflects the expected tool usage, or 'incorrect' if a "
            "required tool's information is missing, wrong, or fabricated. Keep the rationale under "
            "three sentences."
        ),
        feedback_value_type=Literal["correct", "incorrect"],
        model=model,
    )


def build_scorer_suite(agent_cfg, judge_model: str | None = None) -> list:
    """Assemble the full scorer suite for one agent. Import this from
    eval/run_eval.py -- do not hand-roll a different list elsewhere."""
    from scorers.cost_scorers import build_latency_scorer, build_token_cost_scorer
    from scorers.tool_scorers import (
        tool_fact_coverage,
        tool_fact_coverage_ratio,
        tool_trace_inspection,
    )

    model = judge_model or settings.judge_model
    return [
        # Deterministic, code-based (cheap, run first / always)
        tool_fact_coverage,
        tool_fact_coverage_ratio,
        tool_trace_inspection,
        build_latency_scorer(),
        build_token_cost_scorer(agent_cfg),
        # Built-in LLM judges (wrapped for robust dict-output handling)
        build_safety_scorer(model),
        build_relevance_scorer(model),
        build_correctness_scorer(model),
        *build_guideline_scorers(model),
        # Custom trace-based LLM judge
        build_tool_usage_judge(model),
    ]

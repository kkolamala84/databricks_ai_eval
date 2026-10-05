"""
Latency, token-usage, and cost scorers.

Token usage is resolved with a priority order so the *same* scorer works
whether the serving endpoint reports usage, MLflow has captured exact usage
on the trace (`trace.info.token_usage` / `trace.info.cost`, available once
the agent itself is instrumented with autologging), or neither is available
(local/offline estimate via tiktoken). Every emitted metric states its
`source` so dashboards and quality gates can distinguish exact vs. estimated
cost.
"""

from __future__ import annotations

from typing import Any, Optional

from mlflow.entities import Feedback
from mlflow.genai.scorers import scorer

from config.settings import AgentConfig, settings


def _trace_usage(trace) -> Optional[dict[str, Any]]:
    """Pull exact token usage/cost off the MLflow trace, if present
    (requires MLflow >= 3.2 for token_usage, >= 3.10 for cost)."""
    if trace is None:
        return None
    info = getattr(trace, "info", None)
    token_usage = getattr(info, "token_usage", None) if info else None
    if not token_usage:
        return None
    # Sanity check: input_tokens == output_tokens == 0 while total_tokens > 0
    # is an impossible state for a real token count. MLflow's automatic
    # token-usage rollup can report exactly this if any span attribute name
    # happens to collide with its internal detection heuristics (observed
    # while building this framework -- see the comment in
    # eval/agent_client.py's make_predict_fn). Treat it as "not actually
    # available" and let the caller fall back to endpoint-reported usage.
    if not token_usage.get("input_tokens") and not token_usage.get("output_tokens") and token_usage.get("total_tokens"):
        return None
    cost = getattr(info, "cost", None)
    return {
        "input_tokens": token_usage.get("input_tokens"),
        "output_tokens": token_usage.get("output_tokens"),
        "total_tokens": token_usage.get("total_tokens"),
        "total_cost_usd": (cost or {}).get("total_cost"),
        "source": "mlflow_trace_exact",
    }


def _endpoint_usage(outputs: Any) -> Optional[dict[str, Any]]:
    if not isinstance(outputs, dict):
        return None
    usage = outputs.get("usage")
    if not usage or not usage.get("input_tokens"):
        return None
    return {
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "total_cost_usd": None,
        "source": outputs.get("usage_source", "endpoint_reported_or_estimated"),
    }


def build_latency_scorer():
    budget_ms = settings.quality_gates.max_p90_latency_ms

    @scorer
    def latency(outputs: Any) -> list[Feedback]:
        latency_ms = outputs.get("latency_ms") if isinstance(outputs, dict) else None
        if latency_ms is None:
            return [Feedback(name="latency_ms", error="No latency captured.")]
        return [
            Feedback(name="latency_ms", value=round(float(latency_ms), 1)),
            Feedback(
                name="latency_within_budget",
                value="yes" if latency_ms <= budget_ms else "no",
                rationale=f"{latency_ms:.0f}ms observed vs {budget_ms:.0f}ms budget.",
            ),
        ]

    return latency


def build_token_cost_scorer(agent_cfg: AgentConfig):
    pricing = agent_cfg.pricing

    @scorer
    def token_usage_and_cost(outputs: Any, trace=None) -> list[Feedback]:
        usage = _trace_usage(trace)
        fallback = _endpoint_usage(outputs)
        if usage is None:
            usage = fallback
        elif fallback is not None:
            # Trace usage can be partial; fill gaps from the endpoint/estimated counts.
            for key in ("input_tokens", "output_tokens", "total_tokens"):
                if usage.get(key) is None:
                    usage[key] = fallback[key]
        if usage is None:
            return [Feedback(name="total_tokens", error="No token usage available from trace or endpoint response.")]

        # MLflow rejects null feedback values that carry no error, so emit only populated fields.
        feedbacks = [
            Feedback(name=k, value=usage[k])
            for k in ("input_tokens", "output_tokens", "total_tokens")
            if usage.get(k) is not None
        ]

        if usage.get("total_cost_usd") is not None:
            cost = usage["total_cost_usd"]
            rationale = "Exact cost reported by MLflow trace."
        elif usage["input_tokens"] is not None and usage["output_tokens"] is not None:
            cost = (
                usage["input_tokens"] / 1_000_000 * pricing.input_usd_per_1m_tokens
                + usage["output_tokens"] / 1_000_000 * pricing.output_usd_per_1m_tokens
            )
            rationale = (
                f"Computed from {usage['source']} token counts using configured pricing "
                f"(${pricing.input_usd_per_1m_tokens}/1M in, "
                f"${pricing.output_usd_per_1m_tokens}/1M out)."
            )
        else:
            cost, rationale = None, "Insufficient data to compute cost."

        if cost is not None:
            feedbacks.append(Feedback(name="estimated_cost_usd", value=round(cost, 8), rationale=rationale))
        else:
            feedbacks.append(Feedback(name="estimated_cost_usd", error=rationale))
        feedbacks.append(Feedback(name="token_usage_source", value=usage["source"]))
        return feedbacks

    return token_usage_and_cost

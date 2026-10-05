"""
Thin REST client for invoking a Databricks Model Serving agent endpoint, plus
a `predict_fn` factory for `mlflow.genai.evaluate()`.

Works identically for local and Databricks execution: both environments call
the same deployed endpoint over HTTPS using a Databricks personal access
token (or an on-cluster service-principal token when run as a Databricks
job). This keeps "what we're evaluating" perfectly identical across both
run modes -- only *where the evaluation harness and MLflow tracking store
run* differs.
"""

from __future__ import annotations

import json
import os
import random
import time
from typing import Any, Optional

import mlflow
import requests

from config.settings import AgentConfig

try:
    import tiktoken

    _ENCODING = tiktoken.get_encoding("cl100k_base")
except ImportError:  # pragma: no cover - tiktoken is an optional dependency
    _ENCODING = None


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    if _ENCODING is not None:
        return len(_ENCODING.encode(text))
    # Crude fallback (~4 characters/token for English text) when tiktoken
    # isn't installed. Always tagged as "estimated" downstream.
    return max(1, len(text) // 4)


_MAX_RETRIES = 6


def _is_retryable(response: requests.Response) -> bool:
    if response.status_code in (429, 502, 503, 504):
        return True
    text = response.text
    return response.status_code == 400 and ("REQUEST_LIMIT_EXCEEDED" in text or "Error code: 429" in text)


class AgentEndpointClient:
    """Calls one Databricks Model Serving agent endpoint and normalizes its
    response, latency, and (best-effort) token usage."""

    def __init__(
        self,
        agent_cfg: AgentConfig,
        databricks_host: Optional[str] = None,
        databricks_token: Optional[str] = None,
        timeout_s: int = 120,
    ):
        self.agent_cfg = agent_cfg
        self.databricks_host = (
            databricks_host or os.environ.get("DATABRICKS_HOST", "")
        ).rstrip("/")
        self.databricks_token = databricks_token or os.environ.get("DATABRICKS_TOKEN")
        if not self.databricks_token:
            raise EnvironmentError(
                "DATABRICKS_TOKEN is not set. Export it (or put it in a .env file) "
                "before running evaluations -- see docs/RUNBOOK.md."
            )
        self.url = agent_cfg.resolved_endpoint_url(self.databricks_host)
        self.timeout_s = timeout_s

    def invoke(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        payload = {"messages": messages}
        headers = {
            "Authorization": f"Bearer {self.databricks_token}",
            "Content-Type": "application/json",
        }
        response, latency_ms = self._post_with_retry(payload, headers)
        response.raise_for_status()
        body = response.json()

        content = self._extract_content(body)
        usage, usage_source = self._extract_usage(body, messages, content)

        return {
            "response": content,
            "latency_ms": latency_ms,
            "raw_response": body,
            "usage": usage,
            "usage_source": usage_source,
        }

    def _post_with_retry(self, payload: dict[str, Any], headers: dict[str, str]):
        """POST with exponential backoff for rate limiting. Agent endpoints backed by
        pay-per-token foundation models surface the model's 429 (REQUEST_LIMIT_EXCEEDED)
        as an HTTP 400, so rate-limit detection must inspect the body, not just the status."""
        for attempt in range(_MAX_RETRIES + 1):
            start = time.perf_counter()
            response = requests.post(self.url, headers=headers, json=payload, timeout=self.timeout_s)
            latency_ms = (time.perf_counter() - start) * 1000.0
            if response.ok or attempt == _MAX_RETRIES or not _is_retryable(response):
                return response, latency_ms
            time.sleep(min(2 ** attempt, 30) + random.uniform(0, 1))
        return response, latency_ms

    @staticmethod
    def _extract_content(body: dict[str, Any]) -> str:
        messages = body.get("messages")
        if isinstance(messages, list) and messages:
            return messages[-1].get("content", "")
        predictions = body.get("predictions")
        if isinstance(predictions, list) and predictions:
            first = predictions[0]
            if isinstance(first, dict):
                return first.get("content") or json.dumps(first)
            return str(first)
        return json.dumps(body)

    def _extract_usage(
        self, body: dict[str, Any], messages: list[dict[str, str]], content: str
    ) -> tuple[dict[str, Any], str]:
        # 1) Endpoint explicitly reports usage (OpenAI-compatible or
        #    Databricks agent-framework responses sometimes include this).
        usage = body.get("usage") or (body.get("databricks_output") or {}).get("usage")
        if usage:
            return (
                {
                    "input_tokens": usage.get("prompt_tokens", usage.get("input_tokens")),
                    "output_tokens": usage.get("completion_tokens", usage.get("output_tokens")),
                    "total_tokens": usage.get("total_tokens"),
                },
                "endpoint_reported",
            )
        # 2) Fallback: local token estimate (tiktoken if available).
        input_text = "\n".join(m.get("content", "") for m in messages)
        in_tok = _estimate_tokens(input_text)
        out_tok = _estimate_tokens(content)
        return (
            {"input_tokens": in_tok, "output_tokens": out_tok, "total_tokens": in_tok + out_tok},
            "estimated_tiktoken",
        )


def make_predict_fn(agent_cfg: AgentConfig, client: AgentEndpointClient):
    """Returns a `predict_fn` compatible with `mlflow.genai.evaluate(data=...,
    predict_fn=...)`. The function is wrapped in `@mlflow.trace` so every
    evaluation row produces a trace (with latency/usage attached as span
    attributes) even though the agent's own internal tool-calling steps run
    remotely and aren't visible to this trace."""

    @mlflow.trace(name=f"{agent_cfg.name}_endpoint_call", span_type="AGENT")
    def predict_fn(messages: list[dict[str, str]]) -> dict[str, Any]:
        result = client.invoke(messages)
        try:
            span = mlflow.get_current_active_span()
            if span is not None:
                # NOTE: these keys are deliberately namespaced under
                # "databricks_ai_eval." -- generic names like
                # "usage.total_tokens" collide with MLflow's own token-usage
                # auto-detection heuristics (trace.info.token_usage), which
                # then reports bogus zeros for fields it doesn't recognize
                # instead of leaving token_usage unset. Confirmed while
                # building this framework: naming a span attribute
                # "usage.total_tokens" caused trace.info.token_usage to
                # report {"input_tokens": 0, "output_tokens": 0,
                # "total_tokens": <our value>}, which then shadowed our own
                # (correct) endpoint-reported usage in
                # scorers/cost_scorers.py's `_trace_usage` lookup.
                span.set_attributes(
                    {
                        "databricks_ai_eval.latency_ms": result["latency_ms"],
                        "databricks_ai_eval.usage_source": result["usage_source"],
                        "databricks_ai_eval.input_tokens": result["usage"]["input_tokens"],
                        "databricks_ai_eval.output_tokens": result["usage"]["output_tokens"],
                        "databricks_ai_eval.total_tokens": result["usage"]["total_tokens"],
                    }
                )
        except Exception:
            # Span attribute capture is best-effort; never fail the eval row
            # over it.
            pass
        # Keep the trace payload lean -- raw_response can be large/nested.
        return {k: v for k, v in result.items() if k != "raw_response"}

    return predict_fn

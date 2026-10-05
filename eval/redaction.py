"""
Lightweight secret-pattern redaction for any user/agent text that gets
persisted outside MLflow's own trace store -- specifically the `request`/
`response` columns written to Unity Catalog (`eval_results`) and surfaced on
the Lakeview dashboard's "failing examples" table.

Why this exists: a user could paste a credential, API key, or other secret
into a chat message sent to the evaluated agent. Without this step, that
text would be persisted verbatim into Delta tables and shown on a
dashboard that may have a broader viewer audience than the raw MLflow
experiment. This is defense-in-depth, not a guarantee -- it only catches
recognizable secret *shapes* (token prefixes, bearer headers, etc.), not
arbitrary sensitive text.

Known limitation: this redaction applies only to the copies we write to
Unity Catalog (`eval/uc_sync.py`). The original, unredacted text is still
captured by MLflow's own trace logging (local SQLite or the Databricks
experiment) as part of normal evaluate() behavior -- restrict access to
those accordingly if this matters for your use case.
"""

from __future__ import annotations

import re

# Each pattern matches a recognizable secret *shape*, not arbitrary secrets.
# Add more as needed; order doesn't matter since all are applied.
_SECRET_PATTERNS: list[re.Pattern] = [
    re.compile(r"\bdapi[0-9a-f]{32}\b", re.IGNORECASE),  # Databricks PAT
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),  # AWS access key id
    re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),  # OpenAI-style secret key
    re.compile(r"\bghp_[A-Za-z0-9]{36}\b"),  # GitHub personal access token
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9\-_.=]{20,}\b"),  # generic bearer tokens
    re.compile(r"(?i)\b(?:api[_-]?key|secret|password|token)\s*[:=]\s*['\"]?[A-Za-z0-9\-_.=]{8,}['\"]?"),
]

REDACTED_PLACEHOLDER = "[REDACTED]"
MAX_PERSISTED_LENGTH = 4000  # bound storage/display size regardless of redaction


def redact_secrets(text: str | None) -> str | None:
    """Replaces recognizable secret shapes with a placeholder and truncates
    overly long text before it's persisted to Unity Catalog or shown on the
    dashboard. Safe to call on None / non-string-coercible input."""
    if text is None:
        return None
    text = str(text)
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(REDACTED_PLACEHOLDER, text)
    if len(text) > MAX_PERSISTED_LENGTH:
        text = text[:MAX_PERSISTED_LENGTH] + "... [truncated]"
    return text

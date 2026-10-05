"""
Unified evaluation runner: evaluates one registered agent endpoint against
its MLflow evaluation dataset, using the shared scorer suite, and logs
results both to MLflow (always) and to the Unity Catalog metrics tables
(optional, --sync-uc).

Local execution:
    python -m eval.run_eval --agent vacation_planner_agent --run-source local

Databricks execution (notebook or job):
    python -m eval.run_eval --agent vacation_planner_agent --run-source databricks --sync-uc

See docs/RUNBOOK.md for full setup instructions for both modes.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402

from config.settings import settings  # noqa: E402
from eval.agent_client import AgentEndpointClient, make_predict_fn  # noqa: E402
from eval.redaction import redact_secrets  # noqa: E402
from eval.uc_sync import UCWriter, build_run_summary, new_run_id  # noqa: E402
from scorers.judges import build_scorer_suite  # noqa: E402


def _git_sha() -> Optional[str]:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return None


def load_local_dataset(agent_name: str, limit: Optional[int] = None) -> list[dict[str, Any]]:
    path = Path(__file__).resolve().parents[1] / "datasets" / f"{agent_name}_eval_set.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"No local dataset found at {path}. Generate it first with:\n"
            f"  python -m datasets.build_dataset --agent {agent_name}"
        )
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    # Always refresh calendar-dependent expected_facts at eval time (see
    # datasets/expectation_builder.py docstring for why).
    from datasets.expectation_builder import load_static_data, materialize_expectations

    agent_cfg = settings.agent(agent_name)
    static_data = load_static_data(agent_cfg.static_data_path)
    for row in rows:
        spec = row["expectations"]["expectation_spec"]
        row["expectations"].update(materialize_expectations({"expectation_spec": spec}, static_data))
    if limit:
        rows = rows[:limit]
    return rows


def load_uc_dataset(agent_name: str):
    import mlflow.genai.datasets as mlflow_datasets

    agent_cfg = settings.agent(agent_name)
    uc_table = settings.unity_catalog.fq_dataset_table(agent_cfg.eval_dataset_table)
    return mlflow_datasets.get_dataset(name=uc_table)


def _pin_local_eval_concurrency() -> None:
    """MLflow's evaluate() harness runs predict_fn + scorers for all rows
    concurrently via a thread pool (MLFLOW_GENAI_EVAL_MAX_WORKERS, default
    >1). A local SQLite-backed tracking store cannot safely serve that many
    concurrent trace/assessment writes and will deadlock (confirmed while
    building this framework: a 17-row eval hung indefinitely with the
    default worker count against sqlite:///mlflow.db, and completed in ~1s
    once pinned to 1). Databricks-hosted tracking has no such limitation, so
    this only applies to local runs, and only if the caller hasn't already
    set an explicit value."""
    os.environ.setdefault("MLFLOW_GENAI_EVAL_MAX_WORKERS", "1")


def configure_mlflow(run_source: str, agent_name: str) -> None:
    if run_source == "local":
        _pin_local_eval_concurrency()
    env = settings.mlflow_env(run_source)
    experiment_name = env.experiment_name_for(agent_name)
    mlflow.set_tracking_uri(env.tracking_uri)
    if run_source == "databricks":
        from mlflow.entities.trace_location import UnityCatalog

        # Agent and judge calls share the workspace foundation-model QPS limit; keep
        # parallelism modest (override with MLFLOW_GENAI_EVAL_MAX_WORKERS).
        os.environ.setdefault("MLFLOW_GENAI_EVAL_MAX_WORKERS", "2")
        # Traces stored in UC are read back through a SQL warehouse.
        if os.environ.get("DATABRICKS_SQL_WAREHOUSE_ID"):
            os.environ.setdefault("MLFLOW_TRACING_SQL_WAREHOUSE_ID", os.environ["DATABRICKS_SQL_WAREHOUSE_ID"])

        mlflow.set_experiment(
            experiment_name=experiment_name,
            trace_location=UnityCatalog(
                catalog_name=settings.unity_catalog.catalog,
                schema_name=settings.unity_catalog.schema,
                table_prefix=agent_name,
            ),
        )
    else:
        mlflow.set_experiment(experiment_name=experiment_name)


def _coerce_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_results_df(eval_results) -> pd.DataFrame:
    """Best-effort extraction of the per-row results table across MLflow
    versions: prefers `result_df`, falls back to `tables["eval_results"]`,
    falls back to reconstructing it via `mlflow.search_traces(run_id=...)`.
    All three return the same verified schema (one row per evaluated
    example) with columns including: trace_id, request, response,
    execution_duration, assessments -- see `build_detail_rows` below."""
    for attr in ("result_df",):
        df = getattr(eval_results, attr, None)
        if df is not None:
            return df
    tables = getattr(eval_results, "tables", None) or {}
    if "eval_results" in tables:
        return tables["eval_results"]
    return mlflow.search_traces(run_id=eval_results.run_id)


_PASS_VALUES = {"yes", "true", "correct", "1"}
_FAIL_VALUES = {"no", "false", "incorrect", "0"}


def _normalize_assessments(raw_assessments: Any) -> list[dict[str, Any]]:
    """Normalizes the `assessments` column (a list of dicts, verified against
    MLflow 3.16's `result_df` / `search_traces()` schema) into
    {scorer_name, value, rationale}. Entries without a `feedback` key are
    logged `expectations` (e.g. our own `expected_facts`), not scorer
    output, and are skipped here."""
    out = []
    for a in raw_assessments or []:
        feedback = a.get("feedback")
        if feedback is None:
            continue  # an `expectation`-type assessment, not scorer feedback
        out.append(
            {
                "scorer_name": a.get("assessment_name"),
                "value": feedback.get("value"),
                "rationale": a.get("rationale"),
            }
        )
    return out


def build_detail_rows(
    run_id: str,
    agent_name: str,
    rows: list[dict[str, Any]],
    results_df: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Returns (eval_results rows, eval_scores_long rows) built from the
    per-row results table, joined back to our dataset rows by position (the
    evaluate() harness preserves input order)."""
    now = datetime.now(timezone.utc)
    result_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []

    for idx, row in enumerate(rows):
        tags = row.get("tags", {})
        example_id = tags.get("example_id", f"row_{idx}")
        category = tags.get("category")
        spec = row["expectations"].get("expectation_spec", {})
        expected_behavior = spec.get("expected_behavior")

        if idx >= len(results_df):
            continue
        res_row = results_df.iloc[idx].to_dict()

        outputs = res_row.get("response")
        if isinstance(outputs, str):
            try:
                outputs = json.loads(outputs)
            except Exception:
                outputs = {"response": outputs}
        outputs = outputs or {}

        trace_id = res_row.get("trace_id")
        # `execution_duration` (ms) is MLflow's own measurement of the full
        # traced predict_fn call and is preferred over our self-reported
        # latency_ms; both should closely agree for a real endpoint call.
        mlflow_latency_ms = _coerce_float(res_row.get("execution_duration"))
        self_reported_latency_ms = _coerce_float(outputs.get("latency_ms")) if isinstance(outputs, dict) else None
        latency_ms = mlflow_latency_ms if mlflow_latency_ms else self_reported_latency_ms

        usage = outputs.get("usage", {}) if isinstance(outputs, dict) else {}
        response_text = outputs.get("response") if isinstance(outputs, dict) else str(outputs)

        assessments = _normalize_assessments(res_row.get("assessments"))
        pass_values = []
        estimated_cost = None
        for a in assessments:
            value_str = "" if a["value"] is None else str(a["value"]).lower()
            pass_fail = "pass" if value_str in _PASS_VALUES else "fail" if value_str in _FAIL_VALUES else None
            score_rows.append(
                {
                    "run_id": run_id,
                    "agent_name": agent_name,
                    "example_id": example_id,
                    "category": category,
                    "scorer_name": a["scorer_name"],
                    "value": None if a["value"] is None else str(a["value"]),
                    "numeric_value": _coerce_float(a["value"]),
                    "pass_fail": pass_fail,
                    "rationale": a["rationale"],
                    "created_at": now,
                    "run_date": now.date(),
                }
            )
            if pass_fail == "pass":
                pass_values.append(True)
            elif pass_fail == "fail":
                pass_values.append(False)
            if a["scorer_name"] == "estimated_cost_usd":
                estimated_cost = _coerce_float(a["value"])

        result_rows.append(
            {
                "run_id": run_id,
                "agent_name": agent_name,
                "example_id": example_id,
                "category": category,
                "trace_id": trace_id,
                # Redacted before persisting to Unity Catalog / the dashboard
                # (defense-in-depth against a user pasting a secret into a
                # chat message) -- see eval/redaction.py. The unredacted
                # text still exists in the MLflow trace itself; restrict
                # access to that store accordingly if this matters for you.
                "request": redact_secrets(row["inputs"]["messages"][-1]["content"]),
                "response": redact_secrets(response_text),
                "expected_behavior": expected_behavior,
                "latency_ms": latency_ms,
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "token_usage_source": outputs.get("usage_source") if isinstance(outputs, dict) else None,
                "estimated_cost_usd": estimated_cost,
                "overall_pass": all(pass_values) if pass_values else None,
                "created_at": now,
                "run_date": now.date(),
            }
        )

    return result_rows, score_rows


def run(
    agent_name: str,
    run_source: str,
    dataset_source: str = "local",
    limit: Optional[int] = None,
    sync_uc: bool = False,
    warehouse_id: Optional[str] = None,
    dry_run: bool = False,
    tags: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    agent_cfg = settings.agent(agent_name)
    run_tags = {"run_type": "dry_run" if dry_run else "scheduled", **(tags or {})}
    configure_mlflow(run_source, agent_name)

    if dataset_source == "local":
        rows = load_local_dataset(agent_name, limit=limit)
        data: Any = rows
        dataset_table = str(Path("datasets") / f"{agent_name}_eval_set.jsonl")
    else:
        dataset = load_uc_dataset(agent_name)
        data = dataset
        dataset_table = settings.unity_catalog.fq_dataset_table(agent_cfg.eval_dataset_table)
        rows = dataset.to_df().to_dict(orient="records")
        for r in rows:
            r.setdefault("tags", {})

    client = AgentEndpointClient(agent_cfg)
    predict_fn = make_predict_fn(agent_cfg, client)
    scorers = build_scorer_suite(agent_cfg, judge_model=settings.judge_model)

    started_at = datetime.now(timezone.utc)
    with mlflow.start_run(
        run_name=f"{agent_name}-{run_source}-{started_at.isoformat()}", tags=run_tags
    ) as mlflow_run:
        eval_results = mlflow.genai.evaluate(data=data, predict_fn=predict_fn, scorers=scorers)
    finished_at = datetime.now(timezone.utc)

    results_df = extract_results_df(eval_results)
    run_id = new_run_id()
    result_rows, score_rows = build_detail_rows(run_id, agent_name, rows, results_df)
    results_pdf = pd.DataFrame(result_rows)

    run_summary = build_run_summary(
        run_id=run_id,
        agent_name=agent_name,
        endpoint_name=agent_cfg.endpoint_name,
        dataset_table=dataset_table,
        run_source=run_source,
        results_df=results_pdf,
        mlflow_run_id=mlflow_run.info.run_id,
        mlflow_experiment_id=mlflow_run.info.experiment_id,
        started_at=started_at,
        finished_at=finished_at,
        triggered_by=os.environ.get("USER", "unknown"),
        git_sha=_git_sha(),
    )
    # UC has no tags column; `notes` carries the run type so cleanup/dashboards can filter on it.
    run_summary["notes"] = f"run_type={run_tags['run_type']}"

    print(json.dumps({k: str(v) for k, v in run_summary.items()}, indent=2))
    print(f"\nAggregated MLflow metrics: {getattr(eval_results, 'metrics', {})}")

    gates = settings.quality_gates
    violations = []
    if run_summary["overall_pass_rate"] is not None and run_summary["overall_pass_rate"] < gates.min_pass_rate:
        violations.append(
            f"pass_rate {run_summary['overall_pass_rate']:.2%} < required {gates.min_pass_rate:.2%}"
        )
    if run_summary["p90_latency_ms"] and run_summary["p90_latency_ms"] > gates.max_p90_latency_ms:
        violations.append(
            f"p90 latency {run_summary['p90_latency_ms']:.0f}ms > budget {gates.max_p90_latency_ms:.0f}ms"
        )
    if run_summary["avg_cost_usd"] and run_summary["avg_cost_usd"] > gates.max_avg_cost_usd:
        violations.append(
            f"avg cost ${run_summary['avg_cost_usd']:.4f} > budget ${gates.max_avg_cost_usd:.4f}"
        )
    if violations:
        warnings.warn("Quality gate violation(s):\n  - " + "\n  - ".join(violations))

    if sync_uc:
        writer = UCWriter(settings.unity_catalog, warehouse_id=warehouse_id)
        writer.write_run(run_summary)
        writer.write_results(result_rows)
        writer.write_scores_long(score_rows)
        print(
            f"Synced run {run_id} to {settings.unity_catalog.catalog}.{settings.unity_catalog.schema}"
        )

    return {"run_summary": run_summary, "violations": violations, "eval_results": eval_results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", default="vacation_planner_agent")
    parser.add_argument("--run-source", choices=["local", "databricks"], required=True)
    parser.add_argument("--dataset-source", choices=["local", "uc"], default="local")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N rows (dry run).")
    parser.add_argument("--sync-uc", action="store_true", help="Write results to Unity Catalog tables.")
    parser.add_argument("--warehouse-id", default=None, help="SQL warehouse id for local UC writes.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Tag this run run_type=dry_run so eval.cleanup_runs can delete it later.")
    parser.add_argument("--tag", action="append", default=[], metavar="KEY=VALUE",
                        help="Extra MLflow run tag (repeatable).")
    args = parser.parse_args()

    result = run(
        agent_name=args.agent,
        run_source=args.run_source,
        dataset_source=args.dataset_source,
        limit=args.limit,
        sync_uc=args.sync_uc,
        warehouse_id=args.warehouse_id,
        dry_run=args.dry_run,
        tags=dict(t.split("=", 1) for t in args.tag),
    )
    if result["violations"]:
        sys.exit(1)


if __name__ == "__main__":
    main()

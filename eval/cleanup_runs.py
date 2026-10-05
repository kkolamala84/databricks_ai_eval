"""Delete eval runs for one agent: dry-run-tagged runs only, or everything (fresh start).

Examples:
    python -m eval.cleanup_runs --agent vacation_planner_agent --run-source local --dry-runs-only
    python -m eval.cleanup_runs --agent vacation_planner_agent --run-source databricks --all --uc --traces --yes

Without --yes it only lists what would be deleted. The MLflow experiment itself is kept
(deleted experiment names stay reserved in the trash); only its runs are removed.
"""
from __future__ import annotations

import argparse
import os
import time

import mlflow
from dotenv import load_dotenv

from config.settings import settings
from eval.run_eval import configure_mlflow


def _delete_traces(exp_id: str, runs: list, dry_only: bool, apply: bool) -> None:
    """Deletes traces: every trace in the experiment for --all, otherwise only the traces
    linked to the matched dry runs. Must run before the runs are deleted."""
    client = mlflow.MlflowClient()
    try:
        if dry_only:
            ids = [
                t.info.trace_id
                for r in runs
                for t in mlflow.search_traces(run_id=r.info.run_id, locations=[exp_id], return_type="list")
            ]
            print(f"Traces: {len(ids)} trace(s) linked to matched runs")
            if apply and ids:
                for i in range(0, len(ids), 100):
                    client.delete_traces(experiment_id=exp_id, trace_ids=ids[i:i + 100])
        else:
            total = len(mlflow.search_traces(locations=[exp_id], return_type="list"))
            print(f"Traces: {total} trace(s) in the experiment")
            if apply and total:
                deleted = client.delete_traces(
                    experiment_id=exp_id, max_timestamp_millis=int(time.time() * 1000)
                )
                print(f"Traces deleted: {deleted}")
    except Exception as e:  # UC-stored traces may not support delete in every MLflow version
        print(f"Trace cleanup failed: {e}\n  Fallback: TRUNCATE the '<agent>_otel_*' tables in the UC schema "
              "(SHOW TABLES IN <catalog>.<schema> LIKE '<agent>*').")


def _delete_mlflow_runs(
    agent: str, run_source: str, dry_only: bool, apply: bool, traces: bool = False
) -> list[str]:
    configure_mlflow(run_source, agent)
    exp = mlflow.get_experiment_by_name(settings.mlflow_env(run_source).experiment_name_for(agent))
    if exp is None:
        print("No MLflow experiment found.")
        return []
    flt = "tags.run_type = 'dry_run'" if dry_only else ""
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], filter_string=flt, output_format="list")
    print(f"MLflow: {len(runs)} run(s) matched in experiment {exp.name}")
    if traces:
        _delete_traces(exp.experiment_id, runs, dry_only, apply)
    for r in runs:
        print(f"  {r.info.run_id}  {r.info.run_name}  run_type={r.data.tags.get('run_type')}")
        if apply:
            mlflow.delete_run(r.info.run_id)
    return [r.info.run_id for r in runs]


def _delete_uc_rows(agent: str, dry_only: bool, apply: bool, warehouse_id: str | None) -> None:
    from databricks import sql as dbsql

    warehouse_id = warehouse_id or os.environ.get("DATABRICKS_SQL_WAREHOUSE_ID")
    if not warehouse_id:
        raise SystemExit("Set --warehouse-id or DATABRICKS_SQL_WAREHOUSE_ID for UC cleanup")
    uc = settings.unity_catalog
    runs, results, scores = (uc.fq_table(k) for k in ("runs", "results", "scores_long"))
    cond = "agent_name = ?" + (" AND notes = 'run_type=dry_run'" if dry_only else "")
    host = os.environ["DATABRICKS_HOST"].replace("https://", "").rstrip("/")
    with dbsql.connect(server_hostname=host, http_path=f"/sql/1.0/warehouses/{warehouse_id}",
                       access_token=os.environ["DATABRICKS_TOKEN"]) as conn, conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {runs} WHERE {cond}", [agent])
        print(f"UC: {cur.fetchone()[0]} run row(s) matched in {runs}")
        if not apply:
            return
        for table in (results, scores):
            cur.execute(f"DELETE FROM {table} WHERE run_id IN (SELECT run_id FROM {runs} WHERE {cond})", [agent])
        cur.execute(f"DELETE FROM {runs} WHERE {cond}", [agent])
        print("UC rows deleted from eval_results, eval_scores_long, eval_runs.")


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agent", required=True)
    p.add_argument("--run-source", choices=["local", "databricks"], required=True)
    scope = p.add_mutually_exclusive_group(required=True)
    scope.add_argument("--dry-runs-only", action="store_true", help="Only runs tagged run_type=dry_run.")
    scope.add_argument("--all", action="store_true", help="Every run for the agent (fresh start).")
    p.add_argument("--traces", action="store_true",
                   help="Also delete MLflow traces (all traces for --all, linked traces for --dry-runs-only).")
    p.add_argument("--uc", action="store_true", help="Also delete the matching Unity Catalog rows.")
    p.add_argument("--warehouse-id")
    p.add_argument("--yes", action="store_true", help="Actually delete (default: preview only).")
    a = p.parse_args()

    _delete_mlflow_runs(a.agent, a.run_source, a.dry_runs_only, a.yes, a.traces)
    if a.uc:
        _delete_uc_rows(a.agent, a.dry_runs_only, a.yes, a.warehouse_id)
    if not a.yes:
        print("\nPreview only. Re-run with --yes to delete.")


if __name__ == "__main__":
    main()

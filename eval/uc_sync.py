"""
Writes `mlflow.genai.evaluate()` results into the Unity Catalog tables
defined in uc/ddl.sql, so the Databricks dashboard (dashboard/) has a stable,
query-friendly surface independent of the MLflow experiment UI.

Supports two write paths, auto-selected:
  * Databricks runtime (job/notebook): uses an active Spark session.
  * Local machine: uses the Databricks SQL connector against a SQL warehouse.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

from config.settings import UnityCatalogConfig


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _percentile(values: list[float], pct: float) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    idx = min(len(s) - 1, int(round(pct * (len(s) - 1))))
    return float(s[idx])


def build_run_summary(
    run_id: str,
    agent_name: str,
    endpoint_name: str,
    dataset_table: str,
    run_source: str,
    results_df: pd.DataFrame,
    mlflow_run_id: Optional[str],
    mlflow_experiment_id: Optional[str],
    started_at: datetime,
    finished_at: datetime,
    triggered_by: str,
    git_sha: Optional[str],
) -> dict[str, Any]:
    latencies = [v for v in results_df.get("latency_ms", pd.Series(dtype=float)).dropna().tolist()]
    costs = [v for v in results_df.get("estimated_cost_usd", pd.Series(dtype=float)).dropna().tolist()]
    passes = results_df.get("overall_pass", pd.Series(dtype=bool)).dropna().tolist()

    return {
        "run_id": run_id,
        "mlflow_run_id": mlflow_run_id,
        "mlflow_experiment_id": mlflow_experiment_id,
        "agent_name": agent_name,
        "endpoint_name": endpoint_name,
        "dataset_table": dataset_table,
        "dataset_version": None,
        "run_source": run_source,
        "triggered_by": triggered_by,
        "git_sha": git_sha,
        "num_examples": int(len(results_df)),
        "num_passed": int(sum(1 for p in passes if p)),
        "num_failed": int(sum(1 for p in passes if not p)),
        "overall_pass_rate": (sum(1 for p in passes if p) / len(passes)) if passes else None,
        "p50_latency_ms": _percentile(latencies, 0.5),
        "p90_latency_ms": _percentile(latencies, 0.9),
        "p99_latency_ms": _percentile(latencies, 0.99),
        "avg_cost_usd": (sum(costs) / len(costs)) if costs else None,
        "total_cost_usd": sum(costs) if costs else None,
        "total_input_tokens": int(results_df.get("input_tokens", pd.Series(dtype=float)).dropna().sum() or 0),
        "total_output_tokens": int(results_df.get("output_tokens", pd.Series(dtype=float)).dropna().sum() or 0),
        "total_tokens": int(results_df.get("total_tokens", pd.Series(dtype=float)).dropna().sum() or 0),
        "started_at": started_at,
        "finished_at": finished_at,
        "status": "SUCCESS",
        "notes": None,
        "run_date": started_at.date(),
    }


class UCWriter:
    """Writes pandas DataFrames to the eval_runs / eval_results /
    eval_scores_long Delta tables. Picks Spark (Databricks runtime) or the
    Databricks SQL connector (local) automatically."""

    def __init__(self, uc: UnityCatalogConfig, warehouse_id: Optional[str] = None):
        self.uc = uc
        self.warehouse_id = warehouse_id or os.environ.get("DATABRICKS_SQL_WAREHOUSE_ID")
        self._spark = None
        if "DATABRICKS_RUNTIME_VERSION" in os.environ:
            import pyspark

            self._spark = pyspark.sql.SparkSession.builder.getOrCreate()

    def write_run(self, run_row: dict[str, Any]) -> None:
        self._write_rows([run_row], self.uc.fq_table("runs"))

    def write_results(self, rows: list[dict[str, Any]]) -> None:
        self._write_rows(rows, self.uc.fq_table("results"))

    def write_scores_long(self, rows: list[dict[str, Any]]) -> None:
        self._write_rows(rows, self.uc.fq_table("scores_long"))

    # -- internals ------------------------------------------------------
    def _write_rows(self, rows: list[dict[str, Any]], fq_table: str) -> None:
        if not rows:
            return
        df = pd.DataFrame(rows)
        if self._spark is not None:
            int_cols = [
                f.name for f in self._spark.table(fq_table).schema.fields
                if f.dataType.simpleString() in _INT_TYPES and f.name in df.columns
            ]
            df = _coerce_int_columns(df, int_cols)
            self._spark.createDataFrame(df).write.mode("append").saveAsTable(fq_table)
        else:
            self._write_via_sql_connector(df, fq_table)

    def _write_via_sql_connector(self, df: pd.DataFrame, fq_table: str) -> None:
        from databricks import sql as dbsql

        if not self.warehouse_id:
            raise EnvironmentError(
                "DATABRICKS_SQL_WAREHOUSE_ID is not set. Required to write eval "
                "results to Unity Catalog from a local run -- see docs/RUNBOOK.md."
            )
        host = os.environ["DATABRICKS_HOST"].replace("https://", "").rstrip("/")
        token = os.environ["DATABRICKS_TOKEN"]
        http_path = f"/sql/1.0/warehouses/{self.warehouse_id}"

        columns = list(df.columns)
        placeholders = ", ".join(["?"] * len(columns))
        insert_sql = f"INSERT INTO {fq_table} ({', '.join(columns)}) VALUES ({placeholders})"

        with dbsql.connect(server_hostname=host, http_path=http_path, access_token=token) as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"DESCRIBE TABLE {fq_table}")
                int_cols = [r[0] for r in cursor.fetchall() if r[1] in _INT_TYPES and r[0] in columns]
                df = _coerce_int_columns(df, int_cols)
                records = [
                    {k: _to_native(v) for k, v in rec.items()}
                    for rec in df.to_dict(orient="records")
                ]
                for record in records:
                    values = [record[c] for c in columns]
                    cursor.execute(insert_sql, values)


_INT_TYPES = {"tinyint", "smallint", "int", "bigint"}


def _coerce_int_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Pandas upcasts int columns containing nulls to float; restore nullable ints."""
    df = df.copy()
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").round().astype("Int64")
    return df


def _to_native(v: Any) -> Any:
    """Convert numpy/pandas scalars (and NaN/NA) to plain Python values for the SQL connector."""
    if v is None or v is pd.NA or (isinstance(v, float) and v != v):
        return None
    if hasattr(v, "item"):
        return v.item()
    return v


def new_run_id() -> str:
    return str(uuid.uuid4())

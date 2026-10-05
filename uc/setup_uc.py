"""
Idempotently creates the Unity Catalog catalog/schema/tables used to track
evaluation metrics over time (see uc/ddl.sql).

Works in two modes:
  * Databricks notebook/job (a Spark session is already available) -- uses
    `spark.sql(...)` directly.
  * Local machine -- uses the Databricks SQL connector against a SQL
    warehouse (`DATABRICKS_SQL_WAREHOUSE_ID` or `--warehouse-id`).

Usage (local):
    python -m uc.setup_uc --warehouse-id <sql_warehouse_id>

Usage (Databricks notebook):
    %run ./uc/setup_uc.py   # or import and call main() programmatically
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import settings  # noqa: E402

DDL_PATH = Path(__file__).parent / "ddl.sql"


def _strip_leading_comment_lines(statement: str) -> str:
    """Drop leading `-- ...` comment-only lines from a statement chunk,
    keeping any SQL that follows them on later lines."""
    lines = statement.splitlines()
    start = 0
    while start < len(lines) and (not lines[start].strip() or lines[start].strip().startswith("--")):
        start += 1
    return "\n".join(lines[start:]).strip()


def _statements() -> list[str]:
    raw = DDL_PATH.read_text()
    filled = raw.format(catalog=settings.unity_catalog.catalog, schema=settings.unity_catalog.schema)
    # Split on semicolons that end a statement; DDL file has no semicolons
    # inside string literals other than the trailing ones, so this is safe.
    chunks = [_strip_leading_comment_lines(s) for s in filled.split(";")]
    return [s for s in chunks if s]


def run_with_spark() -> None:
    from databricks.connect import DatabricksSession  # type: ignore

    try:
        import pyspark  # noqa: F401

        spark = pyspark.sql.SparkSession.builder.getOrCreate()
    except Exception:
        spark = DatabricksSession.builder.serverless(True).getOrCreate()

    for stmt in _statements():
        print(f"Executing:\n{stmt[:120]}...")
        spark.sql(stmt)
    print(
        f"Done. Tables ready under {settings.unity_catalog.catalog}.{settings.unity_catalog.schema}"
    )


def run_with_sql_connector(warehouse_id: str) -> None:
    from databricks import sql as dbsql

    host = os.environ["DATABRICKS_HOST"].replace("https://", "").rstrip("/")
    token = os.environ["DATABRICKS_TOKEN"]
    http_path = f"/sql/1.0/warehouses/{warehouse_id}"

    with dbsql.connect(server_hostname=host, http_path=http_path, access_token=token) as conn:
        with conn.cursor() as cursor:
            for stmt in _statements():
                print(f"Executing:\n{stmt[:120]}...")
                cursor.execute(stmt)
    print(
        f"Done. Tables ready under {settings.unity_catalog.catalog}.{settings.unity_catalog.schema}"
    )


def in_databricks_runtime() -> bool:
    return "DATABRICKS_RUNTIME_VERSION" in os.environ


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--warehouse-id",
        default=os.environ.get("DATABRICKS_SQL_WAREHOUSE_ID"),
        help="SQL warehouse id to use when running from a local machine.",
    )
    args = parser.parse_args()

    if in_databricks_runtime():
        run_with_spark()
    else:
        if not args.warehouse_id:
            raise SystemExit(
                "Running locally: pass --warehouse-id or set "
                "DATABRICKS_SQL_WAREHOUSE_ID. Find it in Databricks under "
                "SQL Warehouses > <warehouse> > Connection details."
            )
        run_with_sql_connector(args.warehouse_id)


if __name__ == "__main__":
    main()

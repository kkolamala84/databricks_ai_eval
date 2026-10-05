"""
Upserts every agent defined in config/agents.yaml into the
<catalog>.<schema>.agent_registry Unity Catalog table, so the dashboard can
discover and filter by agent without code changes.

Usage:
    python -m uc.seed_agent_registry                     # on Databricks
    python -m uc.seed_agent_registry --warehouse-id <id>  # from a local machine
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import settings  # noqa: E402
from uc.setup_uc import in_databricks_runtime  # noqa: E402


def build_rows() -> list[dict]:
    now = datetime.now(timezone.utc)
    rows = []
    for agent in settings.agents.values():
        rows.append(
            {
                "agent_name": agent.name,
                "endpoint_name": agent.endpoint_name,
                "endpoint_url": agent.endpoint_url or None,
                "tools": agent.tools,
                "supported_locations": agent.supported_locations,
                "owner": os.environ.get("USER", "unknown"),
                "registered_at": now,
                "updated_at": now,
            }
        )
    return rows


def upsert_with_spark(rows: list[dict]) -> None:
    import pyspark

    spark = pyspark.sql.SparkSession.builder.getOrCreate()
    fq_table = settings.unity_catalog.fq_table("agent_registry")
    df = spark.createDataFrame(rows)
    df.createOrReplaceTempView("_agent_registry_updates")
    spark.sql(
        f"""
        MERGE INTO {fq_table} AS target
        USING _agent_registry_updates AS source
        ON target.agent_name = source.agent_name
        WHEN MATCHED THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
        """
    )
    print(f"Upserted {len(rows)} agent(s) into {fq_table}")


def upsert_with_sql_connector(rows: list[dict], warehouse_id: str) -> None:
    from databricks import sql as dbsql

    fq_table = settings.unity_catalog.fq_table("agent_registry")
    host = os.environ["DATABRICKS_HOST"].replace("https://", "").rstrip("/")
    token = os.environ["DATABRICKS_TOKEN"]
    http_path = f"/sql/1.0/warehouses/{warehouse_id}"

    with dbsql.connect(server_hostname=host, http_path=http_path, access_token=token) as conn:
        with conn.cursor() as cursor:
            for row in rows:
                cursor.execute(f"DELETE FROM {fq_table} WHERE agent_name = ?", [row["agent_name"]])
                cursor.execute(
                    f"""
                    INSERT INTO {fq_table}
                    (agent_name, endpoint_name, endpoint_url, tools, supported_locations,
                     owner, registered_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        row["agent_name"],
                        row["endpoint_name"],
                        row["endpoint_url"],
                        row["tools"],
                        row["supported_locations"],
                        row["owner"],
                        row["registered_at"],
                        row["updated_at"],
                    ],
                )
    print(f"Upserted {len(rows)} agent(s) into {fq_table}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warehouse-id", default=os.environ.get("DATABRICKS_SQL_WAREHOUSE_ID"))
    args = parser.parse_args()

    rows = build_rows()
    if in_databricks_runtime():
        upsert_with_spark(rows)
    else:
        if not args.warehouse_id:
            raise SystemExit("Pass --warehouse-id or set DATABRICKS_SQL_WAREHOUSE_ID.")
        upsert_with_sql_connector(rows, args.warehouse_id)


if __name__ == "__main__":
    main()

"""Create or update the eval Lakeview dashboard through the Databricks API.

Usage:
    python -m dashboard.deploy_dashboard [--warehouse-id ID] [--parent-path /Shared] [--no-publish]

Reads DATABRICKS_HOST / DATABRICKS_TOKEN / DATABRICKS_SQL_WAREHOUSE_ID from the environment or .env.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.dashboards import Dashboard
from dotenv import load_dotenv

DASHBOARD_JSON = Path(__file__).parent / "eval_dashboard.lvdash.json"
DISPLAY_NAME = "Agent Evaluation Monitoring"


def deploy(warehouse_id: str, parent_path: str, publish: bool = True) -> str:
    w = WorkspaceClient()
    serialized = json.dumps(json.loads(DASHBOARD_JSON.read_text()))
    path = f"{parent_path.rstrip('/')}/{DISPLAY_NAME}.lvdash.json"

    try:
        existing = w.workspace.get_status(path)
        dash = w.lakeview.update(
            existing.resource_id,
            Dashboard(display_name=DISPLAY_NAME, serialized_dashboard=serialized, warehouse_id=warehouse_id),
        )
        print(f"Updated dashboard {dash.dashboard_id}")
    except NotFound:
        dash = w.lakeview.create(
            Dashboard(
                display_name=DISPLAY_NAME,
                serialized_dashboard=serialized,
                warehouse_id=warehouse_id,
                parent_path=parent_path,
            )
        )
        print(f"Created dashboard {dash.dashboard_id}")

    if publish:
        w.lakeview.publish(dash.dashboard_id, embed_credentials=True, warehouse_id=warehouse_id)
        print("Published (viewers use the publisher's credentials for queries)")
    host = w.config.host.rstrip("/")
    url = f"{host}/dashboardsv3/{dash.dashboard_id}"
    print(url)
    return url


if __name__ == "__main__":
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--warehouse-id", default=os.getenv("DATABRICKS_SQL_WAREHOUSE_ID"))
    p.add_argument("--parent-path", default="/Shared")
    p.add_argument("--no-publish", action="store_true")
    a = p.parse_args()
    if not a.warehouse_id:
        raise SystemExit("Set --warehouse-id or DATABRICKS_SQL_WAREHOUSE_ID")
    deploy(a.warehouse_id, a.parent_path, not a.no_publish)

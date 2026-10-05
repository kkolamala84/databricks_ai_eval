# Databricks notebook source
# MAGIC %md
# MAGIC # Agent Evaluation — `vacation_planner_agent`
# MAGIC
# MAGIC Runs the full evaluation suite (tool-coverage dataset + code-based scorers +
# MAGIC LLM judges) against a deployed agent serving endpoint, logs to a
# MAGIC Unity Catalog-backed MLflow experiment, and syncs results to the
# MAGIC `krishna.agent_eval` metrics tables that back the dashboard.
# MAGIC
# MAGIC See `docs/RUNBOOK.md` and `docs/ARCHITECTURE.md` in the repo for full details.

# COMMAND ----------

# MAGIC %pip install -q --upgrade "mlflow[databricks]>=3.14.0" pandas requests tiktoken python-dotenv PyYAML
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("agent_name", "vacation_planner_agent", "Agent name")
dbutils.widgets.text("secret_scope", "", "Secret scope for DATABRICKS_TOKEN (optional)")
dbutils.widgets.text("secret_key", "databricks-token", "Secret key for DATABRICKS_TOKEN")
dbutils.widgets.text("repo_path", "", "Path to databricks_ai_eval/ (e.g. /Workspace/Repos/you/databricks-ai-eval-ws/databricks_ai_eval)")
dbutils.widgets.text("dataset_source", "local", "Dataset source: local or uc")
dbutils.widgets.text("limit", "", "Optional: limit to first N rows (dry run)")

# COMMAND ----------

import os
import sys

agent_name = dbutils.widgets.get("agent_name")
repo_path = dbutils.widgets.get("repo_path")
secret_scope = dbutils.widgets.get("secret_scope")
secret_key = dbutils.widgets.get("secret_key")
dataset_source = dbutils.widgets.get("dataset_source")
limit_str = dbutils.widgets.get("limit")
limit = int(limit_str) if limit_str.strip() else None

if repo_path:
    sys.path.insert(0, repo_path)

# DATABRICKS_HOST is auto-available in a notebook context; DATABRICKS_TOKEN
# should come from a secret scope rather than being pasted in plaintext.
if "DATABRICKS_HOST" not in os.environ:
    ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
    os.environ["DATABRICKS_HOST"] = ctx.apiUrl().get()
if secret_scope:
    os.environ["DATABRICKS_TOKEN"] = dbutils.secrets.get(secret_scope, secret_key)

print(f"Agent: {agent_name}")
print(f"DATABRICKS_HOST: {os.environ.get('DATABRICKS_HOST')}")
print(f"Dataset source: {dataset_source}, limit: {limit}")

# COMMAND ----------

from eval.run_eval import run

result = run(
    agent_name=agent_name,
    run_source="databricks",
    dataset_source=dataset_source,
    limit=limit,
    sync_uc=True,
)

print("\n=== Run summary ===")
for k, v in result["run_summary"].items():
    print(f"{k}: {v}")

print("\n=== Aggregated MLflow metrics ===")
print(result["eval_results"].metrics)

if result["violations"]:
    print("\n=== QUALITY GATE VIOLATIONS ===")
    for v in result["violations"]:
        print(f" - {v}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next steps
# MAGIC - Open the **Experiments** tab to inspect per-example traces and scorer feedback.
# MAGIC - Query `krishna.agent_eval.eval_runs` / `eval_results` / `eval_scores_long`
# MAGIC   directly, or open the Lakeview dashboard
# MAGIC   (`dashboard/eval_dashboard.lvdash.json`) for trend charts.
# MAGIC - To fail a job on a quality-gate violation, raise at the end of this
# MAGIC   notebook: `if result["violations"]: raise SystemExit(result["violations"])`.

# COMMAND ----------

if result["violations"]:
    raise SystemExit(f"Quality gate violation(s): {result['violations']}")

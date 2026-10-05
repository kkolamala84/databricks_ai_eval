# Runbook — Running Evaluations Locally and on Databricks

See [ARCHITECTURE.md](./ARCHITECTURE.md) for *why* the framework is built
this way. This document is the step-by-step *how*.

## 0. Prerequisites

- Python 3.10+ (the framework was built and tested against 3.12;
  `mlflow[databricks]>=3.14.0` requires 3.10+).
- A Databricks workspace with:
  - The `vacation_planner_agent` serving endpoint deployed and reachable
    (`CAN QUERY` permission on it).
  - `CREATE TABLE` on the target Unity Catalog schema (default:
    `krishna.agent_eval`; change via `config/eval_config.yaml` or
    `EVAL_UC_CATALOG`/`EVAL_UC_SCHEMA` env vars).
  - A running SQL warehouse (for local -> Unity Catalog writes only; not
    needed when running inside Databricks).
- A Databricks personal access token (or service-principal token for CI).

## 1. Install

```bash
cd databricks_ai_eval
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Configure secrets

```bash
cp .env.template.example .env
```

Edit `.env` and fill in at minimum:

```ini
DATABRICKS_HOST=https://dbc-08ff8153-2dcb.cloud.databricks.com
DATABRICKS_TOKEN=<your personal access token>
DATABRICKS_SQL_WAREHOUSE_ID=<warehouse id, only needed for local --sync-uc>
```

`.env` is loaded automatically (via `python-dotenv`) by every script in this
project and is already excluded from git via `.gitignore`. See
`.env.template.example` for a fully documented reference of every variable.

## 3. Provision Unity Catalog tables (one-time, or after a DDL change)

From a local machine (uses a SQL warehouse):

```bash
python -m uc.setup_uc --warehouse-id "$DATABRICKS_SQL_WAREHOUSE_ID"
python -m uc.seed_agent_registry --warehouse-id "$DATABRICKS_SQL_WAREHOUSE_ID"
```

From inside a Databricks notebook (uses the notebook's Spark session):

```python
%run ./uc/setup_uc.py
%run ./uc/seed_agent_registry.py
```

This creates (if not already present):
`krishna.agent_eval.{agent_registry, eval_runs, eval_results, eval_scores_long}`.

## 4. Generate the evaluation dataset

```bash
# Local JSONL only (fast, no Databricks access needed):
python -m datasets.build_dataset --agent vacation_planner_agent

# Also create/update the Unity Catalog-backed MLflow evaluation dataset
# (needs CREATE TABLE on the target schema):
python -m datasets.build_dataset --agent vacation_planner_agent --push-to-uc
```

This writes `datasets/vacation_planner_agent_eval_set.jsonl` (17 rows
covering every tool individually, every tool combination, and graceful
degradation edge cases -- see ARCHITECTURE.md §3). Re-run this whenever you
add a city/tool to `config/agents.yaml`, or want to resync `--push-to-uc`
(the dataset itself doesn't need regenerating monthly -- calendar-dependent
facts are recomputed at evaluation time automatically).

## 5. Run locally

```bash
export $(grep -v '^#' .env | xargs)   # or just `source .env` in bash, or rely on python-dotenv

# Dry run on 3 rows first -- catches broken endpoint auth / misconfigured
# scorers before spending time/tokens on the full dataset:
python -m eval.run_eval --agent vacation_planner_agent --run-source local --limit 3

# Full local run, logging to sqlite:///mlflow.db (see config/eval_config.yaml):
python -m eval.run_eval --agent vacation_planner_agent --run-source local

# Full local run AND sync results to Unity Catalog:
python -m eval.run_eval --agent vacation_planner_agent --run-source local --sync-uc
```

View results:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# open http://127.0.0.1:5000, find /local/agent_eval/<agent_name>
```

The command exits non-zero if a quality gate from
`config/eval_config.yaml: quality_gates` is violated (pass rate, p90
latency, avg cost) -- convenient for CI.

## 6. Run on Databricks

Run as a notebook cell or a Databricks Job task (Python script task
pointing at `eval/run_eval.py`):

```python
%pip install -q --upgrade "mlflow[databricks]>=3.14.0" pandas requests tiktoken python-dotenv
dbutils.library.restartPython()
```

```python
import os
os.environ["DATABRICKS_HOST"] = "https://dbc-08ff8153-2dcb.cloud.databricks.com"
os.environ["DATABRICKS_TOKEN"] = dbutils.secrets.get("your-scope", "databricks-token")
# DATABRICKS_SQL_WAREHOUSE_ID is not needed here -- UC writes use Spark directly.

import sys
sys.path.insert(0, "/Workspace/Repos/<you>/databricks-ai-eval-ws/databricks_ai_eval")
from eval.run_eval import run

result = run(
    agent_name="vacation_planner_agent",
    run_source="databricks",
    dataset_source="local",   # or "uc" to pull from the Unity Catalog dataset
    sync_uc=True,
)
print(result["run_summary"])
if result["violations"]:
    raise SystemExit(f"Quality gates failed: {result['violations']}")
```

Or from the command line on a cluster with Databricks Connect / in a job
using the `%python` entry point:

```bash
python -m eval.run_eval --agent vacation_planner_agent --run-source databricks --sync-uc
```

Traces and metrics log to a stable Databricks MLflow experiment per agent at
`/Shared/<agent_name>` (for example,
`/Shared/vacation_planner_agent`), with trace storage in Unity
Catalog under `krishna.agent_eval` (table prefix = agent name). Each daily
evaluation creates a new run in the same agent experiment.

### Scheduling

Wrap the Databricks snippet above in a Databricks Job (daily/hourly
schedule, or triggered after each agent redeployment) with task parameter
`--sync-uc` always on, so `eval_runs` / `eval_results` / `eval_scores_long`
accumulate a continuous history for the dashboard.

## 7. Dashboard

**Via API (recommended, repeatable):**

```bash
python -m dashboard.deploy_dashboard            # uses DATABRICKS_SQL_WAREHOUSE_ID
python -m dashboard.deploy_dashboard --parent-path /Users/<you>@<domain> --no-publish
```

It creates (or updates, if it already exists) the Lakeview dashboard
"Agent Evaluation Monitoring" from `eval_dashboard.lvdash.json` through the
Lakeview API, publishes it, and prints the URL. Re-run it after editing the JSON.

**Via UI:**

1. In Databricks, go to **Dashboards > Create dashboard > Import dashboard
   file** and upload `dashboard/eval_dashboard.lvdash.json`.
2. If any dataset query fails to bind automatically (Lakeview's JSON schema
   has changed slightly across Databricks releases), open
   `dashboard/dashboard_queries.sql` and paste the relevant query into a new
   dataset manually -- every widget's SQL is there, independently runnable
   and documented.
3. Update `{catalog}.{schema}` references if you changed
   `config/eval_config.yaml` away from the `krishna.agent_eval` default.
4. Re-run step 5 or 6 a few times (with `--sync-uc`) to get enough history
   for the trend charts and the deviation-detection widget (which needs at
   least ~6 runs for a meaningful rolling baseline).

## 8. Adding another agent

See ARCHITECTURE.md §9. In short: add it to `config/agents.yaml`, generate
its dataset, seed the registry, run.

## 9. Troubleshooting

**Local run hangs indefinitely (no output after "Evaluating: 0%...").**
This was a real issue found while building this framework: MLflow's
`evaluate()` harness parallelizes across rows by default, and a local SQLite
tracking store deadlocks under concurrent writes. `eval/run_eval.py` already
pins `MLFLOW_GENAI_EVAL_MAX_WORKERS=1` for `run_source=local` automatically.
If you're calling `mlflow.genai.evaluate()` directly (bypassing
`eval/run_eval.py`) against a sqlite tracking URI, set this env var yourself
before calling it.

**Many rows fail with empty responses, or `latency` / `token_usage_and_cost` "failed" on some rows.**
The agent's foundation model (e.g. pay-per-token `databricks-meta-llama-3-3-70b-instruct`)
hits the workspace QPS limit under parallel eval calls; the endpoint returns this as
HTTP 400 with `REQUEST_LIMIT_EXCEEDED`. `eval/agent_client.py` retries with exponential
backoff, and Databricks runs default to `MLFLOW_GENAI_EVAL_MAX_WORKERS=2` (override via env).
Use provisioned throughput for higher parallelism. Latency budget failures in a run can
partly reflect this throttling.

**Expected-fact scoring.** `tool_fact_coverage` matches atomic `expected_key_facts`
(temperature, condition words, best months; month ranges like "November to February" are
expanded) instead of verbatim tool strings, because the agent paraphrases tool output.

**`EnvironmentError: DATABRICKS_TOKEN is not set`.** Your `.env` isn't
populated or wasn't picked up -- confirm it's at
`databricks_ai_eval/.env` (not project root) and that you didn't rename the
keys.

**LLM judge scorers (`safety`, `correctness`, `relevance_to_query`,
`no_fabricated_destinations`, `actionable_vacation_recommendation`,
`tool_usage_validator`) all fail, but the code-based scorers
(`tool_fact_coverage`, `latency`, `token_usage_and_cost`) succeed.** This
means `DATABRICKS_HOST`/`DATABRICKS_TOKEN` can't reach the managed judge
model (invalid/expired token, or the workspace doesn't have judge access
enabled). Code-based scorers never need judge-model access, so they
continuing to work is expected; fix the credentials for the judges.

**`estimated_cost_usd` / token counts show as 0 even though the endpoint is
responding.** Check `token_usage_source` in `eval_results` -- if it says
anything other than `endpoint_reported` or `estimated_tiktoken`, the trace's
auto-detected `token_usage` may be interfering (see ARCHITECTURE.md §5's
"span-attribute name collisions" note). This is defended against in
`scorers/cost_scorers.py`, but if you add your own custom span attributes to
`eval/agent_client.py`, avoid names containing `token`/`usage` patterns that
might collide with MLflow's internal heuristics.

**Unity Catalog writes fail locally with an auth or warehouse error.**
Confirm `DATABRICKS_SQL_WAREHOUSE_ID` is set and the warehouse is running
(cold warehouses take ~30-60s to start on first query).

**Dataset facts don't match the agent's actual answer even though the tool
was called correctly.** Weather facts are calendar-dependent (see
ARCHITECTURE.md §3) -- `tool_fact_coverage` recomputes "this month's"
weather fresh on every run using `datetime.date.today()`. If your system
clock/timezone differs from the Databricks cluster's, the two could pick
different months right at a month boundary; this is a known, narrow edge
case (impacts at most the last/first day of a month).

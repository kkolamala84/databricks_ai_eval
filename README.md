# databricks_ai_eval

A reusable MLflow 3 / Databricks GenAI evaluation framework for Databricks
Model Serving agent endpoints — starting with `vacation_planner_agent`, and
extensible to any number of additional agents via `config/agents.yaml`.

**Start here:** [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md) (design &
rationale) and [docs/RUNBOOK.md](./docs/RUNBOOK.md) (step-by-step: install,
configure, run locally, run on Databricks, import the dashboard).

## Layout

| Path | What |
|---|---|
| `config/` | Agent registry (`agents.yaml`) + framework settings (`eval_config.yaml`, `settings.py`) |
| `datasets/` | Tool-coverage evaluation dataset generator + calendar-aware ground truth |
| `scorers/` | Code-based scorers (tool coverage, latency, token cost) + LLM judges |
| `eval/` | Agent endpoint client + the unified local/Databricks evaluation runner |
| `uc/` | Unity Catalog DDL + setup scripts for the metrics tables |
| `dashboard/` | Lakeview dashboard + backing SQL queries (latency, cost, deviations) |
| `notebooks/` | Ready-to-import Databricks notebook to run evaluations |
| `docs/` | Architecture and runbook documentation |
| `agents/`, `data/` | The agent source (for reference) and its static tool data |

## Quick start

```bash
cd databricks_ai_eval
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.template.example .env   # fill in DATABRICKS_HOST / DATABRICKS_TOKEN
python -m uc.setup_uc           # uses DATABRICKS_SQL_WAREHOUSE_ID from .env
python -m datasets.build_dataset --agent vacation_planner_agent
python -m eval.run_eval --agent vacation_planner_agent --run-source local --limit 3
```

For local setup commands that access Unity Catalog, set
`DATABRICKS_SQL_WAREHOUSE_ID` in `.env` to the SQL warehouse ID (the final
segment of the warehouse's HTTP path). `uc.setup_uc` and
`uc.seed_agent_registry` read it automatically. You can also pass it
explicitly with `--warehouse-id <id>` to override the `.env` value.

## View local results in the MLflow UI

After a local evaluation, start the MLflow UI from the `databricks_ai_eval`
directory using the same SQLite tracking database configured for local runs:

```bash
source .venv/bin/activate
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) and select the
`/local/agent_eval/<agent_name>` experiment (for example,
`/local/agent_eval/vacation_planner_agent`) to inspect runs,
evaluation metrics, traces, and per-example scorer feedback. Keep the
terminal process running while using the UI; press `Ctrl+C` to stop it.

Full instructions (including running on Databricks and importing the
dashboard): [docs/RUNBOOK.md](./docs/RUNBOOK.md).

## Dashboard (Databricks)

The Lakeview dashboard reads the Unity Catalog tables (`krishna.agent_eval.*`), not MLflow,
so runs must be synced with `--sync-uc`.

**Prerequisites:** tables created (`python -m uc.setup_uc`), at least one run with `--sync-uc`
(about 6+ runs for the deviation widget), and a running SQL warehouse.

**Deploy via API (recommended):**

```bash
python -m dashboard.deploy_dashboard                       # uses DATABRICKS_SQL_WAREHOUSE_ID
python -m dashboard.deploy_dashboard --parent-path /Users/<you>@<domain> --no-publish
```

This creates or updates the "Agent Evaluation Monitoring" dashboard from
`dashboard/eval_dashboard.lvdash.json`, publishes it (viewers use the publisher's credentials),
and prints the URL. Re-run it after editing the JSON.

**Deploy via UI:** Dashboards > Create > Import dashboard file > upload the JSON. Fallback: add each
query in `dashboard/dashboard_queries.sql` as a dataset (replace `{catalog}.{schema}`) and build the widgets manually.

**Widgets:** latest pass rate / p90 latency / avg cost counters; latency percentiles, token cost and
scorer pass-rate trends; category breakdown, deviation vs baseline and failing-examples tables.

**Widget JSON requirements** (Lakeview renders nothing if these are missing):
- every widget needs a `queries` entry named `main_query` (dataset + backtick-quoted `fields`);
- `encodings` must be populated (counter: `value`; line: `x`/`y`/`color`; table: `columns`);
- spec versions: counter and table = 2, line/bar/pie = 3.

**Known limitations:** the dashboard does not auto-refresh (refresh manually or schedule it after the daily run);
"latest run" widgets use the global latest run across agents; the JSON was authored by hand, so if a widget
fails, build it once in the UI, export it, and align the JSON.

Full details: [docs/RUNBOOK.md](docs/RUNBOOK.md) §7.

## Dry runs and resetting evals

Tag throwaway runs so they can be deleted later without touching real history:

```bash
python -m eval.run_eval --agent vacation_planner_agent --run-source local --limit 3 --dry-run
python -m eval.run_eval --agent vacation_planner_agent --run-source databricks --dry-run --sync-uc --tag purpose=smoke
```

`--dry-run` sets the MLflow run tag `run_type=dry_run` (normal runs get `run_type=scheduled`) and writes
`notes = 'run_type=dry_run'` to `eval_runs` when synced. `--tag KEY=VALUE` adds any extra MLflow tag.

Clean up with `eval.cleanup_runs` (preview by default; add `--yes` to delete):

```bash
# delete only dry runs (MLflow runs; add --uc for the Unity Catalog rows)
python -m eval.cleanup_runs --agent vacation_planner_agent --run-source local --dry-runs-only --yes
python -m eval.cleanup_runs --agent vacation_planner_agent --run-source databricks --dry-runs-only --uc --traces --yes

# fresh start for an agent: delete every run
python -m eval.cleanup_runs --agent vacation_planner_agent --run-source databricks --all --uc --traces --yes
```

Notes: the MLflow experiment itself is kept (a deleted experiment's name stays reserved in the trash);
only its runs are removed, and deleted runs are soft-deleted until `mlflow gc`. For a complete local reset
you can also delete `mlflow.db` and `mlruns/`. Add `--traces` to also delete MLflow traces (and the UC-stored trace data behind them); without it
traces are kept. Only runs synced with `--sync-uc` exist in UC, so MLflow may show more runs than UC.

## References

Documentation consulted while designing this framework. Each design
decision in [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md) traces back to
one or more of these.

### Databricks documentation

- [Agents overview](https://docs.databricks.com/aws/en/agents/) — build,
  evaluate, and deploy agents on Databricks.
- [Scorers (concepts)](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/concepts/scorers) —
  how scorers work, built-in vs. custom LLM judges vs. code-based scorers,
  model selection for judges.
- [Built-in LLM judges](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/concepts/judges/) —
  the predefined judges (Safety, Correctness, RelevanceToQuery, Guidelines,
  RetrievalGroundedness, ...) and multi-turn judges.
- [Custom judges (`make_judge()`)](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/custom-judge/) —
  field-based vs. trace-based judges, template variables (`{{ inputs }}`,
  `{{ outputs }}`, `{{ expectations }}`, `{{ trace }}`), model requirements
  for trace-based judges — basis for `scorers/judges.py::build_tool_usage_judge`.
- [Code-based scorers (concepts)](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/custom-scorers) —
  `@scorer` decorator, production-monitoring constraints.
- [Code-based scorer examples](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/code-based-scorer-examples) —
  accessing `Trace`/`Feedback`/`SpanType`, wrapping a predefined judge,
  using `expectations`, returning multiple `Feedback` objects, using a
  custom LLM for judging — directly informed `scorers/tool_scorers.py` and
  `scorers/judges.py`'s wrapping pattern.
- [Evaluate GenAI apps during development (tutorial)](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/evaluate-app) —
  end-to-end `mlflow.genai.evaluate()` workflow, Unity Catalog trace
  location setup, `mlflow.search_traces()` — basis for
  `eval/run_eval.py::configure_mlflow`.
- [Building MLflow evaluation datasets](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/build-eval-dataset) —
  `mlflow.genai.datasets.create_dataset()` / `get_dataset()` /
  `merge_records()`, dataset size/expectation limits — basis for
  `datasets/build_dataset.py::push_to_unity_catalog`.

### MLflow documentation (linked from / complementary to the above)

- [MLflow Tracing overview](https://mlflow.org/docs/latest/genai/tracing/observe-with-traces/) —
  `@mlflow.trace`, span types, trace structure.
- [Token usage and cost tracking](https://mlflow.org/docs/latest/genai/tracing/token-usage-cost/) —
  `trace.info.token_usage` / `trace.info.cost` schema — basis for
  `scorers/cost_scorers.py::_trace_usage`.
- [Access trace data](https://docs.databricks.com/aws/en/mlflow3/genai/tracing/observe-with-traces/access-trace-data) —
  `trace.info`, `trace.data.spans`, `trace.search_spans()`.
- [`mlflow.genai` Python API reference](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.genai.html) —
  `mlflow.genai.evaluate()` signature, `EvaluationResult` (`.metrics`,
  `.result_df`, `.run_id`), `BuiltinScorerName`.

### MLflow Skills (agentic coding guidance)

- [mlflow/skills](https://github.com/mlflow/skills) — "Turn your coding
  agent into an LLMOps expert."
  - `agent-evaluation` skill — end-to-end evaluation workflow: use native
    MLflow APIs (not a custom framework), register scorers, discover/build
    datasets, dry-run before a full run, model selection for scorers.
  - `agent-evaluation/references/scorers.md` — built-in vs. custom scorer
    decision guide, MLflow model URI format, scorer registration/testing.
  - `build-a-scorer` skill — "small suite, one scorer per criterion,
    cheapest reliable implementation, binary outputs" doctrine that shaped
    the code-based-scorer-first design in `scorers/tool_scorers.py` and
    `scorers/cost_scorers.py`.

### Findings verified empirically against the installed package (not purely
### doc-sourced — see docs/ARCHITECTURE.md §5 and §7 "Known limitation")

- `mlflow[databricks]==3.16.1` installed and imported locally to verify
  `mlflow.genai.scorers`, `mlflow.genai.judges.make_judge`,
  `mlflow.genai.datasets`, and the actual `result_df`/`search_traces()`
  schema (column names: `trace_id`, `request`, `response`,
  `execution_duration`, `assessments`), since published docs don't enumerate
  every column.
- The `MLFLOW_GENAI_EVAL_MAX_WORKERS` concurrency/SQLite deadlock and the
  span-attribute-name collision with `trace.info.token_usage` auto-detection
  were discovered by running the harness directly, not documented anywhere
  — see the "Known limitation" and "Gotcha discovered" notes in
  `docs/ARCHITECTURE.md`.

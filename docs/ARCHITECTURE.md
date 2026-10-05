# Architecture — Databricks AI Agent Evaluation Framework

This document explains **how** `databricks_ai_eval` is built and **why**, so the
design can be extended to new agents without re-deriving these decisions. For
step-by-step usage, see [RUNBOOK.md](./RUNBOOK.md).

## 1. Goals and constraints

1. Evaluate one or more Databricks Model Serving **agent endpoints** (not
   notebook code) -- the framework calls the deployed endpoint over REST,
   the same way a real client would.
2. Use **MLflow's native GenAI evaluation APIs** end-to-end
   (`mlflow.genai.evaluate`, `mlflow.genai.datasets`, `mlflow.genai.scorers`,
   `mlflow.genai.judges.make_judge`) rather than a bespoke evaluation loop --
   this is the explicit recommendation in Databricks' and MLflow's own
   evaluation guidance, because it keeps datasets, scorers, traces, and
   results all linked and queryable in one place.
3. Work identically whether run **locally** (SQLite-backed MLflow tracking)
   or **on Databricks** (Unity Catalog-backed MLflow tracking) -- same
   dataset, same scorers, same agent endpoint, same judge model.
4. Track **tool coverage**, **quality**, **latency**, and **token cost**
   together, and persist all of it to Unity Catalog so a dashboard can show
   trends and flag regressions over time.
5. Be **extensible to multiple agents** without code changes --
   `config/agents.yaml` is the single place a new agent is registered.

## 2. Component map

```
config/
  agents.yaml         <- agent registry (endpoint, tools, pricing, ...)
  eval_config.yaml     <- Unity Catalog location, MLflow envs, judge model, gates
  settings.py           <- typed loader for the two YAML files + env vars

datasets/
  expectation_builder.py  <- recomputes calendar-dependent ground truth at eval time
  build_dataset.py          <- generates the tool-coverage matrix dataset
  <agent>_eval_set.jsonl     <- generated dataset (also push-able to Unity Catalog)

scorers/
  tool_scorers.py   <- deterministic, code-based tool/fact-coverage scorers
  cost_scorers.py   <- latency + token/cost scorers
  judges.py         <- built-in LLM judges (wrapped) + custom make_judge() judges
                       + build_scorer_suite(), the single assembly point

eval/
  agent_client.py   <- REST client for the agent endpoint + predict_fn factory
  run_eval.py       <- the unified runner (local + Databricks)
  uc_sync.py        <- writes run/result/score rows to Unity Catalog

uc/
  ddl.sql                  <- table definitions (krishna.agent_eval.*)
  setup_uc.py              <- idempotently creates catalog/schema/tables
  seed_agent_registry.py    <- upserts config/agents.yaml into agent_registry

dashboard/
  dashboard_queries.sql     <- the SQL behind every dashboard widget (also useful standalone)
  eval_dashboard.lvdash.json <- Lakeview dashboard definition (importable)

docs/
  ARCHITECTURE.md (this file), RUNBOOK.md
```

## 3. Dataset design: tool-coverage matrix

The `vacation_planner_agent` has exactly two tools (`get_current_weather`,
`get_vacation_suggestions`) over four cities. `datasets/build_dataset.py`
generates a matrix that exercises every tool individually, every tool
combination, across multiple cities, plus graceful-degradation edge cases:

| Category | Rows | Purpose |
|---|---|---|
| `single_tool_weather` | 4 (one per city) | Only `get_current_weather` should fire |
| `single_tool_vacation` | 4 (one per city) | Only `get_vacation_suggestions` should fire |
| `combo_both_tools` | 4 (one per city) | Both tools required in one turn |
| `multi_city_combo` | 2 | Both tools, two cities each -- stresses repeated tool calls |
| `edge_unsupported_location` | 1 | Agent must decline gracefully, not fabricate data |
| `edge_no_tool_needed` | 1 | Chit-chat; no tool call, no fabricated facts |
| `edge_ambiguous_phrasing` | 1 | Ambiguous enough that *either* tool firing is acceptable |

Add more cities/tools by editing `config/agents.yaml`
(`supported_locations`, `tools`) -- `build_dataset.py` reads from there, no
hardcoded city list in the generator.

### Calendar-dependent ground truth

`get_current_weather` returns a different answer depending on **which month
the eval is run in**. Baking "expected weather facts" into the dataset once
would make it silently go stale every month. Instead:

- Each dataset row stores a **declarative spec**
  (`expectations.expectation_spec`): category, locations, which tools should
  fire, and expected behavior.
- `datasets/expectation_builder.materialize_expectations()` recomputes the
  concrete `expected_facts` **every time evaluation runs**, from the same
  static JSON file the agent's tools are built from
  (`data/vacation_planner_data.json`), using "today" as the reference date.
- `eval/run_eval.py::load_local_dataset` always calls this before evaluating,
  so the dataset file in git is a stable, reviewable artifact, but the
  ground truth it's checked against is always current.

Vacation-suggestion facts (best/avoid months) aren't calendar-dependent and
are also computed from the same static JSON for consistency (single source
of truth).

## 4. Scorer suite (`scorers/judges.py::build_scorer_suite`)

| Scorer | Type | What it checks |
|---|---|---|
| `tool_fact_coverage` | code-based | Does the final answer contain the facts only the *correct* tool(s) could produce? Primary tool-selection + grounding signal. |
| `tool_fact_coverage_ratio` | code-based | Numeric 0-1 companion, for trend/drift tracking that a binary pass/fail would mask. |
| `tool_trace_inspection` | code-based | Best-effort: if the trace has nested `TOOL` spans (only when the deployed agent itself is instrumented), reports which tools fired. Never fails the row when spans are absent (the common case for a remote, un-instrumented endpoint). |
| `latency` | code-based | `latency_ms` (numeric) + `latency_within_budget` (pass/fail vs. `config/eval_config.yaml` quality gate). |
| `token_usage_and_cost` | code-based | `input_tokens`, `output_tokens`, `total_tokens`, `estimated_cost_usd`, `token_usage_source`. See §5. |
| `safety`, `relevance_to_query`, `correctness` | built-in LLM judges, wrapped | Wraps MLflow's `Safety`/`RelevanceToQuery`/`Correctness` judges, extracting a plain string from our dict-shaped `outputs` first (same pattern as the official "wrap a predefined judge" example). `correctness` is skipped (not run) on edge-case rows that have no `expected_facts`. |
| `no_fabricated_destinations`, `actionable_vacation_recommendation` | built-in `Guidelines` judges | Domain-specific policy checks in plain English. |
| `tool_usage_validator` | custom `make_judge()`, trace-based | Analyzes `{{ trace }}` + `{{ inputs }}` + `{{ outputs }}` to judge tool selection; falls back to inferring tool usage from answer content when no nested tool spans are present. |

Why code-based scorers come first: per MLflow's own scorer-design guidance,
the cheapest reliable implementation should be preferred over an LLM judge.
`tool_fact_coverage` and the latency/cost scorers are deterministic, free,
and instantaneous; they also still work when Databricks judge-model access
isn't available (e.g. local dev before you've authenticated).

### All judges pin an explicit model

`config/eval_config.yaml: judge_model` (default `"databricks"`) is passed
explicitly to every judge/scorer constructor. Without this, MLflow silently
uses `"openai:/gpt-4.1-mini"` (needs `OPENAI_API_KEY`) for any
non-`"databricks"` tracking URI -- which would make local runs behave
differently from Databricks runs. Pinning the model keeps behavior
identical in both environments (both still need `DATABRICKS_HOST`/`TOKEN` to
reach the managed judge model and the agent endpoint).

## 5. Token usage & cost tracking

Token usage is resolved with a priority order so the same scorer works
across deployment states:

1. **MLflow trace-exact** (`trace.info.token_usage` / `trace.info.cost`) --
   available once the *agent itself* is instrumented with MLflow
   autologging (e.g. `mlflow.langchain.autolog()` before the agent is
   logged/deployed). This is the long-term target state.
2. **Endpoint-reported** -- if the serving endpoint's JSON response includes
   a `usage` field (OpenAI-compatible shape), `eval/agent_client.py` reads it
   directly.
3. **Estimated (tiktoken)** -- fallback token count using `cl100k_base`
   encoding over the request/response text, clearly tagged
   `usage_source=estimated_tiktoken` everywhere it's stored, so dashboards
   and cost figures are never silently presented as exact when they aren't.

Cost is computed from configurable `$/1M tokens` pricing in
`config/agents.yaml` (`pricing.input_usd_per_1m_tokens` /
`output_usd_per_1m_tokens`) -- update these to match whatever model serves
the agent (see `LLM_ENDPOINT` in `agents/vacation_planner_agent.py`).

### Gotcha discovered while building this: span-attribute name collisions

`eval/agent_client.py`'s `predict_fn` sets custom span attributes
(latency/usage) on the trace for debugging/cross-checking. Early versions
named them `usage.input_tokens` / `usage.total_tokens` etc. -- these names
**collided with MLflow's own internal token-usage auto-detection
heuristics**, which then reported a bogus `trace.info.token_usage =
{"input_tokens": 0, "output_tokens": 0, "total_tokens": <our value>}`
instead of leaving it unset. That zeroed out real token counts from the
endpoint-reported fallback (because `_trace_usage()` found a truthy-but-wrong
dict and never fell through to `_endpoint_usage()`). Fixed by:

1. Namespacing our span attributes under `databricks_ai_eval.*`
   (`eval/agent_client.py`).
2. Adding a sanity check in `scorers/cost_scorers.py::_trace_usage` that
   rejects an impossible `{input: 0, output: 0, total: >0}` reading and
   falls back to endpoint-reported usage, as defense in depth against any
   future collision.

## 6. Unity Catalog schema (`uc/ddl.sql`)

Default location: **`krishna.agent_eval`** (configurable via
`config/eval_config.yaml` or `EVAL_UC_CATALOG`/`EVAL_UC_SCHEMA` env vars --
set `schema: default` to use `krishna.default` instead).

| Table | Grain | Purpose |
|---|---|---|
| `agent_registry` | one row per agent | Discoverability for the dashboard; seeded from `config/agents.yaml` via `uc/seed_agent_registry.py`. |
| `eval_runs` | one row per `evaluate()` call | Run-level aggregates: pass rate, p50/p90/p99 latency, avg/total cost, total tokens. Primary trend table. |
| `eval_results` | one row per (run, example) | Wide/denormalized drill-down: request, response, latency, tokens, cost, overall pass/fail. |
| `eval_scores_long` | one row per (run, example, scorer) | Tidy/long scorer feedback -- the dashboard's primary source, since adding/removing a scorer never requires a schema change here. |

Evaluation **datasets** themselves (the input data, not the results) are
stored separately as MLflow GenAI evaluation datasets
(`mlflow.genai.datasets`), one UC table per agent, named
`<catalog>.<schema>.<agent_cfg.eval_dataset_table>` -- distinct from the
three metrics tables above.

## 7. Execution model: local vs. Databricks

Both modes call the **same deployed agent endpoint** and use the **same
scorer suite** -- only the MLflow tracking backend and (optionally) where
results get synced differ:

| | Local | Databricks |
|---|---|---|
| MLflow tracking URI | `sqlite:///mlflow.db` | `databricks` |
| Trace storage | local SQLite file | Unity Catalog-backed MLflow experiment |
| Experiment path | `/local/agent_eval/<agent_name>` | `/Shared/<agent_name>` |
| Agent invocation | REST call to the same endpoint, using a personal access token | same |
| Judge model | `databricks` (managed judge, via REST) | `databricks` (managed judge, in-workspace) |
| Unity Catalog writes | `databricks-sql-connector` against a SQL warehouse | native Spark session |
| Typical trigger | developer's laptop, ad hoc | scheduled Databricks Job / CI |

Each agent uses one stable experiment per tracking environment. Daily
evaluations create new MLflow runs inside that agent's experiment; they do
not create a new experiment for each evaluation. For example, the
`vacation_planner_agent` Databricks experiment is
`/Shared/vacation_planner_agent`. Experiments are placed directly under the
existing `/Shared` workspace directory: `/Shared/agent_eval` already names
an experiment, and a nested parent directory would need to exist before
MLflow can create a child experiment.

### Known limitation: SQLite + MLflow's concurrent eval harness

`mlflow.genai.evaluate()` runs `predict_fn` and scorers for **all rows
concurrently** via a thread pool (`MLFLOW_GENAI_EVAL_MAX_WORKERS`, default
>1). A local SQLite-backed tracking store cannot safely serve many
concurrent trace/assessment writes -- confirmed while building this
framework: a 17-row evaluation **hung indefinitely** with the default
worker count against `sqlite:///mlflow.db`, and completed in ~1 second once
pinned to a single worker. `eval/run_eval.py::_pin_local_eval_concurrency()`
sets `MLFLOW_GENAI_EVAL_MAX_WORKERS=1` automatically for any `run_source=local`
call (only if you haven't already set it yourself). This does **not**
apply to Databricks runs, which use a proper concurrent-safe tracking
backend.

## 8. Dashboard

`dashboard/dashboard_queries.sql` are the authoritative, independently
runnable queries; `dashboard/eval_dashboard.lvdash.json` is a Lakeview
dashboard definition built from the same queries, importable directly into
Databricks (Dashboards > Create > Import). Widgets:

- Latest pass rate / p90 latency / avg cost counters
- Latency percentiles over time (line, by agent)
- Token cost over time (line, by agent)
- Scorer pass-rate over time (line, by scorer -- the main regression-finder)
- Tool-coverage category breakdown for the latest run (table)
- Deviation detection: latest run vs. rolling 5-run baseline, flags >2σ
  moves in pass rate / p90 latency / avg cost (table)
- Failing examples drill-down for the latest run (table)

## 9. Extending to a new agent

1. Add an entry to `config/agents.yaml` (endpoint, tools, supported
   locations, pricing).
2. `python -m datasets.build_dataset --agent <name>` (customize
   `datasets/build_dataset.py::build_rows` if the new agent's tool shape
   differs from "N tools over M locations").
3. `python -m uc.seed_agent_registry` to register it.
4. `python -m eval.run_eval --agent <name> --run-source local` (or
   `databricks`).

No other code changes are required -- scorers, the runner, UC sync, and the
dashboard are all parameterized by `config/agents.yaml`.

-- =============================================================================
-- Unity Catalog schema for the Databricks AI agent evaluation framework.
--
-- Catalog/schema names are parameterized via {catalog} / {schema} placeholders
-- and are substituted at run time by uc/setup_uc.py from
-- config/eval_config.yaml (defaults: catalog = krishna, schema = agent_eval).
-- Run this file directly in a Databricks SQL editor/notebook if you prefer
-- not to use the Python helper -- just replace {catalog} / {schema} first.
-- =============================================================================

CREATE CATALOG IF NOT EXISTS {catalog};

CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}
COMMENT 'GenAI agent evaluation metrics: runs, per-example results, and scorer feedback for all agents tracked by databricks_ai_eval.';

-- -----------------------------------------------------------------------------
-- agent_registry: one row per agent endpoint under evaluation. Lets the
-- dashboard and runner discover "which agents exist" without code changes.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS {catalog}.{schema}.agent_registry (
    agent_name          STRING        NOT NULL COMMENT 'Logical agent name, matches config/agents.yaml',
    endpoint_name       STRING        NOT NULL COMMENT 'Databricks Model Serving endpoint name',
    endpoint_url        STRING        COMMENT 'Full invocation URL',
    tools               ARRAY<STRING> COMMENT 'Tool names this agent can call',
    supported_locations ARRAY<STRING> COMMENT 'Supported entities/locations for this agent',
    owner               STRING        COMMENT 'Team or user responsible for the agent',
    registered_at       TIMESTAMP     NOT NULL,
    updated_at          TIMESTAMP     NOT NULL
)
USING DELTA
COMMENT 'Registry of agent endpoints evaluated by this framework.';

-- -----------------------------------------------------------------------------
-- eval_runs: one row per `mlflow.genai.evaluate()` invocation (run-level
-- summary + aggregate metrics). Primary table for trend/deviation dashboards.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS {catalog}.{schema}.eval_runs (
    run_id               STRING    NOT NULL COMMENT 'Unique id for this evaluation run (uuid4)',
    mlflow_run_id        STRING    COMMENT 'Underlying MLflow run id',
    mlflow_experiment_id STRING    COMMENT 'MLflow experiment id',
    agent_name           STRING    NOT NULL,
    endpoint_name        STRING,
    dataset_table        STRING    COMMENT 'Fully qualified eval dataset table used',
    dataset_version      STRING,
    run_source           STRING    COMMENT "'local' or 'databricks'",
    triggered_by         STRING    COMMENT 'user or job/pipeline that triggered the run',
    git_sha              STRING,
    num_examples         INT,
    num_passed           INT,
    num_failed           INT,
    overall_pass_rate    DOUBLE,
    p50_latency_ms       DOUBLE,
    p90_latency_ms       DOUBLE,
    p99_latency_ms       DOUBLE,
    avg_cost_usd         DOUBLE,
    total_cost_usd       DOUBLE,
    total_input_tokens   BIGINT,
    total_output_tokens  BIGINT,
    total_tokens         BIGINT,
    started_at           TIMESTAMP NOT NULL,
    finished_at          TIMESTAMP,
    status               STRING    COMMENT "'RUNNING', 'SUCCESS', 'FAILED'",
    notes                STRING,
    run_date             DATE      COMMENT 'Partition column, derived from started_at'
)
USING DELTA
PARTITIONED BY (run_date)
COMMENT 'One row per evaluation run (mlflow.genai.evaluate invocation) with aggregate metrics.';

-- -----------------------------------------------------------------------------
-- eval_results: one row per dataset example per run (wide/denormalized,
-- convenient for drill-down from a failing run to the specific inputs).
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS {catalog}.{schema}.eval_results (
    run_id              STRING  NOT NULL,
    agent_name          STRING  NOT NULL,
    example_id          STRING  NOT NULL,
    category            STRING  COMMENT 'Tool-coverage category, e.g. single_tool_weather',
    trace_id            STRING,
    request             STRING  COMMENT 'Last user message content',
    response            STRING  COMMENT "Agent's final answer text",
    expected_behavior   STRING,
    latency_ms          DOUBLE,
    input_tokens        BIGINT,
    output_tokens       BIGINT,
    total_tokens        BIGINT,
    token_usage_source  STRING,
    estimated_cost_usd  DOUBLE,
    overall_pass        BOOLEAN COMMENT 'AND of all pass/fail scorers for this row',
    created_at          TIMESTAMP NOT NULL,
    run_date            DATE
)
USING DELTA
PARTITIONED BY (run_date)
COMMENT 'Per-example evaluation results, one row per (run_id, example_id).';

-- -----------------------------------------------------------------------------
-- eval_scores_long: tidy/long table, one row per (run, example, scorer). This
-- is the main source for the Lakeview dashboard -- easy to filter/group by
-- scorer_name without schema changes every time a scorer is added or removed.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS {catalog}.{schema}.eval_scores_long (
    run_id         STRING  NOT NULL,
    agent_name     STRING  NOT NULL,
    example_id     STRING  NOT NULL,
    category       STRING,
    scorer_name    STRING  NOT NULL,
    value          STRING  COMMENT 'Raw feedback value, e.g. "yes"/"no"/"correct" or a stringified number',
    numeric_value  DOUBLE  COMMENT 'Parsed numeric value when applicable (latency_ms, tokens, cost, coverage ratio)',
    pass_fail      STRING  COMMENT "Normalized 'pass'/'fail'/NULL for non-binary scorers",
    rationale      STRING,
    created_at     TIMESTAMP NOT NULL,
    run_date       DATE
)
USING DELTA
PARTITIONED BY (run_date)
COMMENT 'Long/tidy scorer feedback, one row per scorer per example per run. Primary dashboard source.';

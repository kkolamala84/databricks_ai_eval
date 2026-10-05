-- =============================================================================
-- Backing SQL queries for the Databricks Lakeview dashboard
-- (dashboard/eval_dashboard.lvdash.json). Each query is also useful on its
-- own in a Databricks SQL editor/notebook for ad-hoc investigation.
--
-- Replace {catalog}.{schema} with your values (default: krishna.agent_eval)
-- if running these manually instead of via uc/setup_uc.py-provisioned
-- dashboard datasets.
-- =============================================================================

-- -----------------------------------------------------------------------------
-- 1. Run-level summary trend: latency percentiles, cost, pass rate over time,
--    one point per evaluation run. Primary chart for "are we regressing?".
-- -----------------------------------------------------------------------------
SELECT
    run_id,
    agent_name,
    run_source,
    started_at,
    num_examples,
    overall_pass_rate,
    p50_latency_ms,
    p90_latency_ms,
    p99_latency_ms,
    avg_cost_usd,
    total_cost_usd,
    total_tokens
FROM {catalog}.{schema}.eval_runs
ORDER BY started_at;

-- -----------------------------------------------------------------------------
-- 2. Latency trend by agent (for a multi-agent dashboard filter)
-- -----------------------------------------------------------------------------
SELECT
    agent_name,
    started_at,
    p50_latency_ms,
    p90_latency_ms,
    p99_latency_ms
FROM {catalog}.{schema}.eval_runs
ORDER BY agent_name, started_at;

-- -----------------------------------------------------------------------------
-- 3. Token cost trend by agent
-- -----------------------------------------------------------------------------
SELECT
    agent_name,
    started_at,
    total_input_tokens,
    total_output_tokens,
    total_tokens,
    avg_cost_usd,
    total_cost_usd
FROM {catalog}.{schema}.eval_runs
ORDER BY agent_name, started_at;

-- -----------------------------------------------------------------------------
-- 4. Scorer pass-rate trend: per-scorer pass rate over time, the main signal
--    for catching a specific quality dimension regressing (e.g. Safety drops
--    but Correctness stays flat).
-- -----------------------------------------------------------------------------
SELECT
    s.run_id,
    r.agent_name,
    r.started_at,
    s.scorer_name,
    COUNT(*) AS n,
    SUM(CASE WHEN s.pass_fail = 'pass' THEN 1 ELSE 0 END) AS n_pass,
    SUM(CASE WHEN s.pass_fail = 'pass' THEN 1 ELSE 0 END) / COUNT(*) AS pass_rate,
    AVG(s.numeric_value) AS avg_numeric_value
FROM {catalog}.{schema}.eval_scores_long s
JOIN {catalog}.{schema}.eval_runs r USING (run_id)
WHERE s.pass_fail IS NOT NULL
GROUP BY s.run_id, r.agent_name, r.started_at, s.scorer_name
ORDER BY r.started_at, s.scorer_name;

-- -----------------------------------------------------------------------------
-- 5. Tool-coverage category breakdown for the latest run: pass rate and
--    average latency/cost per category (single_tool_weather, combo_both_tools,
--    edge_unsupported_location, ...). Surfaces whether failures cluster in a
--    specific tool-combination scenario.
-- -----------------------------------------------------------------------------
WITH latest_run AS (
    SELECT run_id FROM {catalog}.{schema}.eval_runs
    ORDER BY started_at DESC LIMIT 1
)
SELECT
    e.category,
    COUNT(*) AS n_examples,
    SUM(CASE WHEN e.overall_pass THEN 1 ELSE 0 END) / COUNT(*) AS pass_rate,
    AVG(e.latency_ms) AS avg_latency_ms,
    AVG(e.estimated_cost_usd) AS avg_cost_usd
FROM {catalog}.{schema}.eval_results e
JOIN latest_run USING (run_id)
GROUP BY e.category
ORDER BY pass_rate ASC;

-- -----------------------------------------------------------------------------
-- 6. Failing examples drill-down for the latest run (debug view)
-- -----------------------------------------------------------------------------
WITH latest_run AS (
    SELECT run_id FROM {catalog}.{schema}.eval_runs
    ORDER BY started_at DESC LIMIT 1
)
SELECT
    e.example_id,
    e.category,
    e.request,
    e.response,
    e.latency_ms,
    e.estimated_cost_usd,
    e.overall_pass
FROM {catalog}.{schema}.eval_results e
JOIN latest_run USING (run_id)
WHERE e.overall_pass = FALSE
ORDER BY e.category, e.example_id;

-- -----------------------------------------------------------------------------
-- 7. Deviation / anomaly detection: compare the latest run's key metrics
--    against a rolling baseline (previous 5 runs) and flag runs that moved
--    more than 2 standard deviations -- the core "find regressions
--    automatically" query for the dashboard's "Deviations" widget.
-- -----------------------------------------------------------------------------
WITH ordered_runs AS (
    SELECT
        run_id,
        agent_name,
        started_at,
        overall_pass_rate,
        p90_latency_ms,
        avg_cost_usd,
        ROW_NUMBER() OVER (PARTITION BY agent_name ORDER BY started_at DESC) AS rn
    FROM {catalog}.{schema}.eval_runs
),
baseline AS (
    SELECT
        agent_name,
        AVG(overall_pass_rate) AS baseline_pass_rate,
        STDDEV(overall_pass_rate) AS stddev_pass_rate,
        AVG(p90_latency_ms) AS baseline_p90_latency_ms,
        STDDEV(p90_latency_ms) AS stddev_p90_latency_ms,
        AVG(avg_cost_usd) AS baseline_avg_cost_usd,
        STDDEV(avg_cost_usd) AS stddev_avg_cost_usd
    FROM ordered_runs
    WHERE rn BETWEEN 2 AND 6 -- previous 5 runs, excluding the latest
    GROUP BY agent_name
),
latest AS (
    SELECT * FROM ordered_runs WHERE rn = 1
)
SELECT
    l.agent_name,
    l.started_at,
    l.overall_pass_rate,
    b.baseline_pass_rate,
    (l.overall_pass_rate - b.baseline_pass_rate) AS pass_rate_delta,
    l.p90_latency_ms,
    b.baseline_p90_latency_ms,
    (l.p90_latency_ms - b.baseline_p90_latency_ms) AS p90_latency_delta_ms,
    l.avg_cost_usd,
    b.baseline_avg_cost_usd,
    (l.avg_cost_usd - b.baseline_avg_cost_usd) AS avg_cost_delta_usd,
    CASE
        WHEN b.stddev_pass_rate > 0
             AND ABS(l.overall_pass_rate - b.baseline_pass_rate) > 2 * b.stddev_pass_rate
        THEN TRUE ELSE FALSE
    END AS pass_rate_is_anomalous,
    CASE
        WHEN b.stddev_p90_latency_ms > 0
             AND ABS(l.p90_latency_ms - b.baseline_p90_latency_ms) > 2 * b.stddev_p90_latency_ms
        THEN TRUE ELSE FALSE
    END AS latency_is_anomalous,
    CASE
        WHEN b.stddev_avg_cost_usd > 0
             AND ABS(l.avg_cost_usd - b.baseline_avg_cost_usd) > 2 * b.stddev_avg_cost_usd
        THEN TRUE ELSE FALSE
    END AS cost_is_anomalous
FROM latest l
JOIN baseline b USING (agent_name);

-- -----------------------------------------------------------------------------
-- 8. Agent registry (for a dashboard filter / "which agents are tracked")
-- -----------------------------------------------------------------------------
SELECT * FROM {catalog}.{schema}.agent_registry ORDER BY agent_name;

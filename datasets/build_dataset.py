"""
Builds the evaluation dataset for the vacation_planner_agent (or any agent
registered in config/agents.yaml with the same two-tool shape).

The dataset is designed around *tool coverage*: every supported location is
exercised against each tool individually and in combination, plus a small set
of edge cases that probe graceful degradation (unsupported city, no-tool-needed
chit-chat, ambiguous phrasing that should trigger at least one tool).

Usage
-----
Generate the JSONL file only (no Databricks / Unity Catalog required):
    python -m datasets.build_dataset --agent vacation_planner_agent

Also push to a Unity Catalog-backed MLflow evaluation dataset (requires
Databricks auth + CREATE TABLE on the target schema):
    python -m datasets.build_dataset --agent vacation_planner_agent --push-to-uc
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import settings  # noqa: E402
from datasets.expectation_builder import load_static_data, materialize_expectations  # noqa: E402

WEATHER_TOOL = "get_current_weather"
VACATION_TOOL = "get_vacation_suggestions"


def _row(example_id: str, query: str, spec: dict[str, Any]) -> dict[str, Any]:
    """One evaluation dataset record in MLflow's {inputs, expectations, tags}
    shape. `expectation_spec` is a framework-specific extension carried inside
    `expectations` so expectation_builder can recompute calendar-sensitive
    facts at evaluation time (see datasets/expectation_builder.py)."""
    return {
        "inputs": {"messages": [{"role": "user", "content": query}]},
        "expectations": {"expectation_spec": spec},
        "tags": {
            "example_id": example_id,
            "category": spec["category"],
            "tool_combo": "+".join(spec.get("expected_tools", [])) or "none",
        },
    }


def build_rows(cities: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    # --- Category: single tool - weather only (one per city) ---------------
    for city in cities:
        rows.append(
            _row(
                f"weather_only_{city}",
                f"What's the current weather like in {city.title()}?",
                {
                    "category": "single_tool_weather",
                    "locations": [city],
                    "expected_tools": [WEATHER_TOOL],
                    "tool_match_mode": "all",
                    "expected_behavior": "answer",
                },
            )
        )

    # --- Category: single tool - vacation suggestions only (one per city) --
    for city in cities:
        rows.append(
            _row(
                f"vacation_only_{city}",
                f"When is the best time of year to visit {city.title()}?",
                {
                    "category": "single_tool_vacation",
                    "locations": [city],
                    "expected_tools": [VACATION_TOOL],
                    "tool_match_mode": "all",
                    "expected_behavior": "answer",
                },
            )
        )

    # --- Category: combined tools, single city (one per city) --------------
    for city in cities:
        rows.append(
            _row(
                f"combo_both_{city}",
                f"I'm thinking about a trip to {city.title()}. What's the "
                f"weather like right now, and what are the best months to visit?",
                {
                    "category": "combo_both_tools",
                    "locations": [city],
                    "expected_tools": [WEATHER_TOOL, VACATION_TOOL],
                    "tool_match_mode": "all",
                    "expected_behavior": "answer",
                },
            )
        )

    # --- Category: combined tools across multiple cities (tool-combination
    # stress test -- forces the agent to call each tool multiple times) -----
    multi_city_pairs = list(itertools.combinations(cities, 2))[:2]
    for i, (city_a, city_b) in enumerate(multi_city_pairs, start=1):
        rows.append(
            _row(
                f"multi_city_combo_{i}",
                f"Compare {city_a.title()} and {city_b.title()} for a vacation: "
                f"tell me the current weather in each and the best months to go "
                f"for each city.",
                {
                    "category": "multi_city_combo",
                    "locations": [city_a, city_b],
                    "expected_tools": [WEATHER_TOOL, VACATION_TOOL],
                    "tool_match_mode": "all",
                    "expected_behavior": "answer",
                },
            )
        )

    # --- Edge case: unsupported location ------------------------------------
    rows.append(
        _row(
            "edge_unsupported_city",
            "What's the weather like in Tokyo right now?",
            {
                "category": "edge_unsupported_location",
                "locations": ["tokyo"],
                "expected_tools": [],
                "tool_match_mode": "any",
                "expected_behavior": "unsupported_location",
            },
        )
    )

    # --- Edge case: no tool needed (chit-chat / meta question) --------------
    rows.append(
        _row(
            "edge_no_tool_needed",
            "Hi! What can you help me with?",
            {
                "category": "edge_no_tool_needed",
                "locations": [],
                "expected_tools": [],
                "tool_match_mode": "any",
                "expected_behavior": "no_tool_needed",
            },
        )
    )

    # --- Edge case: ambiguous phrasing, at least one tool should fire -------
    rows.append(
        _row(
            "edge_ambiguous_single_city",
            f"Tell me about visiting {cities[-1].title()}.",
            {
                "category": "edge_ambiguous_phrasing",
                "locations": [cities[-1]],
                "expected_tools": [WEATHER_TOOL, VACATION_TOOL],
                "tool_match_mode": "any",
                "expected_behavior": "answer",
            },
        )
    )

    return rows


def seed_expected_facts(rows: list[dict[str, Any]], static_data: dict[str, Any]) -> None:
    """Pre-populate expectations with today's materialized facts, purely so
    the JSONL file is human-readable / reviewable in git. The eval runner
    ALWAYS recomputes these fresh at run time -- see
    datasets/expectation_builder.py and eval/run_eval.py."""
    for row in rows:
        spec = row["expectations"]["expectation_spec"]
        materialized = materialize_expectations({"expectation_spec": spec}, static_data)
        row["expectations"].update(materialized)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", default="vacation_planner_agent")
    parser.add_argument(
        "--out",
        default=None,
        help="Output JSONL path (default: datasets/<agent>_eval_set.jsonl)",
    )
    parser.add_argument(
        "--push-to-uc",
        action="store_true",
        help="Also create/update the MLflow evaluation dataset in Unity Catalog.",
    )
    args = parser.parse_args()

    agent_cfg = settings.agent(args.agent)
    static_data = load_static_data(agent_cfg.static_data_path)

    rows = build_rows(agent_cfg.supported_locations)
    seed_expected_facts(rows, static_data)

    out_path = Path(args.out) if args.out else Path(__file__).parent / f"{args.agent}_eval_set.jsonl"
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {len(rows)} rows to {out_path}")
    by_category: dict[str, int] = {}
    for row in rows:
        cat = row["tags"]["category"]
        by_category[cat] = by_category.get(cat, 0) + 1
    print("Tool-coverage breakdown:")
    for cat, count in sorted(by_category.items()):
        print(f"  {cat:30s} {count}")

    if args.push_to_uc:
        push_to_unity_catalog(args.agent, rows)


def push_to_unity_catalog(agent_name: str, rows: list[dict[str, Any]]) -> None:
    import mlflow
    import mlflow.genai.datasets as mlflow_datasets

    agent_cfg = settings.agent(agent_name)
    uc_table = settings.unity_catalog.fq_dataset_table(agent_cfg.eval_dataset_table)

    mlflow.set_tracking_uri(settings.mlflow_databricks.tracking_uri)
    try:
        dataset = mlflow_datasets.get_dataset(name=uc_table)
        print(f"Found existing dataset {uc_table}, merging {len(rows)} records.")
    except Exception:
        print(f"Creating new evaluation dataset {uc_table}.")
        dataset = mlflow_datasets.create_dataset(name=uc_table)

    # mlflow.genai.datasets expects plain {"inputs": ..., "expectations": ...}
    # records (the "tags" field we use locally is informational only).
    records = [{"inputs": r["inputs"], "expectations": r["expectations"]} for r in rows]
    dataset = dataset.merge_records(records)
    print(f"Dataset {uc_table} now has {len(dataset.to_df())} record(s).")


if __name__ == "__main__":
    main()

"""
Builds the `expectations` dict for an evaluation dataset row at *evaluation
time* (not dataset-authoring time).

Why: `get_current_weather` returns a different answer depending on the
calendar month the eval is run in (see WEATHER_DATA in
agents/vacation_planner_agent.py). If we baked "expected weather facts" into
the dataset once, the dataset would silently go stale every month. Instead,
each dataset row stores a *declarative* spec (which tools should fire, which
locations are involved, what kind of behavior is expected) and this module
recomputes the concrete ground-truth facts from the same static JSON file the
agent's tools are built from (databricks_ai_eval/data/vacation_planner_data.json),
using "today" every time evaluation runs. Vacation-suggestion facts (best /
avoid months) are NOT calendar-dependent, so they are safe to compute once
and are also recomputed here for consistency with a single source of truth.
"""

from __future__ import annotations

import datetime
import json
import re
from pathlib import Path
from typing import Any

MONTH_NAMES = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December",
}


def load_static_data(static_data_path: Path) -> dict[str, Any]:
    with open(static_data_path) as f:
        return json.load(f)


def _weather_facts(location: str, static_data: dict[str, Any], as_of: datetime.date) -> list[str]:
    weather_data = static_data["weather_data"]
    loc_key = location.strip().lower()
    if loc_key not in weather_data:
        return []
    month = as_of.month
    weather = weather_data[loc_key][str(month)]
    month_name = MONTH_NAMES[month]
    return [
        f"weather in {location.title()} this month ({month_name}): {weather}".lower(),
        weather.lower(),
    ]


def _vacation_facts(location: str, static_data: dict[str, Any]) -> list[str]:
    vacation_data = static_data["vacation_data"]
    loc_key = location.strip().lower()
    if loc_key not in vacation_data:
        return []
    data = vacation_data[loc_key]
    best = ", ".join(MONTH_NAMES[m] for m in data["best_months"])
    avoid = ", ".join(MONTH_NAMES[m] for m in data["avoid_months"])
    return [
        f"best months to visit {location.title()}: {best}".lower(),
        f"avoid: {avoid}. {data['avoid_reason']}".lower(),
        data["description"].lower(),
    ]


_FILLER_WORDS = {"and", "with", "occasional", "very", "extremely"}


def _weather_key_facts(location: str, static_data: dict[str, Any], as_of: datetime.date) -> list[str]:
    """Atomic, paraphrase-tolerant tokens: the Celsius temperature plus the
    condition words (e.g. 'Mild and rainy (avg 16C/61F)' -> '16c', 'mild', 'rainy')."""
    weather_data = static_data["weather_data"]
    loc_key = location.strip().lower()
    if loc_key not in weather_data:
        return []
    weather = weather_data[loc_key][str(as_of.month)].lower()
    description, _, rest = weather.partition("(")
    facts = [t for t in re.findall(r"\d+c", rest)][:1]
    facts += [w for w in re.findall(r"[a-z]+", description) if w not in _FILLER_WORDS]
    return facts


def _vacation_key_facts(location: str, static_data: dict[str, Any]) -> list[str]:
    vacation_data = static_data["vacation_data"]
    loc_key = location.strip().lower()
    if loc_key not in vacation_data:
        return []
    data = vacation_data[loc_key]
    return [MONTH_NAMES[m].lower() for m in data["best_months"]]


def materialize_expectations(
    row: dict[str, Any],
    static_data: dict[str, Any],
    as_of: datetime.date | None = None,
) -> dict[str, Any]:
    """Given a declarative dataset row (see datasets/build_dataset.py), return
    the full `expectations` dict to pass into mlflow.genai.evaluate(), with
    calendar-dependent facts freshly computed as of `as_of` (default: today).
    """
    as_of = as_of or datetime.date.today()
    spec = row["expectation_spec"]
    category = spec["category"]
    locations = spec.get("locations", [])
    expected_tools = spec.get("expected_tools", [])
    tool_match_mode = spec.get("tool_match_mode", "all")
    expected_behavior = spec.get("expected_behavior", "answer")

    expected_facts: list[str] = []
    expected_key_facts: list[str] = []
    expected_keywords: list[str] = []

    if expected_behavior == "answer":
        for loc in locations:
            if "get_current_weather" in expected_tools:
                expected_facts.extend(_weather_facts(loc, static_data, as_of))
                expected_key_facts.extend(_weather_key_facts(loc, static_data, as_of))
            if "get_vacation_suggestions" in expected_tools:
                expected_facts.extend(_vacation_facts(loc, static_data))
                expected_key_facts.extend(_vacation_key_facts(loc, static_data))
            expected_keywords.append(loc.lower())
    elif expected_behavior == "unsupported_location":
        supported = static_data["weather_data"].keys()
        expected_keywords = [loc.lower() for loc in locations] + list(supported)
    elif expected_behavior == "no_tool_needed":
        expected_keywords = []

    return {
        "expected_facts": expected_facts,
        "expected_key_facts": expected_key_facts,
        "expected_keywords": expected_keywords,
        "expected_tools": expected_tools,
        "tool_match_mode": tool_match_mode,
        "expected_behavior": expected_behavior,
        "locations": locations,
        "category": category,
    }

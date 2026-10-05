"""
Central configuration loader for the Databricks AI agent evaluation framework.

Everything that differs between agents (endpoint URL, tools, pricing) or
between environments (local vs. Databricks) is read from
``config/agents.yaml`` and ``config/eval_config.yaml`` plus a small number of
environment variables. No other module should hardcode catalog/schema names,
endpoint URLs, or pricing -- import from here instead.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = Path(__file__).resolve().parent

# Load a local .env file if python-dotenv is available, so DATABRICKS_HOST /
# DATABRICKS_TOKEN / etc. can be kept out of the shell profile.
try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass


@dataclass
class Pricing:
    input_usd_per_1m_tokens: float
    output_usd_per_1m_tokens: float


@dataclass
class AgentConfig:
    name: str
    description: str
    endpoint_name: str
    endpoint_url: str
    input_schema: str
    source_file: str
    static_data_file: str
    tools: list[str]
    supported_locations: list[str]
    eval_dataset_table: str
    pricing: Pricing

    @property
    def static_data_path(self) -> Path:
        return PROJECT_ROOT / self.static_data_file

    def resolved_endpoint_url(self, databricks_host: Optional[str] = None) -> str:
        """Return the invocation URL, deriving it from DATABRICKS_HOST if the
        YAML entry didn't hardcode one (useful when promoting the same config
        across workspaces/environments)."""
        if self.endpoint_url:
            return self.endpoint_url
        host = databricks_host or os.environ.get("DATABRICKS_HOST", "").rstrip("/")
        if not host:
            raise ValueError(
                f"Agent '{self.name}' has no endpoint_url and DATABRICKS_HOST is not set."
            )
        return f"{host}/serving-endpoints/{self.endpoint_name}/invocations"


@dataclass
class UnityCatalogConfig:
    catalog: str
    schema: str
    tables: dict[str, str]

    def fq_table(self, key: str) -> str:
        """Fully qualified `catalog.schema.table` name for one of the
        tables.runs/results/scores_long/agent_registry keys."""
        return f"{self.catalog}.{self.schema}.{self.tables[key]}"

    def fq_dataset_table(self, dataset_table_name: str) -> str:
        return f"{self.catalog}.{self.schema}.{dataset_table_name}"


@dataclass
class MlflowEnvConfig:
    tracking_uri: str
    experiment_base_path: str

    def experiment_name_for(self, agent_name: str) -> str:
        """Stable MLflow experiment path for one agent in this environment."""
        return f"{self.experiment_base_path.rstrip('/')}/{agent_name}"


@dataclass
class QualityGates:
    min_pass_rate: float
    max_p90_latency_ms: float
    max_avg_cost_usd: float


class Settings:
    """Loads and exposes the merged YAML configuration."""

    def __init__(
        self,
        agents_yaml: Path = CONFIG_DIR / "agents.yaml",
        eval_yaml: Path = CONFIG_DIR / "eval_config.yaml",
    ):
        with open(agents_yaml) as f:
            agents_raw = yaml.safe_load(f)
        with open(eval_yaml) as f:
            eval_raw = yaml.safe_load(f)

        self._agents: dict[str, AgentConfig] = {}
        for a in agents_raw["agents"]:
            pricing = Pricing(**a["pricing"])
            self._agents[a["name"]] = AgentConfig(
                name=a["name"],
                description=a.get("description", ""),
                endpoint_name=a["endpoint_name"],
                endpoint_url=a.get("endpoint_url", ""),
                input_schema=a.get("input_schema", "chat_messages"),
                source_file=a.get("source_file", ""),
                static_data_file=a.get("static_data_file", ""),
                tools=a.get("tools", []),
                supported_locations=a.get("supported_locations", []),
                eval_dataset_table=a["eval_dataset_table"],
                pricing=pricing,
            )

        uc = eval_raw["unity_catalog"]
        self.unity_catalog = UnityCatalogConfig(
            catalog=os.environ.get("EVAL_UC_CATALOG", uc["catalog"]),
            schema=os.environ.get("EVAL_UC_SCHEMA", uc["schema"]),
            tables=uc["tables"],
        )

        mlf = eval_raw["mlflow"]
        self.mlflow_local = MlflowEnvConfig(
            tracking_uri=os.environ.get(
                "MLFLOW_LOCAL_TRACKING_URI", mlf["local"]["tracking_uri"]
            ),
            experiment_base_path=mlf["local"]["experiment_base_path"],
        )
        self.mlflow_databricks = MlflowEnvConfig(
            tracking_uri=mlf["databricks"]["tracking_uri"],
            experiment_base_path=mlf["databricks"]["experiment_base_path"],
        )

        self.judge_model = os.environ.get("MLFLOW_JUDGE_MODEL", eval_raw["judge_model"])

        qg = eval_raw["quality_gates"]
        self.quality_gates = QualityGates(**qg)

    @property
    def agents(self) -> dict[str, AgentConfig]:
        return self._agents

    def agent(self, name: str) -> AgentConfig:
        if name not in self._agents:
            raise KeyError(
                f"Unknown agent '{name}'. Known agents: {list(self._agents)}. "
                "Add it to config/agents.yaml."
            )
        return self._agents[name]

    def mlflow_env(self, run_source: str) -> MlflowEnvConfig:
        if run_source == "local":
            return self.mlflow_local
        if run_source == "databricks":
            return self.mlflow_databricks
        raise ValueError("run_source must be 'local' or 'databricks'")


settings = Settings()

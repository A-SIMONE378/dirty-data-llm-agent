"""B4 environment-factory adapter for TRL GRPO on the A3 sandbox."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .a1_corruption import load_corruption_profiles
from .a2_tasks import load_task_records
from .agent_env import AgentEnvironmentConfig, AgentEpisode
from .b_reward import RewardConfig, compute_b_reward
from .schema_semantics import load_table_semantics


class B4EnvironmentFactory:
    """A single environment factory; task_id is supplied by the GRPO dataset."""

    def __init__(
        self,
        *,
        project_root: str | Path,
        task_ids: list[str],
        runtime_root: str | Path,
        reward_config: RewardConfig | None = None,
        dirty_cache: str | Path | None = None,
    ) -> None:
        self.root = Path(project_root).resolve()
        self.dirty_cache = Path(
            dirty_cache or self.root / "artifacts" / "b4_dirty_cache"
        )
        records = load_task_records(self.root / "data" / "tasks" / "a2_tasks.jsonl")
        self.records = {record["task_id"]: record for record in records}
        self.task_ids = list(task_ids)
        missing = [task_id for task_id in self.task_ids if task_id not in self.records]
        if missing:
            raise ValueError(f"B4 task IDs missing from A2: {missing[:5]}")
        if any(self.records[task_id]["split"] != "train" for task_id in self.task_ids):
            raise ValueError("B4 only supports train tasks.")
        self.semantics = load_table_semantics(self.root / "configs" / "table_semantics.json")
        self.profiles = load_corruption_profiles(
            self.root / "configs" / "a1_corruption_profiles.json"
        )
        config = AgentEnvironmentConfig.from_json(self.root / "configs" / "a3_agent_env.json")
        self.env_config = replace(
            config,
            max_steps=10,
            run_timeout_seconds=15.0,
            max_memory_mb=8192,
            max_cpu_seconds=120,
        )
        self.runtime_root = Path(runtime_root)
        self.reward_config = reward_config or RewardConfig()

    def __call__(self) -> "B4TaskEnvironment":
        return B4TaskEnvironment(self)


class B4TaskEnvironment:
    """Expose the three A3 tools and an environment-owned 0/1 reward."""

    def __init__(self, factory: B4EnvironmentFactory) -> None:
        self._factory = factory
        self._episode: AgentEpisode | None = None
        self._task_id: str | None = None
        self._submitted = False

    def reset(self, task_id: str, **kwargs: Any) -> None:
        self._close()
        if task_id not in self._factory.records:
            raise ValueError(f"Unknown B4 task: {task_id}")
        record = self._factory.records[task_id]
        if record["split"] != "train":
            raise ValueError("B4 reset received a non-train task.")
        self._task_id = task_id
        self._submitted = False
        cache_path = self._factory.dirty_cache / f"{task_id}.csv"
        if not cache_path.is_file():
            raise ValueError(f"Missing B4 dirty-table cache: {cache_path}")
        dirty_frame = pd.read_csv(cache_path, low_memory=False)
        self._episode = AgentEpisode(
            task_id=task_id,
            table=record["table"],
            split=record["split"],
            question=record["question"],
            dirty_profile=record["dirty_profile"],
            dirty_frame=dirty_frame,
            expected_answer=record["answer"],
            config=self._factory.env_config,
            runtime_root=self._factory.runtime_root,
        )
        return None

    def inspect_table(self, head_rows: int = 5) -> str:
        """Inspect columns, dtypes, missing counts, row count, and a bounded head.

        Args:
            head_rows: Number of head rows to show, from 1 to 10.

        Returns:
            A JSON string with table metadata and head rows.
        """
        return self._call("inspect_table", {"head_rows": head_rows})

    def run_python(self, code: str) -> str:
        """Run Python against the persistent dataframe variable df.

        Args:
            code: Python code to execute in the sandbox.

        Returns:
            The tool output, or an error message.
        """
        return self._call("run_python", {"code": code})

    def submit_answer(self, answer: Any) -> str:
        """Submit the final JSON-compatible answer for strict verification.

        Args:
            answer: Scalar, boolean, list, or JSON object answer.

        Returns:
            Acceptance or rejection text.
        """
        self._submitted = True
        return self._call("submit_answer", {"answer": answer})

    def get_reward(self) -> float:
        """Return the terminal verifiable reward for the completed episode."""

        if self._episode is None:
            return 0.0
        trajectory = self._episode.trajectory.to_dict()
        return compute_b_reward(trajectory, self._factory.reward_config).total

    def _call(self, tool: str, arguments: dict[str, Any]) -> str:
        if self._episode is None:
            return "ERROR [EnvironmentError]: reset() has not been called."
        result = self._episode.call_tool(tool, arguments)
        return result.content

    def _close(self) -> None:
        if self._episode is not None:
            self._episode.close()
            self._episode = None


def load_b4_task_ids(
    project_root: str | Path,
    *,
    limit: int | None = None,
) -> list[str]:
    path = Path(project_root) / "artifacts" / "b3_rollouts" / "filtered_tasks.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"No filtered B3 tasks in {path}.")
    task_ids = [str(row["task_id"]) for row in rows]
    if limit is not None:
        if limit < 1:
            raise ValueError("task limit must be positive.")
        task_ids = task_ids[:limit]
    return task_ids

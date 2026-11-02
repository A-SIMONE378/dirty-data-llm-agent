"""B1 reset/step environment wrapper around the bounded A3 agent episode."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable, Mapping

from .agent_env import AgentEnvironmentConfig, AgentEpisode
from .b_reward import RewardBreakdown, RewardConfig, compute_b_reward


class BEnvironmentError(ValueError):
    pass


@dataclass(frozen=True)
class BStep:
    observation: dict[str, Any]
    reward: float
    done: bool
    info: dict[str, Any]


class BEnvironment:
    """One deterministic agent environment with a terminal verifiable reward."""

    def __init__(
        self,
        *,
        task_id: str,
        episode_factory: Callable[[], AgentEpisode],
        config: AgentEnvironmentConfig | None = None,
        reward_config: RewardConfig | None = None,
        rollout_timeout_seconds: float | None = 300.0,
    ) -> None:
        if not isinstance(task_id, str) or not task_id:
            raise BEnvironmentError("task_id must be a non-empty string.")
        if rollout_timeout_seconds is not None and rollout_timeout_seconds <= 0:
            raise BEnvironmentError("rollout_timeout_seconds must be positive.")
        self.task_id = task_id
        self._episode_factory = episode_factory
        self.config = config or AgentEnvironmentConfig()
        self.reward_config = reward_config or RewardConfig()
        self.rollout_timeout_seconds = rollout_timeout_seconds
        self._episode: AgentEpisode | None = None
        self._started: float | None = None
        self._last_reward: RewardBreakdown | None = None
        self._done = False

    @classmethod
    def from_task_record(
        cls,
        record: Mapping[str, Any],
        *,
        tables_dir: str,
        semantics_registry: Mapping[str, Any],
        profile_config: Mapping[str, Any],
        config: AgentEnvironmentConfig | None = None,
        reward_config: RewardConfig | None = None,
        runtime_root: str | None = None,
        rollout_timeout_seconds: float | None = 300.0,
    ) -> "BEnvironment":
        def factory() -> AgentEpisode:
            return AgentEpisode.from_task_record(
                record,
                tables_dir=tables_dir,
                semantics_registry=semantics_registry,
                profile_config=profile_config,
                config=config,
                runtime_root=runtime_root,
            )

        return cls(
            task_id=str(record["task_id"]),
            episode_factory=factory,
            config=config,
            reward_config=reward_config,
            rollout_timeout_seconds=rollout_timeout_seconds,
        )

    @classmethod
    def from_frame(
        cls,
        *,
        task_id: str,
        question: str,
        dirty_frame: Any,
        expected_answer: Any,
        table: str = "fixture.csv",
        split: str = "train",
        dirty_profile: str = "low",
        config: AgentEnvironmentConfig | None = None,
        reward_config: RewardConfig | None = None,
        runtime_root: str | None = None,
        rollout_timeout_seconds: float | None = 300.0,
    ) -> "BEnvironment":
        source = dirty_frame.copy(deep=True)

        def factory() -> AgentEpisode:
            return AgentEpisode(
                task_id=task_id,
                table=table,
                split=split,
                question=question,
                dirty_profile=dirty_profile,
                dirty_frame=source.copy(deep=True),
                expected_answer=expected_answer,
                config=config,
                runtime_root=runtime_root,
            )

        return cls(
            task_id=task_id,
            episode_factory=factory,
            config=config,
            reward_config=reward_config,
            rollout_timeout_seconds=rollout_timeout_seconds,
        )

    @property
    def done(self) -> bool:
        return self._done

    @property
    def last_reward(self) -> RewardBreakdown | None:
        return self._last_reward

    @property
    def episode(self) -> AgentEpisode | None:
        return self._episode

    def reset(self) -> dict[str, Any]:
        self.close()
        self._episode = self._episode_factory()
        self._started = time.monotonic()
        self._done = False
        self._last_reward = None
        return {
            "task_id": self.task_id,
            "question": self._episode.question,
            "tools": self._episode.tool_schemas(),
            "max_steps": self._episode.config.max_steps,
            "step": 0,
        }

    def step(self, tool: str, arguments: Mapping[str, Any] | None = None) -> BStep:
        if self._episode is None:
            raise BEnvironmentError("Call reset() before step().")
        if self._done:
            raise BEnvironmentError("Episode is already terminal.")
        if not isinstance(tool, str) or not tool:
            raise BEnvironmentError("tool must be a non-empty string.")
        started = time.monotonic()
        result = self._episode.call_tool(tool, dict(arguments or {}))
        timed_out = (
            self.rollout_timeout_seconds is not None
            and self._started is not None
            and time.monotonic() - self._started > self.rollout_timeout_seconds
        )
        done = self._episode.terminal or timed_out
        reward = 0.0
        breakdown = None
        if done:
            trajectory = self._episode.trajectory.to_dict()
            breakdown = compute_b_reward(trajectory, self.reward_config)
            reward = breakdown.total
            self._last_reward = breakdown
            self._done = True
        observation = {
            "ok": result.ok,
            "content": result.content,
            "truncated": result.truncated,
            "error_type": result.error_type,
            "terminal_status": self._episode.trajectory.terminal_status,
            "step": len(self._episode.trajectory.events),
            "elapsed_seconds": round(time.monotonic() - started, 6),
        }
        info = {
            "task_id": self.task_id,
            "terminal_status": self._episode.trajectory.terminal_status,
            "timed_out": timed_out,
            "reward_breakdown": None if breakdown is None else breakdown.to_dict(),
        }
        return BStep(observation=observation, reward=reward, done=done, info=info)

    def replay(self, actions: list[tuple[str, Mapping[str, Any]]]) -> dict[str, Any]:
        self.reset()
        steps: list[dict[str, Any]] = []
        for tool, arguments in actions:
            step = self.step(tool, arguments)
            steps.append(step.info)
            if step.done:
                break
        if self._episode is None:
            raise BEnvironmentError("Episode was not created.")
        return {
            "task_id": self.task_id,
            "reward": self._last_reward.total if self._last_reward else 0.0,
            "done": self._done,
            "terminal_status": self._episode.trajectory.terminal_status,
            "steps": steps,
            "trajectory": self._episode.trajectory.to_dict(),
        }

    def close(self) -> None:
        if self._episode is not None:
            self._episode.close()
            self._episode = None
        self._done = False
        self._started = None

    def __enter__(self) -> "BEnvironment":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

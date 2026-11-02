from pathlib import Path
import sys

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dirty_da.agent_env import AgentEnvironmentConfig  # noqa: E402
from dirty_da.b_env import BEnvironment, BEnvironmentError  # noqa: E402
from dirty_da.b_reward import RewardConfig  # noqa: E402


def make_environment():
    return BEnvironment.from_frame(
        task_id="b1-test",
        question="Return the number of rows.",
        dirty_frame=pd.DataFrame({"group": ["a", "a", "b"], "value": [1, 2, 3]}),
        expected_answer=3,
        config=AgentEnvironmentConfig(max_steps=10),
        reward_config=RewardConfig(),
    )


def test_b1_reset_step_and_terminal_reward_do_not_expose_oracle():
    with make_environment() as environment:
        observation = environment.reset()
        assert "expected_answer" not in str(observation)
        assert observation["question"] == "Return the number of rows."
        assert len(observation["tools"]) == 3
        first = environment.step("inspect_table", {"head_rows": 2})
        assert first.done is False and first.reward == 0.0
        second = environment.step("run_python", {"code": "len(df)"})
        assert second.done is False
        third = environment.step("submit_answer", {"answer": 3})
        assert third.done is True
        assert third.reward == 1.0
        assert third.info["terminal_status"] == "accepted"
        assert environment.done is True
        with pytest.raises(BEnvironmentError):
            environment.step("inspect_table", {"head_rows": 1})


def test_b1_replay_is_deterministic():
    actions = [
        ("inspect_table", {"head_rows": 2}),
        ("run_python", {"code": "len(df)"}),
        ("submit_answer", {"answer": 3}),
    ]
    with make_environment() as environment:
        first = environment.replay(actions)
        second = environment.replay(actions)
        assert first["reward"] == second["reward"] == 1.0
        assert first["terminal_status"] == second["terminal_status"] == "accepted"
        assert [
            (event["tool"], event["arguments"])
            for event in first["trajectory"]["events"]
        ] == [
            (event["tool"], event["arguments"])
            for event in second["trajectory"]["events"]
        ]


def test_b1_reset_starts_a_fresh_episode():
    with make_environment() as environment:
        environment.reset()
        environment.step("inspect_table", {"head_rows": 1})
        environment.reset()
        assert environment.episode is not None
        assert len(environment.episode.trajectory.events) == 0
        assert environment.done is False

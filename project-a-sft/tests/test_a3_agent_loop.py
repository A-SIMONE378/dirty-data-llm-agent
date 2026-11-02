from copy import deepcopy

import pandas as pd
import pytest

from dirty_da.agent_env import AgentEnvironmentConfig, AgentEpisode
from dirty_da.agent_loop import AgentLoopError, AssistantAction, run_agent_episode
from dirty_da.trajectory import validate_trajectory_record


def make_episode(*, expected_answer=3, max_steps=10):
    return AgentEpisode(
        task_id="loop-test",
        table="fixture.csv",
        split="dev",
        question="Return the number of rows.",
        dirty_profile="low",
        dirty_frame=pd.DataFrame({"x": [1, 2, 3]}),
        expected_answer=expected_answer,
        config=AgentEnvironmentConfig(max_steps=max_steps),
    )


class ScriptedAdapter:
    def __init__(self, actions):
        self.actions = list(actions)
        self.calls = []

    def next_action(self, messages, tools):
        self.calls.append((deepcopy(messages), deepcopy(tools)))
        return self.actions.pop(0)


def test_loop_runs_three_tools_and_records_complete_accepted_trajectory():
    adapter = ScriptedAdapter(
        [
            AssistantAction("Inspect first.", "inspect_table", {"head_rows": 1}),
            AssistantAction("Count rows.", "run_python", {"code": "len(df)"}),
            AssistantAction("Submit.", "submit_answer", {"answer": 3}),
        ]
    )
    with make_episode() as episode:
        trajectory = run_agent_episode(episode, adapter)
        assert trajectory.terminal_status == "accepted"
        assert [event.tool for event in trajectory.events] == [
            "inspect_table",
            "run_python",
            "submit_answer",
        ]
        assert trajectory.submitted_answer == 3
        validate_trajectory_record(trajectory.to_dict())


def test_model_context_contains_no_private_oracle_or_spec_fields():
    adapter = ScriptedAdapter(
        [AssistantAction("Submit.", "submit_answer", {"answer": 3})]
    )
    with make_episode(expected_answer=3) as episode:
        run_agent_episode(episode, adapter)
    context = repr(adapter.calls)
    assert "expected_answer" not in context
    assert "canonical_fingerprint" not in context
    assert "task_spec" not in context
    assert {tool["name"] for tool in adapter.calls[0][1]} == {
        "inspect_table",
        "run_python",
        "submit_answer",
    }


def test_loop_rejects_malformed_model_action_without_faking_trajectory():
    adapter = ScriptedAdapter([{"content": "missing fields"}])
    with make_episode() as episode:
        with pytest.raises(AgentLoopError, match="exactly"):
            run_agent_episode(episode, adapter)
        assert episode.trajectory.events == []
        assert episode.trajectory.terminal_status == "running"


def test_model_error_is_a_valid_terminal_status_for_persisted_failures():
    with make_episode() as episode:
        episode.trajectory.terminal_status = "model_error"
        validate_trajectory_record(episode.trajectory.to_dict())


def test_invalid_action_is_a_valid_terminal_capability_failure():
    with make_episode() as episode:
        episode.trajectory.terminal_status = "invalid_action"
        validate_trajectory_record(episode.trajectory.to_dict())

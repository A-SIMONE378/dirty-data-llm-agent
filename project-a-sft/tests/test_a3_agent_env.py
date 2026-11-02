import json
import os
from pathlib import Path

import pandas as pd
import pytest

from dirty_da.a1_corruption import load_corruption_profiles
from dirty_da.a2_tasks import load_task_records
from dirty_da.agent_env import (
    A3_TOOLS,
    AgentEnvironmentConfig,
    AgentEnvironmentError,
    AgentEpisode,
)
from dirty_da.schema_semantics import load_table_semantics
from dirty_da.trajectory import validate_trajectory_record


ROOT = Path(__file__).resolve().parents[1]


def make_episode(
    *,
    expected_answer=3,
    config: AgentEnvironmentConfig | None = None,
) -> AgentEpisode:
    frame = pd.DataFrame(
        {
            "group": ["a", "a", "b"],
            "value": [1.0, 2.0, 3.0],
            "optional": [None, "x", "y"],
        }
    )
    return AgentEpisode(
        task_id="a3-test",
        table="fixture.csv",
        split="train",
        question="Return the number of rows.",
        dirty_profile="low",
        dirty_frame=frame,
        expected_answer=expected_answer,
        config=config,
    )


def test_config_requires_exact_three_tools_and_bounded_steps(tmp_path):
    config = json.loads((ROOT / "configs" / "a3_agent_env.json").read_text())
    assert config["tools"] == list(A3_TOOLS)
    loaded = AgentEnvironmentConfig.from_json(ROOT / "configs" / "a3_agent_env.json")
    assert loaded.max_steps == 12
    bad = dict(config)
    bad["tools"] = ["inspect_table", "submit_answer"]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(AgentEnvironmentError, match="exactly the three"):
        AgentEnvironmentConfig.from_json(path)
    with pytest.raises(AgentEnvironmentError, match="between 10 and 12"):
        AgentEnvironmentConfig(max_steps=13)


def test_inspect_table_is_bounded_and_does_not_expose_oracle_fields():
    with make_episode(expected_answer={"secret": 999}) as episode:
        result = episode.call_tool("inspect_table", {"head_rows": 2})
        payload = json.loads(result.content)
        assert result.ok is True
        assert payload["row_count"] == 3
        assert len(payload["head"]) == 2
        assert payload["columns"][2]["missing"] == 1
        assert "secret" not in result.content
        assert "expected_answer" not in result.content
        assert "task_spec" not in result.content


def test_run_python_keeps_state_and_can_analyze_df():
    with make_episode() as episode:
        first = episode.call_tool("run_python", {"code": "subtotal = df['value'].sum()"})
        second = episode.call_tool("run_python", {"code": "subtotal + len(df)"})
        assert first.ok is True
        assert second.ok is True
        assert second.content.strip() == "9.0"
        assert episode.sandbox_resource_limits_active == (os.name == "posix")


@pytest.mark.parametrize(
    "code",
    [
        "import socket",
        "open('table.csv').read()",
        "pd.read_csv('../secret.csv')",
        "pd.io.common.os.system('echo forbidden')",
    ],
)
def test_run_python_blocks_import_file_escape_and_process_access(code):
    with make_episode() as episode:
        result = episode.call_tool("run_python", {"code": code})
        assert result.ok is False
        assert result.error_type in {"PermissionError", "FileNotFoundError"}


def test_run_python_allows_pandas_replace_cleaning():
    with make_episode() as episode:
        result = episode.call_tool(
            "run_python",
            {
                "code": (
                    "cleaned = df['group'].replace('a', 'z').str.replace('b', 'c')"
                    "\ncleaned.tolist()"
                )
            },
        )
        assert result.ok is True
        assert result.content.strip() == "['z', 'z', 'c']"


def test_run_python_allows_only_pandas_and_numpy_imports():
    with make_episode() as episode:
        allowed = episode.call_tool(
            "run_python",
            {"code": "import pandas as pd\nimport numpy as np\nlen(df)"},
        )
        assert allowed.ok is True
        assert allowed.content.strip() == "3"
        blocked_module = episode.call_tool("run_python", {"code": "import os"})
        assert blocked_module.ok is False
        assert blocked_module.error_type == "PermissionError"
        blocked_from = episode.call_tool(
            "run_python", {"code": "from pandas import read_csv"}
        )
        assert blocked_from.ok is False
        assert blocked_from.error_type == "PermissionError"


def test_run_python_truncates_large_output():
    config = AgentEnvironmentConfig(max_output_chars=200)
    with make_episode(config=config) as episode:
        result = episode.call_tool("run_python", {"code": "print('x' * 2000)"})
        assert result.ok is True
        assert result.truncated is True
        assert len(result.content) == 200
        assert result.content.endswith("...[output truncated]")


def test_run_python_timeout_terminates_sandbox_and_episode():
    config = AgentEnvironmentConfig(run_timeout_seconds=3.0)
    with make_episode(config=config) as episode:
        result = episode.call_tool("run_python", {"code": "while True:\n    pass"})
        assert result.ok is False
        assert result.error_type == "TimeoutError"
        assert episode.trajectory.terminal_status == "sandbox_error"
        after = episode.call_tool("inspect_table")
        assert after.error_type == "EpisodeTerminated"


def test_submit_uses_strict_verifier_without_returning_expected_answer():
    with make_episode(expected_answer=3) as accepted:
        result = accepted.call_tool("submit_answer", {"answer": 3})
        assert result.ok is True
        assert result.content == "Answer accepted."
        assert accepted.trajectory.terminal_status == "accepted"
        serialized = accepted.trajectory.to_dict()
        validate_trajectory_record(serialized)
        assert "expected_answer" not in serialized
        assert "spec" not in serialized
    with make_episode(expected_answer=3) as rejected:
        result = rejected.call_tool("submit_answer", {"answer": 3.0})
        assert result.ok is False
        assert result.content == "Answer rejected."
        assert "expected" not in result.content.casefold()
        assert rejected.trajectory.terminal_status == "rejected"


def test_max_steps_is_enforced_and_recorded():
    config = AgentEnvironmentConfig(max_steps=10)
    with make_episode(config=config) as episode:
        for _ in range(10):
            assert episode.call_tool("inspect_table", {"head_rows": 1}).ok is True
        assert episode.trajectory.terminal_status == "max_steps"
        assert len(episode.trajectory.events) == 10
        blocked = episode.call_tool("inspect_table")
        assert blocked.error_type == "EpisodeTerminated"


def test_a2_record_reconstructs_declared_dirty_table_without_exposing_spec():
    records = load_task_records(ROOT / "data" / "tasks" / "a2_tasks.jsonl")
    record = records[0]
    with AgentEpisode.from_task_record(
        record,
        tables_dir=ROOT / "data" / "tables",
        semantics_registry=load_table_semantics(
            ROOT / "configs" / "table_semantics.json"
        ),
        profile_config=load_corruption_profiles(
            ROOT / "configs" / "a1_corruption_profiles.json"
        ),
        config=AgentEnvironmentConfig(max_steps=10),
    ) as episode:
        assert episode.question == record["question"]
        assert episode.tools == A3_TOOLS
        inspection = episode.call_tool("inspect_table", {"head_rows": 1})
        assert inspection.ok is True
        assert json.dumps(record["spec"], ensure_ascii=False) not in inspection.content


def test_constructor_does_not_mutate_input_frame():
    frame = pd.DataFrame({"x": [1, 2], "g": ["a", "b"]})
    original = frame.copy(deep=True)
    with AgentEpisode(
        task_id="immutability",
        table="fixture.csv",
        split="dev",
        question="Count rows.",
        dirty_profile="low",
        dirty_frame=frame,
        expected_answer=2,
        config=AgentEnvironmentConfig(max_steps=10),
    ):
        pass
    pd.testing.assert_frame_equal(frame, original)

"""Bounded three-tool A3 agent environment with no oracle exposure."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .a1_corruption import generate_dirty_copy
from .canonical import CanonicalTable
from .sandbox import PersistentPythonSandbox, SandboxResult
from .trajectory import AgentTrajectory, ToolEvent
from .verifier import verify_result


A3_TOOLS = ("inspect_table", "run_python", "submit_answer")


class AgentEnvironmentError(ValueError):
    pass


@dataclass(frozen=True)
class AgentEnvironmentConfig:
    max_steps: int = 12
    inspect_head_rows: int = 5
    run_timeout_seconds: float = 5.0
    max_output_chars: int = 8000
    max_memory_mb: int = 2048
    max_cpu_seconds: int = 60

    def __post_init__(self) -> None:
        if not 10 <= self.max_steps <= 12:
            raise AgentEnvironmentError("A3 max_steps must be between 10 and 12.")
        if not 1 <= self.inspect_head_rows <= 10:
            raise AgentEnvironmentError("inspect_head_rows must be between 1 and 10.")
        if self.run_timeout_seconds <= 0:
            raise AgentEnvironmentError("run_timeout_seconds must be positive.")
        if self.max_output_chars < 100:
            raise AgentEnvironmentError("max_output_chars must be at least 100.")
        if self.max_memory_mb < 256 or self.max_cpu_seconds < 1:
            raise AgentEnvironmentError("Invalid sandbox resource limits.")

    @classmethod
    def from_json(cls, path: str | Path) -> "AgentEnvironmentConfig":
        try:
            value = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AgentEnvironmentError(f"Cannot load A3 config: {exc}") from exc
        if value.get("schema_version") != 1:
            raise AgentEnvironmentError("Unsupported A3 config schema version.")
        expected_tools = value.get("tools")
        if expected_tools != list(A3_TOOLS):
            raise AgentEnvironmentError("A3 must expose exactly the three declared tools.")
        if value.get("network_disabled") is not True:
            raise AgentEnvironmentError("A3 config must disable network access.")
        fields = {
            "max_steps",
            "inspect_head_rows",
            "run_timeout_seconds",
            "max_output_chars",
            "max_memory_mb",
            "max_cpu_seconds",
        }
        return cls(**{name: value[name] for name in fields})


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    content: str
    truncated: bool = False
    error_type: str | None = None


def _json_safe(value: Any) -> Any:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _truncate(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    marker = "\n...[output truncated]"
    return text[: max(0, limit - len(marker))] + marker, True


class AgentEpisode:
    """One task episode. The expected answer is private and never returned by tools."""

    def __init__(
        self,
        *,
        task_id: str,
        table: str,
        split: str,
        question: str,
        dirty_profile: str,
        dirty_frame: pd.DataFrame,
        expected_answer: Any,
        config: AgentEnvironmentConfig | None = None,
        runtime_root: str | Path | None = None,
    ) -> None:
        if not isinstance(dirty_frame, pd.DataFrame) or dirty_frame.empty:
            raise AgentEnvironmentError("dirty_frame must be a non-empty DataFrame.")
        self.config = config or AgentEnvironmentConfig()
        self._expected_answer = expected_answer
        self._source_frame = dirty_frame.copy(deep=True)
        resolved_runtime_root = Path(runtime_root or Path.cwd() / ".a3_runtime").resolve()
        resolved_runtime_root.mkdir(parents=True, exist_ok=True)
        self._temporary = tempfile.TemporaryDirectory(
            prefix="dirty-da-a3-",
            dir=resolved_runtime_root,
        )
        self._table_path = Path(self._temporary.name) / "table.csv"
        self._source_frame.to_csv(self._table_path, index=False, lineterminator="\n")
        if os.name == "posix":
            try:
                self._table_path.chmod(0o444)
            except OSError:
                pass
        self._sandbox: PersistentPythonSandbox | None = None
        self.trajectory = AgentTrajectory(
            task_id=task_id,
            table=table,
            split=split,
            question=question,
            dirty_profile=dirty_profile,
            max_steps=self.config.max_steps,
        )

    @classmethod
    def from_task_record(
        cls,
        record: Mapping[str, Any],
        *,
        tables_dir: str | Path,
        semantics_registry: Mapping[str, Any],
        profile_config: Mapping[str, Any],
        config: AgentEnvironmentConfig | None = None,
        runtime_root: str | Path | None = None,
    ) -> "AgentEpisode":
        required = {
            "task_id",
            "table",
            "split",
            "question",
            "dirty_profile",
            "dirty_fingerprint",
            "answer",
        }
        missing = required - set(record)
        if missing:
            raise AgentEnvironmentError(f"Task record is missing {sorted(missing)}.")
        table_name = record["table"]
        frame = pd.read_csv(Path(tables_dir) / table_name, low_memory=False)
        dirty = generate_dirty_copy(
            CanonicalTable(frame, name=Path(table_name).stem),
            table_name,
            semantics_registry,
            profile_config,
            record["dirty_profile"],
        )
        if dirty.fingerprint != record["dirty_fingerprint"]:
            raise AgentEnvironmentError("Dirty table fingerprint differs from the A2 record.")
        return cls(
            task_id=record["task_id"],
            table=table_name,
            split=record["split"],
            question=record["question"],
            dirty_profile=record["dirty_profile"],
            dirty_frame=dirty.result.dirty,
            expected_answer=record["answer"],
            config=config,
            runtime_root=runtime_root,
        )

    @property
    def question(self) -> str:
        return self.trajectory.question

    @property
    def terminal(self) -> bool:
        return self.trajectory.terminal_status != "running"

    @property
    def tools(self) -> tuple[str, str, str]:
        return A3_TOOLS

    @property
    def sandbox_resource_limits_active(self) -> bool | None:
        """Report the worker's verified POSIX hard-limit state after first use."""

        if self._sandbox is None:
            return None
        return self._sandbox.posix_resource_limits_active

    def tool_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "inspect_table",
                "description": "Show columns, dtypes, missing counts, and a bounded head preview.",
                "parameters": {
                    "type": "object",
                    "properties": {"head_rows": {"type": "integer", "minimum": 1, "maximum": 10}},
                    "additionalProperties": False,
                },
            },
            {
                "name": "run_python",
                "description": "Run bounded Python against persistent dataframe variable df.",
                "parameters": {
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "submit_answer",
                "description": "Submit the final JSON-compatible answer for strict verification.",
                "parameters": {
                    "type": "object",
                    "properties": {"answer": {}},
                    "required": ["answer"],
                    "additionalProperties": False,
                },
            },
        ]

    def _inspect(self, arguments: Mapping[str, Any]) -> ToolResult:
        if set(arguments) - {"head_rows"}:
            return ToolResult(False, "inspect_table received unsupported arguments.", error_type="ArgumentsError")
        rows = arguments.get("head_rows", self.config.inspect_head_rows)
        if isinstance(rows, bool) or not isinstance(rows, int) or not 1 <= rows <= 10:
            return ToolResult(False, "head_rows must be an integer from 1 to 10.", error_type="ArgumentsError")
        frame = self._source_frame
        payload = {
            "row_count": int(len(frame)),
            "columns": [
                {
                    "name": str(column),
                    "dtype": str(frame[column].dtype),
                    "missing": int(frame[column].isna().sum()),
                }
                for column in frame.columns
            ],
            "head": _json_safe(frame.head(rows).to_dict(orient="records")),
        }
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        content, truncated = _truncate(text, self.config.max_output_chars)
        return ToolResult(True, content, truncated)

    def _run_python(self, arguments: Mapping[str, Any]) -> ToolResult:
        if set(arguments) != {"code"}:
            return ToolResult(False, "run_python requires only the code argument.", error_type="ArgumentsError")
        if self._sandbox is None:
            self._sandbox = PersistentPythonSandbox(
                self._table_path,
                timeout_seconds=self.config.run_timeout_seconds,
                max_output_chars=self.config.max_output_chars,
                memory_mb=self.config.max_memory_mb,
                cpu_seconds=self.config.max_cpu_seconds,
            )
        result: SandboxResult = self._sandbox.execute(arguments["code"])
        return ToolResult(result.ok, result.content, result.truncated, result.error_type)

    def _submit(self, arguments: Mapping[str, Any]) -> ToolResult:
        if set(arguments) != {"answer"}:
            return ToolResult(False, "submit_answer requires only the answer argument.", error_type="ArgumentsError")
        submitted = arguments["answer"]
        verification = verify_result(self._expected_answer, submitted)
        self.trajectory.submitted_answer = submitted
        if verification.passed:
            return ToolResult(True, "Answer accepted.")
        return ToolResult(False, "Answer rejected.", error_type="VerificationError")

    def call_tool(
        self,
        tool: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        assistant_content: str = "",
    ) -> ToolResult:
        if self.terminal:
            return ToolResult(False, "Episode has already terminated.", error_type="EpisodeTerminated")
        if not isinstance(tool, str) or not tool:
            raise TypeError("tool must be a non-empty string.")
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, Mapping):
            raise TypeError("arguments must be a mapping.")
        normalized_arguments = dict(arguments)
        if tool == "inspect_table":
            result = self._inspect(normalized_arguments)
        elif tool == "run_python":
            result = self._run_python(normalized_arguments)
        elif tool == "submit_answer":
            result = self._submit(normalized_arguments)
        else:
            result = ToolResult(False, f"Unknown tool: {tool}", error_type="UnknownTool")
        event = ToolEvent(
            step=len(self.trajectory.events) + 1,
            assistant_content=assistant_content,
            tool=tool,
            arguments=normalized_arguments,
            ok=result.ok,
            content=result.content,
            truncated=result.truncated,
            error_type=result.error_type,
        )
        self.trajectory.append(event)
        if tool == "submit_answer":
            self.trajectory.terminal_status = "accepted" if result.ok else "rejected"
        if (
            self.trajectory.terminal_status == "running"
            and len(self.trajectory.events) >= self.config.max_steps
        ):
            self.trajectory.terminal_status = "max_steps"
        if result.error_type in {"TimeoutError", "SandboxClosed", "ProtocolError"}:
            self.trajectory.terminal_status = "sandbox_error"
        return result

    def close(self) -> None:
        if self._sandbox is not None:
            self._sandbox.close()
            self._sandbox = None
        if os.name == "posix":
            try:
                self._table_path.chmod(0o644)
            except OSError:
                pass
        attempts = 20 if os.name == "nt" else 1
        for attempt in range(attempts):
            try:
                self._temporary.cleanup()
                break
            except PermissionError:
                if attempt == attempts - 1:
                    raise
                # Windows can retain a just-closed child-process handle for a
                # short interval. Retry cleanup; Linux production does not use
                # this compatibility branch.
                time.sleep(0.05)

    def __enter__(self) -> "AgentEpisode":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

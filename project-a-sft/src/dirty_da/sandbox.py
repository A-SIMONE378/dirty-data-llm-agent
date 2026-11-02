"""Restricted persistent subprocess for A3 ``run_python`` tool calls."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
from typing import Any


@dataclass(frozen=True)
class SandboxResult:
    ok: bool
    content: str
    truncated: bool
    error_type: str | None = None


def _truncate(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    marker = "\n...[output truncated]"
    return text[: max(0, limit - len(marker))] + marker, True


class PersistentPythonSandbox:
    """Run dataframe analysis in one stateful, isolated child interpreter."""

    def __init__(
        self,
        table_path: str | Path,
        *,
        timeout_seconds: float = 5.0,
        max_output_chars: int = 8000,
        memory_mb: int = 2048,
        cpu_seconds: int = 60,
    ) -> None:
        self.table_path = Path(table_path).resolve()
        if not self.table_path.is_file():
            raise FileNotFoundError(self.table_path)
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        if max_output_chars < 100:
            raise ValueError("max_output_chars must be at least 100.")
        if memory_mb < 256 or cpu_seconds < 1:
            raise ValueError("Sandbox resource limits are too small.")
        self.timeout_seconds = float(timeout_seconds)
        self.max_output_chars = int(max_output_chars)
        worker = Path(__file__).with_name("_sandbox_worker.py").resolve()
        environment = {
            key: value
            for key, value in os.environ.items()
            if key.upper() in {"PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP"}
        }
        environment.update(
            {
                "PYTHONHASHSEED": "0",
                "PYTHONIOENCODING": "utf-8",
                "PYTHONUNBUFFERED": "1",
            }
        )
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self._process = subprocess.Popen(
            [
                sys.executable,
                "-I",
                str(worker),
                str(self.table_path),
                "--memory-mb",
                str(memory_mb),
                "--cpu-seconds",
                str(cpu_seconds),
            ],
            cwd=str(self.table_path.parent),
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=creationflags,
        )
        # Importing pandas in a fresh isolated interpreter can take several
        # seconds on a cold Windows filesystem.  This startup allowance is
        # separate from the per-tool execution timeout enforced below.
        ready = self._readline(max(30.0, self.timeout_seconds))
        if ready is None:
            self._terminate()
            raise TimeoutError("Sandbox worker did not start before the timeout.")
        try:
            message = json.loads(ready)
        except json.JSONDecodeError as exc:
            details = self._read_stderr()
            self._terminate()
            raise RuntimeError(f"Sandbox worker startup failed: {details}") from exc
        if not isinstance(message, dict) or message.get("ready") is not True:
            self._terminate()
            raise RuntimeError("Sandbox worker returned an invalid startup message.")
        limits_active = message.get("posix_resource_limits_active")
        if not isinstance(limits_active, bool):
            self._terminate()
            raise RuntimeError("Sandbox worker did not report resource-limit status.")
        self.posix_resource_limits_active = limits_active

    @property
    def alive(self) -> bool:
        return self._process.poll() is None

    def _readline(self, timeout: float) -> str | None:
        assert self._process.stdout is not None
        result: queue.Queue[str] = queue.Queue(maxsize=1)

        def reader() -> None:
            result.put(self._process.stdout.readline())

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        try:
            line = result.get(timeout=timeout)
        except queue.Empty:
            return None
        return line or None

    def _read_stderr(self) -> str:
        if self._process.stderr is None or self.alive:
            return "no details"
        return self._process.stderr.read().strip() or "no details"

    def execute(self, code: str) -> SandboxResult:
        if not isinstance(code, str) or not code.strip():
            return SandboxResult(False, "code must be a non-empty string.", False, "ValueError")
        if not self.alive:
            return SandboxResult(False, "Sandbox worker is not running.", False, "SandboxClosed")
        assert self._process.stdin is not None
        request = json.dumps({"code": code}, ensure_ascii=False)
        try:
            self._process.stdin.write(request + "\n")
            self._process.stdin.flush()
        except (BrokenPipeError, OSError):
            return SandboxResult(False, "Sandbox worker terminated.", False, "SandboxClosed")
        line = self._readline(self.timeout_seconds)
        if line is None:
            self._terminate()
            return SandboxResult(
                False,
                f"run_python exceeded {self.timeout_seconds:g} seconds; sandbox terminated.",
                False,
                "TimeoutError",
            )
        try:
            response: dict[str, Any] = json.loads(line)
        except json.JSONDecodeError:
            self._terminate()
            return SandboxResult(False, "Sandbox returned malformed output.", False, "ProtocolError")
        content, truncated = _truncate(str(response.get("content", "")), self.max_output_chars)
        return SandboxResult(
            bool(response.get("ok")),
            content,
            truncated,
            response.get("error_type"),
        )

    def _terminate(self) -> None:
        if self.alive:
            self._process.kill()
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass

    def close(self) -> None:
        self._terminate()
        for stream in (self._process.stdin, self._process.stdout, self._process.stderr):
            if stream is not None:
                stream.close()

    def __enter__(self) -> "PersistentPythonSandbox":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

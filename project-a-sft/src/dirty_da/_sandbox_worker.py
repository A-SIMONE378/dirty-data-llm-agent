"""Private persistent Python worker used by :mod:`dirty_da.sandbox`.

This file is launched in an isolated child interpreter.  It intentionally has
no dependency on the rest of the package so ``python -I`` can execute it by
absolute path without adding the repository to ``sys.path``.
"""

from __future__ import annotations

import argparse
import ast
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np
import pandas as pd


FORBIDDEN_NAMES = {
    "__import__",
    "breakpoint",
    "compile",
    "delattr",
    "eval",
    "exec",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
}
FORBIDDEN_ATTRIBUTES = {
    "chmod",
    "chown",
    "environ",
    "getenv",
    "listdir",
    "mkdir",
    "open",
    "popen",
    "putenv",
    "remove",
    "rename",
    "rmdir",
    "scandir",
    "spawn",
    "system",
    "unlink",
    "walk",
}
FORBIDDEN_NODES = (ast.Global, ast.Nonlocal)
ALLOWED_IMPORT_MODULES = {"pandas", "numpy"}


class SafetyVisitor(ast.NodeVisitor):
    def visit(self, node: ast.AST) -> None:
        if isinstance(node, FORBIDDEN_NODES):
            raise PermissionError(f"{type(node).__name__} is not allowed in run_python.")
        super().visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name not in ALLOWED_IMPORT_MODULES:
                raise PermissionError(
                    f"Import of {alias.name!r} is not allowed in run_python."
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        raise PermissionError("ImportFrom is not allowed in run_python.")

    def visit_Name(self, node: ast.Name) -> None:
        if node.id.startswith("__") or node.id in FORBIDDEN_NAMES:
            raise PermissionError(f"Name {node.id!r} is not allowed in run_python.")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("_") or node.attr in FORBIDDEN_ATTRIBUTES:
            raise PermissionError(
                f"Attribute {node.attr!r} is not allowed in run_python."
            )
        self.generic_visit(node)


def _set_resource_limits(memory_mb: int, cpu_seconds: int) -> bool:
    if os.name != "posix":
        return False
    import resource

    memory_bytes = memory_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    return (
        resource.getrlimit(resource.RLIMIT_AS) == (memory_bytes, memory_bytes)
        and resource.getrlimit(resource.RLIMIT_CPU)
        == (cpu_seconds, cpu_seconds + 1)
        and resource.getrlimit(resource.RLIMIT_FSIZE) == (0, 0)
    )


def _install_audit_hook(table_path: Path) -> None:
    allowed = table_path.resolve()
    # Pandas and NumPy may lazily read metadata or extension modules after the
    # worker is ready.  Permit read-only access inside the interpreter install,
    # but never to another project/data path.
    runtime_roots = tuple(
        {
            Path(sys.base_prefix).resolve(),
            Path(sys.prefix).resolve(),
        }
    )

    def inside_runtime(candidate: Path) -> bool:
        return any(candidate == root or candidate.is_relative_to(root) for root in runtime_roots)

    def audit(event: str, args: tuple[object, ...]) -> None:
        if event == "open":
            raw_path = args[0] if args else None
            mode = args[1] if len(args) > 1 else "r"
            if not isinstance(raw_path, (str, bytes, os.PathLike)):
                raise PermissionError("File descriptor access is disabled.")
            candidate = Path(raw_path).resolve()
            read_only = isinstance(mode, str) and not any(
                flag in mode for flag in ("w", "a", "x", "+")
            )
            if not read_only or (candidate != allowed and not inside_runtime(candidate)):
                raise PermissionError("Only the current task CSV is readable.")
        denied_prefixes = (
            "socket.",
            "subprocess.",
            "os.system",
            "os.spawn",
            "ctypes.",
            "winreg.",
        )
        if event.startswith(denied_prefixes):
            raise PermissionError("Network and child-process access are disabled.")

    sys.addaudithook(audit)


def _execute(code: str, namespace: dict[str, object]) -> dict[str, object]:
    tree = ast.parse(code, mode="exec")
    SafetyVisitor().visit(tree)
    body = list(tree.body)
    expression = body.pop() if body and isinstance(body[-1], ast.Expr) else None
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        if body:
            module = ast.Module(body=body, type_ignores=[])
            exec(compile(module, "<run_python>", "exec"), namespace, namespace)
        value = None
        if expression is not None:
            value = eval(
                compile(ast.Expression(expression.value), "<run_python>", "eval"),
                namespace,
                namespace,
            )
            namespace["_"] = value
    output = stdout.getvalue()
    error_output = stderr.getvalue()
    if error_output:
        output += ("\n" if output else "") + error_output
    if expression is not None and value is not None:
        display_value = value.item() if isinstance(value, np.generic) else value
        output += ("\n" if output else "") + repr(display_value)
    return {"ok": True, "content": output or "(no output)", "error_type": None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("table_path")
    parser.add_argument("--memory-mb", type=int, required=True)
    parser.add_argument("--cpu-seconds", type=int, required=True)
    args = parser.parse_args()
    table_path = Path(args.table_path).resolve()
    frame = pd.read_csv(table_path, low_memory=False)
    namespace: dict[str, object] = {"df": frame, "pd": pd, "np": np}
    resource_limits_active = _set_resource_limits(args.memory_mb, args.cpu_seconds)
    _install_audit_hook(table_path)
    print(
        json.dumps(
            {
                "ready": True,
                "posix_resource_limits_active": resource_limits_active,
            }
        ),
        flush=True,
    )
    for line in sys.stdin:
        try:
            request = json.loads(line)
            code = request.get("code")
            if not isinstance(code, str) or not code.strip():
                raise ValueError("code must be a non-empty string.")
            response = _execute(code, namespace)
        except BaseException as exc:  # worker must return structured failures
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            response = {
                "ok": False,
                "content": str(exc) or type(exc).__name__,
                "error_type": type(exc).__name__,
            }
        print(json.dumps(response, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

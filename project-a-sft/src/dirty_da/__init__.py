"""Project A: verifiable dirty-data analysis primitives."""
from .canonical import CanonicalTable
from .corruption import apply_corruption, apply_corruptions, CorruptionResult
from .reference_clean import clean_table
from .task_spec import TaskSpec, TaskSpecError
from .executor import execute_task, ExecutionError
from .verifier import verify_result, VerificationResult
from .a1_catalog import A1CatalogError, audit_a1_catalog, load_a1_catalog
from .a1_corruption import (
    A1CorruptionPlanError,
    PlannedDirtyCopy,
    generate_dirty_copy,
    load_corruption_profiles,
)
from .agent_env import (
    A3_TOOLS,
    AgentEnvironmentConfig,
    AgentEnvironmentError,
    AgentEpisode,
    ToolResult,
)
from .sandbox import PersistentPythonSandbox, SandboxResult
from .trajectory import AgentTrajectory, ToolEvent, validate_trajectory_record
from .agent_loop import (
    AgentLoopError,
    AssistantAction,
    DEFAULT_SYSTEM_PROMPT,
    ModelAdapter,
    run_agent_episode,
)

__all__ = [
    "CanonicalTable", "apply_corruption", "apply_corruptions", "CorruptionResult", "clean_table",
    "TaskSpec", "TaskSpecError", "execute_task", "ExecutionError",
    "verify_result", "VerificationResult",
    "A1CatalogError", "audit_a1_catalog", "load_a1_catalog",
    "A1CorruptionPlanError", "PlannedDirtyCopy", "generate_dirty_copy",
    "load_corruption_profiles",
    "A3_TOOLS", "AgentEnvironmentConfig", "AgentEnvironmentError",
    "AgentEpisode", "ToolResult", "PersistentPythonSandbox", "SandboxResult",
    "AgentTrajectory", "ToolEvent", "validate_trajectory_record",
    "AgentLoopError", "AssistantAction", "DEFAULT_SYSTEM_PROMPT",
    "ModelAdapter", "run_agent_episode",
]

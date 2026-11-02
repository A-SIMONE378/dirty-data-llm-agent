"""B1 acceptance audit: repeated deterministic rollouts and replay consistency."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dirty_da.agent_env import AgentEnvironmentConfig  # noqa: E402
from dirty_da.b_env import BEnvironment  # noqa: E402
from dirty_da.b_reward import RewardConfig  # noqa: E402


FRAME = pd.DataFrame(
    {
        "group": ["a", "a", "b"],
        "value": [1.0, 2.0, 3.0],
        "optional": [None, "x", "y"],
    }
)
ACTIONS = [
    ("inspect_table", {"head_rows": 2}),
    ("run_python", {"code": "len(df)"}),
    ("submit_answer", {"answer": 3}),
]


def _make_environment(rollout_timeout_seconds: float) -> BEnvironment:
    return BEnvironment.from_frame(
        task_id="b1-audit",
        question="Return the number of rows.",
        dirty_frame=FRAME,
        expected_answer=3,
        config=AgentEnvironmentConfig(max_steps=10),
        reward_config=RewardConfig(),
        runtime_root=ROOT / ".b1_runtime",
        rollout_timeout_seconds=rollout_timeout_seconds,
    )


def _rollout(_index: int, rollout_timeout_seconds: float) -> dict:
    with _make_environment(rollout_timeout_seconds) as environment:
        observation = environment.reset()
        if "expected_answer" in json.dumps(observation, ensure_ascii=False):
            raise AssertionError("reset exposed expected_answer.")
        reward = 0.0
        done = False
        steps = []
        for tool, arguments in ACTIONS:
            step = environment.step(tool, arguments)
            steps.append(step.observation)
            reward = step.reward
            done = step.done
            if done:
                break
        if not done:
            raise AssertionError("rollout did not terminate.")
        return {
            "reward": reward,
            "terminal_status": environment.episode.trajectory.terminal_status,
            "steps": steps,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=int, default=1000)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()
    if args.rollouts < 1 or args.parallel < 1:
        raise SystemExit("rollouts and parallel must be positive.")

    failures: list[str] = []
    results = []
    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = [
            executor.submit(_rollout, index, args.timeout)
            for index in range(args.rollouts)
        ]
        for index, future in enumerate(futures):
            try:
                result = future.result()
            except Exception as error:  # audit records, does not hide
                failures.append(f"rollout-{index}: {type(error).__name__}: {error}")
                continue
            results.append(result)
            if (
                result["reward"] != 1.0
                or result["terminal_status"] != "accepted"
            ):
                failures.append(
                    f"rollout-{index}: reward={result['reward']} "
                    f"status={result['terminal_status']}"
                )

    replay_results = []
    with _make_environment(args.timeout) as environment:
        first = environment.replay(ACTIONS)
        second = environment.replay(ACTIONS)
        replay_results = [first["reward"], second["reward"]]
        replay_equal = (
            first["reward"] == second["reward"]
            and first["terminal_status"] == second["terminal_status"]
            and [
                (event.get("tool"), event.get("arguments"))
                for event in first["trajectory"]["events"]
            ]
            == [
                (event.get("tool"), event.get("arguments"))
                for event in second["trajectory"]["events"]
            ]
        )

    summary = {
        "schema_version": 1,
        "rollouts_requested": args.rollouts,
        "rollouts_completed": len(results),
        "parallel": args.parallel,
        "all_rewards_one": all(result["reward"] == 1.0 for result in results),
        "all_terminal_accepted": all(
            result["terminal_status"] == "accepted" for result in results
        ),
        "replay_rewards": replay_results,
        "replay_consistent": replay_equal,
        "failures": failures[:20],
        "failure_count": len(failures),
        "b1_acceptance_pass": (
            len(results) == args.rollouts
            and not failures
            and replay_equal
        ),
    }
    output = ROOT / "artifacts" / "b1_environment_audit.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if not summary["b1_acceptance_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

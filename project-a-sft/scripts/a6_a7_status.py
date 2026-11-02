from pathlib import Path
from datetime import datetime
import json
import re
import subprocess

ROOT = Path("/root/project-a")
A6 = ROOT / "artifacts" / "a6_eval"
A7_TRAIN = ROOT / "artifacts" / "a7_sft"
A7_EVAL = ROOT / "artifacts" / "a7_eval"
A5_EVAL = ROOT / "artifacts" / "a5_eval"


def sh(command):
    return subprocess.run(
        command, shell=True, capture_output=True, text=True, errors="replace"
    ).stdout.strip()


def load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def line_count(path):
    try:
        return sum(1 for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip())
    except Exception:
        return 0


def training_progress(key):
    path = Path(f"/root/a7_train_{key}.log")
    if not path.exists():
        return None
    text = path.read_bytes()[-200_000:].decode("utf-8", "replace")
    matches = re.findall(r"(\d+)\s*/\s*(\d+)\s*\[", text)
    step = total = None
    if matches:
        step, total = int(matches[-1][0]), int(matches[-1][1])
    losses = re.findall(r"'loss':\s*([0-9.]+)", text)
    loss = losses[-1] if losses else None
    return step, total, loss


def eval_progress(directory):
    directory = Path(directory)
    summary = load_json(directory / "summary.json")
    manifest = load_json(directory / "manifest.json")
    attempted = line_count(directory / "results.jsonl")
    required = int(manifest.get("task_count", 0)) if manifest else 0
    if summary:
        return summary, attempted, required
    return None, attempted, required


print("=" * 100)
print("host:", sh("hostname"))
print("time:", datetime.now().isoformat(timespec="seconds"))
print("=" * 100)

print("\n[SCREENS]")
screen = sh("screen -ls")
print(screen if screen else "(no screen sessions)")

print("\n[ACTIVE PROCESSES]")
print(sh(
    "ps -eo pid,etime,%cpu,%mem,stat,cmd | "
    "grep -E 'run_a5_sft|run_a3_qwen_acceptance|a7_pipeline' | grep -v grep"
) or "(none)")

print("\n[GPU]")
print(sh(
    "nvidia-smi --query-gpu=index,name,memory.used,memory.total,"
    "utilization.gpu,power.draw,temperature.gpu --format=csv,noheader"
))

print("\n[DOWNLOADS]")
for name in ("Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B"):
    path = Path("/hy-tmp") / name
    if not path.exists():
        print(f"{name:12s} missing")
        continue
    size = sh(f"du -sh {path} 2>/dev/null").split()[0]
    ready = (path / "model.safetensors.index.json").exists() or any(path.glob("model*.safetensors"))
    print(f"{name:12s} {size:>8s}  ready={ready}")

print("\n[A5 DEV BASELINE]")
for name in ("qwen3_1_7b_base_dev_v1", "qwen3_1_7b_sft_dev_v1"):
    summary, attempted, required = eval_progress(A5_EVAL / name)
    if summary:
        print(
            f"{name:32s} {summary.get('terminal_status_counts')} "
            f"accepted={summary.get('accepted_accuracy'):.3f} "
            f"({summary.get('tasks_attempted')}/{summary.get('tasks_required')})"
        )
    else:
        print(f"{name:32s} not started ({attempted}/{required})")

print("\n[A6 TEST + DOMAIN]")
for directory in sorted(A6.glob("qwen3_*_test_domain_v1")):
    summary, attempted, required = eval_progress(directory)
    if summary and summary.get("run_complete"):
        print(
            f"{directory.name:42s} accepted={summary.get('accepted_accuracy'):.3f} "
            f"({summary.get('tasks_attempted')}/{summary.get('tasks_required')}) "
            f"{summary.get('terminal_status_counts')}"
        )
    else:
        print(f"{directory.name:42s} running {attempted}/{required}")

print("\n[A6 COMPARISONS]")
for path in sorted(A6.glob("comparison_*.json")):
    value = load_json(path)
    if not value:
        continue
    print(
        f"{path.name:24s} base={value.get('base_accuracy'):.3f} "
        f"sft={value.get('sft_accuracy'):.3f} "
        f"net={value.get('net_accepted_gain')} "
        f"CI={value.get('accuracy_difference_bootstrap_95')} "
        f"integrity={value.get('a6_integrity_pass')}"
    )
    for split, row in value.get("by_split", {}).items():
        print(
            f"    {split:22s} base={row['base_accuracy']:.3f} "
            f"sft={row['sft_accuracy']:.3f} net={row['net_accepted_gain']}"
        )

print("\n[A7 TRAINING]")
training = {
    "success_only": "qwen3_1_7b_success_only_v1",
    "recovery_augmented": "qwen3_1_7b_recovery_augmented_v1",
    "scale_500": "qwen3_1_7b_scale_500_v1",
    "scale_1000": "qwen3_1_7b_scale_1000_v1",
    "scale_2000": "qwen3_1_7b_scale_2000_v1",
    "4b_main": "qwen3_4b_main_14b_v1",
}
for key, dirname in training.items():
    manifest = load_json(A7_TRAIN / dirname / "run_manifest.json")
    step, total, loss = training_progress(key) or (None, None, None)
    if manifest:
        print(
            f"{key:20s} DONE  records={manifest.get('record_count')} "
            f"loss={manifest.get('train_metrics', {}).get('train_loss')}"
        )
    elif step is not None:
        pct = 100 * step / total if total else 0
        print(f"{key:20s} {step}/{total} ({pct:5.1f}%) loss={loss}")
    else:
        print(f"{key:20s} not started")

print("\n[A7 DEV EVAL]")
for directory in sorted(A7_EVAL.glob("qwen3_*_dev_v1")):
    summary, attempted, required = eval_progress(directory)
    if summary and summary.get("run_complete"):
        print(
            f"{directory.name:42s} accepted={summary.get('accepted_accuracy'):.3f} "
            f"({summary.get('tasks_attempted')}/{summary.get('tasks_required')})"
        )
    else:
        print(f"{directory.name:42s} running {attempted}/{required}")

print("\n[A7 COMPARISONS]")
for path in sorted(A7_EVAL.glob("comparison_*.json")):
    value = load_json(path)
    if not value:
        continue
    print(
        f"{path.name:32s} base={value.get('base_accuracy'):.3f} "
        f"sft={value.get('sft_accuracy'):.3f} "
        f"net={value.get('net_accepted_gain')} "
        f"gate={value.get('a5_dev_gate_pass')}"
    )

print("\n[LATEST LOG LINES]")
for name in (
    "a6_base_17b.log",
    "a6_sft_17b.log",
    "a7_train_success_only.log",
    "a7_train_recovery_augmented.log",
    "a7_train_scale_500.log",
    "a7_train_scale_1000.log",
    "a7_train_scale_2000.log",
    "a7_train_4b_main.log",
):
    path = Path("/root") / name
    if not path.exists():
        continue
    text = path.read_bytes()[-800:].decode("utf-8", "replace").replace("\r", "\n")
    lines = [line for line in text.splitlines() if line.strip()]
    print(f"{name:36s} {lines[-1][:180] if lines else ''}")

# 想每 30 秒自动刷新时，取消下面注释：
# import time
# from IPython.display import clear_output
# while True:
#     clear_output(wait=True)
#     exec(open("/root/project-a/scripts/a6_a7_status.py").read())
#     time.sleep(30)

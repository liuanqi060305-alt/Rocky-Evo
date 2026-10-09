#!/usr/bin/env python3
"""Monitor and verify the official V3 all-188 OpenPI training run."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time


BASE = Path(os.environ.get("BASE", "/share/home/tm904895221620000/a1173895530/openpi训练"))
CONFIG = "pi05_xtrainer_full_v3_all188"
EXPERIMENT = "plug_v3_all188_full_h100_20261009_50k"
DATASET = "xtrainer/plug_and_unplug_task_v3_all188"
RUN_DIR = BASE / "checkpoints" / CONFIG / EXPERIMENT
TRAIN_LOG = BASE / "logs" / f"{EXPERIMENT}.log"
RUNTIME_DIR = BASE / "run"
PID_PATH = RUNTIME_DIR / f"{EXPERIMENT}.pid"
NORM_SHA_PATH = RUNTIME_DIR / f"{EXPERIMENT}.norm_stats.sha256"
STATUS_PATH = RUNTIME_DIR / f"{EXPERIMENT}.watch_status.json"
REPORT_PATH = RUNTIME_DIR / f"{EXPERIMENT}.final_verification.json"
SUCCESS_MARKER = RUNTIME_DIR / f"{EXPERIMENT}.final_verification.success"
FAILURE_MARKER = RUNTIME_DIR / f"{EXPERIMENT}.final_verification.failed"
EXPECTED_FINAL_STEP = 49_999
EXPECTED_LAST_LOGGED_STEP = 49_900
POLL_SECONDS = int(os.environ.get("POLL_SECONDS", "300"))


def now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)


def read_log() -> str:
    try:
        return TRAIN_LOG.read_text(errors="replace")
    except FileNotFoundError:
        return ""


def training_pid() -> int | None:
    try:
        return int(PID_PATH.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None


def training_process_running(pid: int | None = None) -> bool:
    pid = training_pid() if pid is None else pid
    if pid is None:
        return False
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return False
    return "scripts/train.py" in command and EXPERIMENT in command


def logged_steps(text: str) -> list[dict]:
    rows = []
    pattern = re.compile(
        r"Step\s+(\d+):\s+grad_norm=([^,]+),\s+loss=([^,]+),\s+param_norm=([^\s]+)"
    )
    for match in pattern.finditer(text):
        try:
            row = {
                "step": int(match.group(1)),
                "grad_norm": float(match.group(2)),
                "loss": float(match.group(3)),
                "param_norm": float(match.group(4)),
            }
        except ValueError:
            continue
        rows.append(row)
    return rows


def checkpoint_steps() -> list[int]:
    if not RUN_DIR.is_dir():
        return []
    return sorted(int(path.name) for path in RUN_DIR.iterdir() if path.is_dir() and path.name.isdigit())


def expected_norm_sha256() -> str | None:
    try:
        value = NORM_SHA_PATH.read_text().split()[0]
    except (FileNotFoundError, IndexError):
        return None
    return value if re.fullmatch(r"[0-9a-f]{64}", value) else None


def write_live_status() -> None:
    pid = training_pid()
    rows = logged_steps(read_log())
    latest = rows[-1] if rows else None
    atomic_json(
        STATUS_PATH,
        {
            "schema_version": 1,
            "observed_at": now(),
            "experiment": EXPERIMENT,
            "training_pid": pid,
            "process_running": training_process_running(pid),
            "latest_metrics": latest,
            "progress_percent": round(100 * latest["step"] / EXPECTED_FINAL_STEP, 3) if latest else 0.0,
            "checkpoint_steps": checkpoint_steps(),
        },
    )


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_orbax() -> tuple[bool, str]:
    code = r'''import pathlib
import sys
import orbax.checkpoint as ocp

run = pathlib.Path(sys.argv[1])
expected = int(sys.argv[2])
manager = ocp.CheckpointManager(run)
assert manager.latest_step() == expected, (manager.latest_step(), expected)
ocp.PyTreeCheckpointer().metadata(run / str(expected) / "params")
ocp.PyTreeCheckpointer().metadata(run / str(expected) / "train_state")
manager.close()
print("ORBAX_METADATA_OK")
'''
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu"})
    result = subprocess.run(
        [str(BASE / "openpi/.venv/bin/python"), "-c", code, str(RUN_DIR), str(EXPECTED_FINAL_STEP)],
        text=True,
        capture_output=True,
        env=env,
        timeout=900,
    )
    detail = (result.stdout + result.stderr).strip()[-8000:]
    return result.returncode == 0 and "ORBAX_METADATA_OK" in result.stdout, detail


def final_verification() -> dict:
    text = read_log()
    rows = logged_steps(text)
    steps = checkpoint_steps()
    pid = training_pid()
    final_dir = RUN_DIR / str(EXPECTED_FINAL_STEP)
    temp_dirs = sorted(path.name for path in RUN_DIR.glob("*.orbax-checkpoint-tmp-*")) if RUN_DIR.exists() else []
    required = [
        final_dir / "_CHECKPOINT_METADATA",
        final_dir / "params/_METADATA",
        final_dir / "params/manifest.ocdbt",
        final_dir / "train_state/_METADATA",
        final_dir / "train_state/manifest.ocdbt",
    ]
    norm_path = final_dir / "assets" / DATASET / "norm_stats.json"
    expected_norm = expected_norm_sha256()
    fatal_patterns = {
        "traceback": r"Traceback \(most recent call last\)",
        "out_of_memory": r"(?:out of memory|RESOURCE_EXHAUSTED)",
        "uncaught_exception": r"(?:^|\n)(?:ERROR|Exception):",
    }
    fatal_matches = {
        name: bool(re.search(pattern, text, flags=re.IGNORECASE))
        for name, pattern in fatal_patterns.items()
    }
    finite_metrics = bool(rows) and all(
        math.isfinite(row[key]) for row in rows for key in ("grad_norm", "loss", "param_norm")
    )
    orbax_ok = False
    orbax_detail = "final checkpoint is absent"
    if final_dir.is_dir():
        try:
            orbax_ok, orbax_detail = verify_orbax()
        except Exception as exc:
            orbax_detail = f"{type(exc).__name__}: {exc}"
    checks = {
        "training_process_stopped": not training_process_running(pid),
        "final_step_logged": bool(rows) and rows[-1]["step"] >= EXPECTED_LAST_LOGGED_STEP,
        "metrics_finite": finite_metrics,
        "final_checkpoint_present": EXPECTED_FINAL_STEP in steps,
        "no_temporary_checkpoint_directories": not temp_dirs,
        "required_checkpoint_files_present": all(path.exists() for path in required),
        "norm_stats_hash_available": expected_norm is not None,
        "norm_stats_hash_matches": expected_norm is not None and sha256(norm_path) == expected_norm,
        "orbax_metadata_readable": orbax_ok,
        "no_fatal_log_markers": not any(fatal_matches.values()),
    }
    return {
        "schema_version": 1,
        "verified_at": now(),
        "experiment": EXPERIMENT,
        "training_pid": pid,
        "expected_final_step": EXPECTED_FINAL_STEP,
        "latest_metrics": rows[-1] if rows else None,
        "logged_metric_count": len(rows),
        "checkpoint_steps": steps,
        "temporary_checkpoint_directories": temp_dirs,
        "final_checkpoint_size_bytes": sum(path.stat().st_size for path in final_dir.rglob("*") if path.is_file())
        if final_dir.is_dir()
        else 0,
        "expected_norm_stats_sha256": expected_norm,
        "checkpoint_norm_stats_sha256": sha256(norm_path),
        "fatal_log_markers": fatal_matches,
        "orbax_detail": orbax_detail,
        "checks": checks,
        "success": all(checks.values()),
    }


def main() -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    SUCCESS_MARKER.unlink(missing_ok=True)
    FAILURE_MARKER.unlink(missing_ok=True)
    while training_process_running():
        write_live_status()
        time.sleep(POLL_SECONDS)
    time.sleep(60)
    report = final_verification()
    atomic_json(REPORT_PATH, report)
    marker = SUCCESS_MARKER if report["success"] else FAILURE_MARKER
    marker.write_text(f"{now()}\n")
    write_live_status()


if __name__ == "__main__":
    main()

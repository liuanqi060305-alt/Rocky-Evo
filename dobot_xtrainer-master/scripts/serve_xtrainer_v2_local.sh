#!/usr/bin/env bash
set -euo pipefail

OPENPI_ROOT="${OPENPI_ROOT:-/home/iml/RockyEVO/openpi}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$OPENPI_ROOT/checkpoints/pi05_xtrainer_full_v2_107eps/plug_v2_107eps_full_h100_20260927_unlimited/29999}"
POLICY_CONFIG="${POLICY_CONFIG:-pi05_xtrainer_full_v2_107eps}"
PORT="${PORT:-8001}"
DEFAULT_PROMPT="${DEFAULT_PROMPT:-plug and unplug}"
CHECK_ONLY="${CHECK_ONLY:-0}"
NORM_STATS="$CHECKPOINT_DIR/assets/xtrainer/plug_and_unplug_task_v2_107eps/norm_stats.json"
EXPECTED_NORM_SHA256="5bcf29919fb9cf2eef523a8c609f335b67dc8dff4a4c52d09ef23126162248db"

if [[ ! -d "$CHECKPOINT_DIR/params" || ! -f "$CHECKPOINT_DIR/_CHECKPOINT_METADATA" ]]; then
    echo "V2 checkpoint is missing or incomplete: $CHECKPOINT_DIR" >&2
    echo "Download the complete 29999 directory first; see scripts/README.md section 7." >&2
    exit 1
fi
if [[ ! -f "$NORM_STATS" ]]; then
    echo "V2 norm stats are missing: $NORM_STATS" >&2
    exit 1
fi

ACTUAL_NORM_SHA256="$(sha256sum "$NORM_STATS" | awk '{print $1}')"
if [[ "$ACTUAL_NORM_SHA256" != "$EXPECTED_NORM_SHA256" ]]; then
    echo "V2 norm stats checksum mismatch." >&2
    echo "Expected: $EXPECTED_NORM_SHA256" >&2
    echo "Actual:   $ACTUAL_NORM_SHA256" >&2
    exit 1
fi
if [[ ! -x "$OPENPI_ROOT/.venv/bin/python" ]]; then
    echo "OpenPI virtual environment is missing: $OPENPI_ROOT/.venv" >&2
    exit 1
fi
if ! grep -Fq 'name="pi05_xtrainer_full_v2_107eps"' \
        "$OPENPI_ROOT/src/openpi/training/config.py"; then
    echo "OpenPI does not contain the V2 training/policy config: $POLICY_CONFIG" >&2
    exit 1
fi

if [[ "$CHECK_ONLY" == "1" ]]; then
    echo "V2 local policy preflight passed."
    echo "Policy config: $POLICY_CONFIG"
    echo "Checkpoint: $CHECKPOINT_DIR"
    echo "Norm stats SHA256: $ACTUAL_NORM_SHA256"
    exit 0
fi

RUNNING_PID="$(pgrep -f "[s]cripts/serve_policy.py.*--port=$PORT" | head -n 1 || true)"
if [[ -n "$RUNNING_PID" ]]; then
    RUNNING_CMD="$(tr '\0' ' ' < "/proc/$RUNNING_PID/cmdline")"
    if [[ "$RUNNING_CMD" == *"--policy.config=$POLICY_CONFIG"* &&
          "$RUNNING_CMD" == *"--policy.dir=$CHECKPOINT_DIR"* ]]; then
        echo "The requested V2 policy is already running on port $PORT (PID $RUNNING_PID)."
        exit 0
    fi
    echo "Port $PORT already has a different OpenPI policy (PID $RUNNING_PID)." >&2
    echo "Running command: $RUNNING_CMD" >&2
    echo "Stop that process yourself, then run this script again." >&2
    exit 1
fi
if ss -ltn "sport = :$PORT" | tail -n +2 | grep -q .; then
    echo "Port $PORT is occupied by another process; stop it or choose another PORT." >&2
    exit 1
fi

export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/home/iml/.cache/openpi}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export PYTHONUNBUFFERED=1

echo "Starting local X-Trainer V2 policy on 127.0.0.1:$PORT"
echo "Policy config: $POLICY_CONFIG"
echo "Checkpoint: $CHECKPOINT_DIR"
echo "Norm stats SHA256: $ACTUAL_NORM_SHA256"

cd "$OPENPI_ROOT"
exec .venv/bin/python scripts/serve_policy.py \
    --port="$PORT" \
    --default-prompt="$DEFAULT_PROMPT" \
    policy:checkpoint \
    --policy.config="$POLICY_CONFIG" \
    --policy.dir="$CHECKPOINT_DIR"

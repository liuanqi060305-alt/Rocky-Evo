#!/usr/bin/env bash
set -euo pipefail

ACCOUNT_ROOT="${ACCOUNT_ROOT:-/share/home/tm904895221620000/a1173895530}"
BASE="${BASE:-${ACCOUNT_ROOT}/openpi训练}"
OPENPI_ROOT="${OPENPI_ROOT:-${BASE}/openpi}"
ARCHIVE="${ACCOUNT_ROOT}/datasets/raw_snapshots/plug_v3_all188_20261009_raw.tar.zst"
ARCHIVE_SHA256="5f231d04933b2523b062cb80278c6928fcd9ca062f392462e0de6ddcb1afee68"
SNAP_ROOT="${ACCOUNT_ROOT}/datasets/raw_snapshots/plug_v3_all188_20261009"
RAW="${SNAP_ROOT}/plug_and_unplug_task"
PATCH="${BASE}/patches/pi05_xtrainer_full_v3_all188.patch"
TRAIN_CONFIG="pi05_xtrainer_full_v3_all188"
REPO_ID="xtrainer/plug_and_unplug_task_v3_all188"
EXP_NAME="plug_v3_all188_full_h100_20261009_50k"
LOG_ROOT="${BASE}/logs"
RUN_ROOT="${BASE}/run"
HF_LEROBOT_HOME="${BASE}/hf_lerobot"
CONVERTED="${HF_LEROBOT_HOME}/${REPO_ID}"
CHECKPOINT_DIR="${BASE}/checkpoints/${TRAIN_CONFIG}/${EXP_NAME}"
LOCK_FILE="${RUN_ROOT}/${EXP_NAME}.prepare.lock"

mkdir -p "$LOG_ROOT" "$RUN_ROOT" "$HF_LEROBOT_HOME"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "Another V3 preparation process holds $LOCK_FILE" >&2
    exit 1
fi

if [[ -e "$CHECKPOINT_DIR" ]]; then
    echo "Checkpoint directory already exists; refusing to overwrite: $CHECKPOINT_DIR" >&2
    exit 1
fi

test -d "$OPENPI_ROOT"
test -x "$OPENPI_ROOT/.venv/bin/python"
test -f "$ARCHIVE"
test -f "$PATCH"

actual_sha256="$(sha256sum "$ARCHIVE" | awk '{print $1}')"
if [[ "$actual_sha256" != "$ARCHIVE_SHA256" ]]; then
    echo "Archive SHA-256 mismatch: $actual_sha256" >&2
    exit 1
fi

rm -rf "${SNAP_ROOT}.extracting"
mkdir -p "${SNAP_ROOT}.extracting"
tar --zstd -xf "$ARCHIVE" -C "${SNAP_ROOT}.extracting"
test -f "${SNAP_ROOT}.extracting/plug_and_unplug_task/train_manifest.json"
test -f "${SNAP_ROOT}.extracting/plug_and_unplug_task/selection_manifest.json"
rm -rf "$SNAP_ROOT"
mv "${SNAP_ROOT}.extracting" "$SNAP_ROOT"

RAW="$RAW" "$OPENPI_ROOT/.venv/bin/python" - <<'PY'
import json
import os
from pathlib import Path

raw = Path(os.environ["RAW"])
manifest = json.loads((raw / "train_manifest.json").read_text())
selection = json.loads((raw / "selection_manifest.json").read_text())
ids = manifest["episodes"]
actual = sorted(p.name for p in (raw / "collect_data").iterdir() if p.is_dir())
assert len(ids) == len(set(ids)) == 188
assert sorted(ids) == actual
assert manifest["frame_count"] == selection["frame_count"] == 63495
assert selection["selection_policy"]["excluded_count"] == 0
print("raw snapshot verified: 188 episodes, 63495 frames")
PY

cd "$OPENPI_ROOT"
if ! grep -q 'name="pi05_xtrainer_full_v3_all188"' src/openpi/training/config.py; then
    git apply --check "$PATCH"
    git apply "$PATCH"
fi

"$OPENPI_ROOT/.venv/bin/python" - <<'PY'
from openpi.training.config import get_config

c = get_config("pi05_xtrainer_full_v3_all188")
assert c.data.repo_id == "xtrainer/plug_and_unplug_task_v3_all188"
assert c.batch_size == 32
assert c.num_train_steps == 50_000
assert c.save_interval == 1_000
assert c.keep_period == 5_000
assert c.lr_schedule.warmup_steps == 1_000
assert c.lr_schedule.decay_steps == 50_000
print("OpenPI V3 config verified")
PY

export HF_LEROBOT_HOME
export HF_DATASETS_CACHE="${BASE}/hf_datasets_cache"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export UV_OFFLINE=1
export PYTHONUNBUFFERED=1

"$OPENPI_ROOT/.venv/bin/python" examples/xtrainer/convert_xtrainer_data_to_lerobot.py \
    --raw-dir "$RAW" \
    --repo-id "$REPO_ID" \
    --generalization-split train \
    --prompt 'plug and unplug' \
    --fps 15 \
    2>&1 | tee "$LOG_ROOT/plug_v3_all188_20261009_convert.log"

CONVERTED="$CONVERTED" "$OPENPI_ROOT/.venv/bin/python" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["CONVERTED"])
meta = root / "meta"
info = json.loads((meta / "info.json").read_text())
tasks = [json.loads(line) for line in (meta / "tasks.jsonl").read_text().splitlines() if line]
assert info["total_episodes"] == 188, info
assert info["total_frames"] == 63495, info
assert info["fps"] == 15, info
assert tasks == [{"task_index": 0, "task": "plug and unplug"}], tasks
print("LeRobot dataset verified: 188 episodes, 63495 frames, 15 fps")
PY

TRAIN_CONFIG="$TRAIN_CONFIG" \
    "$OPENPI_ROOT/scripts/compute_norm_stats_xtrainer_szu.sh" \
    2>&1 | tee "$LOG_ROOT/plug_v3_all188_20261009_norm.log"

NORM_STATS="$OPENPI_ROOT/assets/${TRAIN_CONFIG}/${REPO_ID}/norm_stats.json"
NORM_STATS="$NORM_STATS" "$OPENPI_ROOT/.venv/bin/python" - <<'PY'
import json
import math
import os
from pathlib import Path

path = Path(os.environ["NORM_STATS"])
obj = json.loads(path.read_text())["norm_stats"]
assert set(obj) == {"state", "actions"}, obj.keys()
for key, stats in obj.items():
    for field, values in stats.items():
        assert all(math.isfinite(float(v)) for v in values), (key, field)
    assert len(stats["mean"]) == len(stats["std"]) == 14, key
    assert all(float(v) > 0 for v in stats["std"]), key
print(f"norm stats verified: {path}")
PY
sha256sum "$NORM_STATS" | tee "$RUN_ROOT/${EXP_NAME}.norm_stats.sha256"

TRAIN_LOG="$LOG_ROOT/${EXP_NAME}.log"
nohup setsid env \
    TRAIN_CONFIG="$TRAIN_CONFIG" \
    EXP_NAME="$EXP_NAME" \
    BATCH_SIZE=32 \
    WANDB_ENABLED=0 \
    WANDB_MODE=offline \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    "$OPENPI_ROOT/scripts/train_xtrainer_szu.sh" \
    >"$TRAIN_LOG" 2>&1 </dev/null &
train_pid=$!
printf '%s\n' "$train_pid" >"$RUN_ROOT/${EXP_NAME}.pid"

sleep 20
if ! kill -0 "$train_pid" 2>/dev/null; then
    echo "Training process exited during startup" >&2
    tail -n 100 "$TRAIN_LOG" >&2
    exit 1
fi

cat >"$RUN_ROOT/${EXP_NAME}.launch.json" <<JSON
{
  "status": "training_process_started",
  "pid": ${train_pid},
  "config": "${TRAIN_CONFIG}",
  "experiment": "${EXP_NAME}",
  "dataset": "${REPO_ID}",
  "episodes": 188,
  "frames": 63495,
  "steps": 50000,
  "expected_final_checkpoint": 49999,
  "archive_sha256": "${ARCHIVE_SHA256}",
  "norm_stats_sha256": "$(awk '{print $1}' "$RUN_ROOT/${EXP_NAME}.norm_stats.sha256")",
  "log": "${TRAIN_LOG}",
  "checkpoint_dir": "${CHECKPOINT_DIR}"
}
JSON

echo "Training process started: PID=$train_pid"
echo "Log: $TRAIN_LOG"
echo "Checkpoint directory: $CHECKPOINT_DIR"

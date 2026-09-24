"""统计任务下所有 episode 的 state / action 分布，产出 dataset_statistics.json。

供后续 Normalization 使用。V1（无时间戳）与 V2 数据都能读，靠 episode_io 统一。

用法：
    python scripts/dataset_stats.py                          # 用默认任务目录
    python scripts/dataset_stats.py --task-dir <任务目录>
    python scripts/dataset_stats.py --task-dir ... --out stats.json

只读，不修改任何采集数据。
"""
import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts import episode_io  # noqa: E402

DEFAULT_TASK_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "datasets", "plug_and_unplug_task")

DIM_NAMES = (
    ["left_j1", "left_j2", "left_j3", "left_j4", "left_j5", "left_j6", "left_gripper"]
    + ["right_j1", "right_j2", "right_j3", "right_j4", "right_j5", "right_j6", "right_gripper"]
)
DIM_UNITS = ["rad"] * 6 + ["normalized[0,1]"] + ["rad"] * 6 + ["normalized[0,1]"]


def per_dim_stats(arr):
    """逐维 min/max/mean/std + 分位数。arr: (T, 14)"""
    out = []
    for i in range(arr.shape[1]):
        col = arr[:, i]
        out.append({
            "index": i,
            "name": DIM_NAMES[i],
            "unit": DIM_UNITS[i],
            "min": float(col.min()),
            "max": float(col.max()),
            "mean": float(col.mean()),
            "std": float(col.std()),
            "q01": float(np.quantile(col, 0.01)),
            "q99": float(np.quantile(col, 0.99)),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-dir", default=DEFAULT_TASK_DIR, help="任务目录")
    ap.add_argument("--out", default=None, help="输出 json（默认写到任务目录下）")
    args = ap.parse_args()

    collect = os.path.join(args.task_dir, "collect_data")
    if not os.path.isdir(collect):
        raise SystemExit(f"[ERROR] 找不到 {collect}")

    eps = sorted(d for d in os.listdir(collect) if os.path.isdir(os.path.join(collect, d)))
    if not eps:
        raise SystemExit(f"[ERROR] {collect} 下没有 episode")

    states, actions, per_ep, skipped = [], [], [], []
    for name in eps:
        ep_dir = os.path.join(collect, name)
        try:
            ep = episode_io.load_episode(ep_dir)
        except Exception as e:
            skipped.append({"episode_id": name, "reason": str(e)})
            continue
        s, a = ep["state"], ep["action"]
        states.append(s)
        actions.append(a)
        meta = episode_io.load_meta(ep_dir)
        per_ep.append({
            "episode_id": name,
            "num_frames": len(ep["indices"]),
            "source_schema": ep["source_schema"],
            "fps_measured": round(episode_io.effective_fps(ep_dir, fallback=float("nan")), 3),
            "has_timestamps": ep["timestamp_ns"] is not None,
            "has_meta": meta is not None,
            "success": (meta or {}).get("success"),
            "valid": (meta or {}).get("valid"),
        })

    if not states:
        raise SystemExit("[ERROR] 没有可读的 episode")

    S = np.concatenate(states, axis=0)
    A = np.concatenate(actions, axis=0)

    doc = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "generator": "scripts/dataset_stats.py",
        "dataset_version": episode_io.DATASET_VERSION,
        "task_dir": os.path.abspath(args.task_dir),
        "num_episodes": len(states),
        "num_frames_total": int(S.shape[0]),
        "num_episodes_skipped": len(skipped),
        "skipped": skipped,
        # 明确声明这些字段在数据中不存在，避免下游误以为忘了统计
        "unavailable_fields": {
            "velocity_available": False,
            "ee_pose_available": False,
            "gripper_feedback": False,
        },
        "state": {
            "shape": list(S.shape),
            "dtype": str(S.dtype),
            "per_dim": per_dim_stats(S),
        },
        "action": {
            "shape": list(A.shape),
            "dtype": str(A.dtype),
            "semantics": "absolute joint target",
            "per_dim": per_dim_stats(A),
        },
        "episodes": per_ep,
    }

    out = args.out or os.path.join(args.task_dir, "dataset_statistics.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)

    print(f"[dataset_stats] {len(states)} 条 episode, {S.shape[0]} 帧 -> {out}")
    print(f"  跳过 {len(skipped)} 条")
    print("\n  idx  name             state[min,max]         action[min,max]        action_std")
    for i in range(14):
        s, a = doc["state"]["per_dim"][i], doc["action"]["per_dim"][i]
        print("  %2d  %-15s [%7.3f,%7.3f]   [%7.3f,%7.3f]   %.4f"
              % (i, s["name"], s["min"], s["max"], a["min"], a["max"], a["std"]))


if __name__ == "__main__":
    main()

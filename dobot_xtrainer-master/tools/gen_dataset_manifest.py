"""为一个 dataset version 生成 dataset_manifest.json。

只读 Raw，输出 manifest 到该任务目录下。不修改任何 episode。

用法：
    python tools/gen_dataset_manifest.py                          # 默认任务
    python tools/gen_dataset_manifest.py --task plug_and_unplug_task
    python tools/gen_dataset_manifest.py --task ... --notes "第一批正式数据"
"""
import argparse
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.data_paths import paths  # noqa: E402
from scripts import episode_io  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=os.environ.get("XTRAINER_TASK_NAME", "plug_and_unplug_task"))
    ap.add_argument("--notes", default="")
    ap.add_argument("--dry-run", action="store_true", help="只打印不写文件")
    args = ap.parse_args()

    task_dir = paths.task_dir(args.task)
    collect = paths.collect_dir(args.task)
    if not collect.is_dir():
        raise SystemExit(f"[ERROR] 找不到 {collect}")

    eps = sorted(d for d in collect.iterdir() if d.is_dir())
    episodes, valid_n, invalid_n, unlabeled_n, frames_total = [], 0, 0, 0, 0

    for d in eps:
        meta = episode_io.load_meta(str(d))
        try:
            ep = episode_io.load_episode(str(d))
            n_frames, schema = len(ep["indices"]), ep["source_schema"]
            fps = episode_io.effective_fps(str(d), fallback=float("nan"))
        except Exception as e:
            episodes.append({"episode_id": d.name, "error": str(e)[:80]})
            continue

        # valid 三态：True / False / None(未标注)。null 不等于 false。
        valid = (meta or {}).get("valid")
        if valid is True:
            valid_n += 1
        elif valid is False:
            invalid_n += 1
        else:
            unlabeled_n += 1
        frames_total += n_frames

        episodes.append({
            "episode_id": d.name,
            "num_frames": n_frames,
            "schema_version": schema,
            "fps_measured": None if fps != fps else round(fps, 3),
            "has_meta_json": meta is not None,
            "valid": valid,
            "success": (meta or {}).get("success"),
            "reject_reason": (meta or {}).get("reject_reason"),
        })

    # 读既有排除名单（Converter 也读这个文件）
    reject_list = paths.reject_list(args.task)
    excluded = []
    if reject_list.is_file():
        excluded = [ln.strip() for ln in reject_list.read_text().splitlines() if ln.strip()]

    manifest = {
        "dataset_version": args.task,
        "schema_version": episode_io.SCHEMA_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "generator": "tools/gen_dataset_manifest.py",
        "raw_root": str(paths.raw),
        "raw_root_real": str(paths.raw.resolve()) if paths.raw.exists() else None,
        "task": args.task,
        "episode_count": len(eps),
        "valid_episode_count": valid_n,
        "invalid_episode_count": invalid_n,
        "unlabeled_episode_count": unlabeled_n,
        "frames_total": frames_total,
        "excluded_by_reject_list": excluded,
        # 下列产物路径由 Converter / 训练阶段填写，此处无法得知则为 null
        "lerobot_repo_id": None,
        "norm_stats_path": None,
        "training_experiment": None,
        "checkpoint_path": None,
        "notes": args.notes or None,
        "episodes": episodes,
    }

    out = paths.manifest(args.task)
    if args.dry_run:
        print(json.dumps(manifest, indent=2, ensure_ascii=False)[:1500])
        print(f"\n[dry-run] 未写入 {out}")
        return
    task_dir.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[manifest] -> {out}")
    print(f"  episode={len(eps)}  帧={frames_total}  "
          f"valid={valid_n} invalid={invalid_n} 未标注={unlabeled_n}")


if __name__ == "__main__":
    main()

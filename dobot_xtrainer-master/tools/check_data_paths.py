"""数据路径体检。只读，不创建、不修改、不删除任何东西。

检查项：
    1. 四类数据根是否存在、来源（env / config / default）
    2. raw 是否为软链接、指向哪里、实体是否存在
    3. raw 与已知的其他 datasets 目录是否为**物理重复副本**（本任务的核心约束）
    4. 各 dataset version（任务目录）的 episode 数、V1/V2 schema 分布
    5. manifest / statistics 是否存在
    6. Git 是否会误提交大型数据

用法：
    python tools/check_data_paths.py
    python tools/check_data_paths.py --json      # 机器可读输出
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.data_paths import paths  # noqa: E402

OK, BAD, WARN = "✓", "✗", "!"

# 历史上出现过 datasets 的位置，用于检测是否有第二份物理 Raw
KNOWN_DATASET_DIRS = [
    pathlib.Path("/home/iml/dobot_xtrainer/datasets"),
    pathlib.Path("/home/iml/RockyEVO/dobot_xtrainer/datasets"),
    pathlib.Path("/home/iml/dobot_xtrainer/dobot_xtrainer-master/datasets"),
    pathlib.Path("/home/iml/RockyEVO/dobot_xtrainer/dobot_xtrainer-master/datasets"),
]


def count_episodes(task_dir: pathlib.Path):
    cdir = task_dir / "collect_data"
    if not cdir.is_dir():
        return []
    return sorted(d.name for d in cdir.iterdir() if d.is_dir())


def probe_episode(ep_dir: pathlib.Path):
    """返回 (schema, num_frames, has_meta)。读不动就返回 (None, 0, False)。"""
    obs = ep_dir / "observation"
    if not obs.is_dir():
        return None, 0, (ep_dir / "meta.json").is_file()
    pkls = list(obs.glob("*.pkl"))
    schema = None
    if pkls:
        try:
            import pickle
            with open(sorted(pkls, key=lambda p: int(p.stem))[0], "rb") as f:
                schema = int(pickle.load(f).get("schema_version", 1))
        except Exception:
            schema = None
    return schema, len(pkls), (ep_dir / "meta.json").is_file()


def dir_signature(d: pathlib.Path):
    """用 (文件数, 总字节) 作为轻量指纹，判断是否物理重复。不读文件内容。"""
    if not d.is_dir():
        return None
    n = total = 0
    for root, _, files in os.walk(d):
        for fn in files:
            try:
                total += os.path.getsize(os.path.join(root, fn))
                n += 1
            except OSError:
                pass
    return n, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    args = ap.parse_args()

    report = {"roots": {}, "raw": {}, "duplicates": [], "datasets": [], "git": {}, "problems": []}
    out = []

    # ── 1. 四类数据根 ──
    out.append("── 数据根目录 ──")
    for name in ("raw", "processed", "lerobot", "rejected", "test_configs"):
        p = getattr(paths, name)
        exists = p.exists()
        report["roots"][name] = {
            "path": str(p), "exists": exists, "source": paths.source_of(p),
            "is_symlink": p.is_symlink(),
            "real": str(p.resolve()) if exists else None,
        }
        mark = OK if exists else BAD
        extra = f" -> {os.readlink(p)}" if p.is_symlink() else ""
        out.append(f"  {mark} {name:<13} {p}  [{paths.source_of(p)}]{extra}")
        if not exists:
            report["problems"].append(f"{name} 不存在: {p}")

    # ── 2. raw 详情 ──
    out.append("\n── 唯一 Raw Dataset ──")
    raw_real = paths.raw.resolve() if paths.raw.exists() else None
    report["raw"] = {
        "entry": str(paths.raw),
        "is_symlink": paths.raw.is_symlink(),
        "real": str(raw_real) if raw_real else None,
    }
    out.append(f"  入口: {paths.raw}")
    out.append(f"  实体: {raw_real}")
    out.append(f"  软链接: {'是' if paths.raw.is_symlink() else '否'}")

    # ── 3. 物理重复检测 ──
    out.append("\n── 物理重复副本检测 ──")
    raw_sig = dir_signature(raw_real) if raw_real else None
    for d in KNOWN_DATASET_DIRS:
        if not d.is_dir():
            out.append(f"  -  {d}  (不存在)")
            continue
        same_entity = raw_real is not None and d.resolve() == raw_real
        sig = dir_signature(d)
        if same_entity:
            out.append(f"  {OK} {d}  = 唯一 Raw 实体本身")
            continue
        n, total = sig
        if n == 0:
            out.append(f"  {OK} {d}  空目录，无重复风险")
        elif raw_sig and sig == raw_sig:
            out.append(f"  {BAD} {d}  疑似 Raw 的物理副本 ({n} 文件 / {total/1e6:.1f} MB)")
            report["duplicates"].append(str(d))
            report["problems"].append(f"疑似重复 Raw 副本: {d}")
        else:
            out.append(f"  {WARN} {d}  含 {n} 文件 / {total/1e6:.1f} MB（与 Raw 指纹不同）")

    # ── 4. dataset version ──
    out.append("\n── Dataset Versions（raw 下的任务目录）──")
    if raw_real and raw_real.is_dir():
        tasks = sorted(d for d in raw_real.iterdir() if d.is_dir())
        if not tasks:
            out.append("  (无)")
        for t in tasks:
            eps = count_episodes(t)
            v1 = v2 = unknown = with_meta = frames = 0
            for e in eps:
                s, n, hm = probe_episode(t / "collect_data" / e)
                frames += n
                with_meta += 1 if hm else 0
                if s == 2:
                    v2 += 1
                elif s == 1:
                    v1 += 1
                else:
                    unknown += 1
            info = {
                "task": t.name, "episode_count": len(eps), "frames": frames,
                "schema_v1": v1, "schema_v2": v2, "schema_unknown": unknown,
                "with_meta_json": with_meta,
                "has_manifest": (t / "dataset_manifest.json").is_file(),
                "has_statistics": (t / "dataset_statistics.json").is_file(),
            }
            report["datasets"].append(info)
            out.append(f"  {t.name}: {len(eps)} episode / {frames} 帧  "
                       f"V1={v1} V2={v2} 未知={unknown}  meta.json={with_meta}")
            out.append(f"      manifest={'有' if info['has_manifest'] else '无'}  "
                       f"statistics={'有' if info['has_statistics'] else '无'}")
            if t.name in ("final", "final2", "new", "latest", "最终版", "最新版"):
                report["problems"].append(f"禁用的版本命名: {t.name}")

    # ── 5. Git 风险 ──
    out.append("\n── Git 大文件风险 ──")
    for repo in (pathlib.Path(__file__).resolve().parent.parent.parent,):
        if not (repo / ".git").is_dir():
            continue
        try:
            files = subprocess.run(["git", "-C", str(repo), "ls-files"],
                                   capture_output=True, text=True, timeout=60).stdout.split("\n")
        except Exception as e:
            out.append(f"  {WARN} 无法查询 {repo}: {e}")
            continue
        big = []
        for rel in files:
            if not rel:
                continue
            fp = repo / rel
            try:
                sz = fp.stat().st_size
            except OSError:
                continue
            if sz > 10 * 1024 * 1024:
                big.append((rel, sz))
        report["git"][str(repo)] = [{"file": f, "mb": round(s / 1e6, 1)} for f, s in big]
        if big:
            out.append(f"  {BAD} {repo}: {len(big)} 个 >10MB 文件被 Git 跟踪")
            for f, s in sorted(big, key=lambda x: -x[1])[:5]:
                out.append(f"       {s/1e6:8.1f} MB  {f}")
            report["problems"].append(f"{repo} 有 {len(big)} 个 >10MB 文件被 Git 跟踪")
        else:
            out.append(f"  {OK} {repo}: 无 >10MB 文件被跟踪")

    out.append("\n── 结论 ──")
    if report["problems"]:
        out.append(f"  发现 {len(report['problems'])} 个问题:")
        for p in report["problems"]:
            out.append(f"    {BAD} {p}")
    else:
        out.append(f"  {OK} 未发现问题")

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print("\n".join(out))
    return 1 if report["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())

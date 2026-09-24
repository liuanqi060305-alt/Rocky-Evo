#!/usr/bin/env python3
"""批量数据质量概览（分布视角）。

1. Frame length distribution (should be 300-800)
2. Robot arm state frames aligned with images
3. Image frame drops (missing frame numbers)

与 manage_episodes.py 的分工：
    本脚本      —— 跨 episode 的**分布概览**，回答「整批数据长什么样」
    manage_episodes audit —— 单条 **PASS/FAIL** 判定并写回 meta.json 的 valid

两者的判定标准共用 scripts/episode_audit.py，不存在两套阈值。
末尾附带一段基于该引擎的 PASS/FAIL 汇总，便于一眼看出哪些条目会被排除训练。
"""

import os
import glob
import pickle
import numpy as np
from collections import defaultdict
import sys

# Data root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.data_paths import paths  # noqa: E402

# 任务名可用环境变量 XTRAINER_TASK_NAME 覆盖；路径一律经统一配置层解析，
# 不再写死 /home/iml/... （换机、换仓库副本就失效）
TASK_NAME = os.environ.get("XTRAINER_TASK_NAME", "plug_and_unplug_task")
DATA_DIR = str(paths.collect_dir(TASK_NAME))

# Get all episode dirs sorted by timestamp (newest first)
all_dirs = sorted(
    glob.glob(os.path.join(DATA_DIR, "2026*")),
    reverse=True
)

# Take the 101 most recent
episode_dirs = all_dirs[:101]
print(f"Checking the {len(episode_dirs)} most recent episodes\n")
print(f"Date range: {os.path.basename(episode_dirs[-1])} ~ {os.path.basename(episode_dirs[0])}\n")

# ============================================================
# 1. FRAME LENGTH DISTRIBUTION
# ============================================================
print("=" * 80)
print("1. FRAME LENGTH DISTRIBUTION (expected: 300-800)")
print("=" * 80)

frame_lengths = {}  # ep_name -> {modality: count}
abnormal_length = []  # episodes outside 300-800 range

for ep_dir in episode_dirs:
    ep_name = os.path.basename(ep_dir)
    counts = {}
    for modality in ['leftImg', 'rightImg', 'topImg', 'observation']:
        mod_dir = os.path.join(ep_dir, modality)
        if os.path.exists(mod_dir):
            files = glob.glob(os.path.join(mod_dir, '*'))
            counts[modality] = len(files)
        else:
            counts[modality] = 0
    frame_lengths[ep_name] = counts

    # Check if any modality is outside 300-800
    for mod, count in counts.items():
        if count < 300 or count > 800:
            abnormal_length.append((ep_name, mod, count))

# Show distribution
all_counts = {}
for mod in ['observation', 'leftImg', 'rightImg', 'topImg']:
    counts = [frame_lengths[ep][mod] for ep in frame_lengths]
    all_counts[mod] = counts
    print(f"\n{mod}:")
    print(f"  Min:    {min(counts):6d}")
    print(f"  Max:    {max(counts):6d}")
    print(f"  Mean:   {np.mean(counts):8.1f}")
    print(f"  Median: {np.median(counts):8.1f}")
    print(f"  Std:    {np.std(counts):8.1f}")

# Print distribution buckets
print("\n--- Distribution buckets (by observation count) ---")
obs_counts = all_counts['observation']
buckets = [(0, 100), (100, 200), (200, 300), (300, 400), (400, 500),
           (500, 600), (600, 700), (700, 800), (800, 900), (900, 1000),
           (1000, 1500), (1500, 9999)]
for lo, hi in buckets:
    in_range = [c for c in obs_counts if lo <= c < hi]
    if in_range:
        print(f"  [{lo:4d}, {hi:4d}): {len(in_range):3d} episodes")

print(f"\n⚠️  Episodes outside 300-800 range: {len(abnormal_length)}")
for ep, mod, count in abnormal_length:
    print(f"  {ep}: {mod}={count}")

# ============================================================
# 2. STATE-IMAGE ALIGNMENT
# ============================================================
print("\n" + "=" * 80)
print("2. STATE-IMAGE FRAME ALIGNMENT")
print("=" * 80)

misaligned_episodes = []

for ep_dir in episode_dirs:
    ep_name = os.path.basename(ep_dir)

    # Get frame numbers for each modality
    obs_dir = os.path.join(ep_dir, 'observation')
    left_dir = os.path.join(ep_dir, 'leftImg')
    right_dir = os.path.join(ep_dir, 'rightImg')
    top_dir = os.path.join(ep_dir, 'topImg')

    obs_frames = set()
    left_frames = set()
    right_frames = set()
    top_frames = set()

    if os.path.exists(obs_dir):
        for f in glob.glob(os.path.join(obs_dir, '*.pkl')):
            try:
                obs_frames.add(int(os.path.basename(f).replace('.pkl', '')))
            except ValueError:
                pass

    for mod_dir, frame_set in [(left_dir, left_frames), (right_dir, right_frames), (top_dir, top_frames)]:
        if os.path.exists(mod_dir):
            for f in glob.glob(os.path.join(mod_dir, '*.jpg')):
                try:
                    frame_set.add(int(os.path.basename(f).replace('.jpg', '')))
                except ValueError:
                    pass

    issues = []

    # Check obs vs each image modality
    for mod_name, img_frames in [('leftImg', left_frames), ('rightImg', right_frames), ('topImg', top_frames)]:
        # Images without obs
        img_only = img_frames - obs_frames
        if img_only:
            issues.append(f"{mod_name} has {len(img_only)} frames without obs: {sorted(img_only)[:10]}{'...' if len(img_only) > 10 else ''}")

        # Obs without images
        obs_only = obs_frames - img_frames
        if obs_only:
            issues.append(f"obs has {len(obs_only)} frames without {mod_name}: {sorted(obs_only)[:10]}{'...' if len(obs_only) > 10 else ''}")

    # Check if all three image modalities have the same frame set
    if left_frames != right_frames:
        diff = (left_frames - right_frames) | (right_frames - left_frames)
        issues.append(f"leftImg ({len(left_frames)}) vs rightImg ({len(right_frames)}) mismatch: {len(diff)} frames differ")
    if left_frames != top_frames:
        diff = (left_frames - top_frames) | (top_frames - left_frames)
        issues.append(f"leftImg ({len(left_frames)}) vs topImg ({len(top_frames)}) mismatch: {len(diff)} frames differ")

    if issues:
        misaligned_episodes.append((ep_name, issues))

print(f"\n⚠️  Episodes with alignment issues: {len(misaligned_episodes)}")
for ep_name, issues in misaligned_episodes:
    print(f"\n  {ep_name}:")
    for iss in issues:
        print(f"    - {iss}")

# ============================================================
# 3. FRAME DROPS (missing frame numbers)
# ============================================================
print("\n" + "=" * 80)
print("3. IMAGE FRAME DROPS (checking for gaps in frame numbering)")
print("=" * 80)

frame_drop_episodes = []

for ep_dir in episode_dirs:
    ep_name = os.path.basename(ep_dir)

    issues = {}

    for modality in ['leftImg', 'rightImg', 'topImg']:
        mod_dir = os.path.join(ep_dir, modality)
        if not os.path.exists(mod_dir):
            continue

        frames = []
        for f in glob.glob(os.path.join(mod_dir, '*.jpg')):
            try:
                frames.append(int(os.path.basename(f).replace('.jpg', '')))
            except ValueError:
                pass

        if not frames:
            continue

        frames = sorted(frames)

        # Check if frames start from 0
        if frames[0] != 0:
            if modality not in issues:
                issues[modality] = []
            issues[modality].append(f"Does not start from 0, first frame is {frames[0]}")

        # Check for gaps in frame numbering
        expected = list(range(frames[0], frames[-1] + 1))
        missing = sorted(set(expected) - set(frames))

        if missing:
            if modality not in issues:
                issues[modality] = []
            total_missing = len(missing)
            # Calculate gap statistics
            gap_ranges = []
            start = missing[0]
            end = missing[0]
            for i in range(1, len(missing)):
                if missing[i] == end + 1:
                    end = missing[i]
                else:
                    gap_ranges.append((start, end, end - start + 1))
                    start = missing[i]
                    end = missing[i]
            gap_ranges.append((start, end, end - start + 1))

            issues[modality].append(f"Total {total_missing} missing frames out of {len(expected)} expected ({100*total_missing/len(expected):.1f}%)")
            issues[modality].append(f"  Frame range: {frames[0]} ~ {frames[-1]}, Seq len: {len(frames)}")
            issues[modality].append(f"  Gap ranges (start-end: count): {[(s, e, c) for s, e, c in gap_ranges[:5]]}{'...' if len(gap_ranges) > 5 else ''}")

    if issues:
        frame_drop_episodes.append((ep_name, issues))

print(f"\n⚠️  Episodes with frame drops: {len(frame_drop_episodes)}")
for ep_name, issues in frame_drop_episodes:
    print(f"\n  {ep_name}:")
    for mod, iss_list in issues.items():
        print(f"    {mod}:")
        for iss in iss_list:
            print(f"      - {iss}")

# ============================================================
# SUMMARY - Episodes to DELETE
# ============================================================
print("\n" + "=" * 80)
print("SUMMARY - EPISODES RECOMMENDED FOR DELETION")
print("=" * 80)

to_delete = set()

# 1. Abnormal frame length
for ep, mod, count in abnormal_length:
    to_delete.add(ep)

# 2. Misaligned
for ep_name, _ in misaligned_episodes:
    to_delete.add(ep_name)

# 3. Frame drops
for ep_name, _ in frame_drop_episodes:
    to_delete.add(ep_name)

# Count good episodes
all_ep_names = set(os.path.basename(d) for d in episode_dirs)
good_episodes = all_ep_names - to_delete

print(f"\nTotal episodes checked: {len(episode_dirs)}")
print(f"Good episodes:           {len(good_episodes)}")
print(f"Episodes to delete:      {len(to_delete)}")

# Break down by reason
reason_abnormal = set(ep for ep, _, _ in abnormal_length)
reason_misalign = set(ep for ep, _ in misaligned_episodes)
reason_framedrop = set(ep for ep, _ in frame_drop_episodes)

print(f"\nBreakdown by issue:")
print(f"  Abnormal frame length (<300 or >800): {len(reason_abnormal)}")
print(f"  State-image misalignment:             {len(reason_misalign)}")
print(f"  Image frame drops:                    {len(reason_framedrop)}")

# Print episodes to delete with reasons
print(f"\n📋 Episodes to DELETE ({len(to_delete)}):")
for ep in sorted(to_delete):
    reasons = []
    if ep in reason_abnormal:
        for e, mod, count in abnormal_length:
            if e == ep:
                reasons.append(f"bad length: {mod}={count}")
                break
    if ep in reason_misalign:
        reasons.append("misalignment")
    if ep in reason_framedrop:
        reasons.append("frame drops")
    print(f"  {ep} ({', '.join(reasons)})")

# Save list to file for deletion
delete_list_path = str(paths.reject_list(TASK_NAME))
with open(delete_list_path, 'w') as f:
    for ep in sorted(to_delete):
        f.write(f"{ep}\n")
print(f"\n💾 Delete list saved to: {delete_list_path}")

# 坏数据记录（不动 Raw 实体）。按数据流约定，Raw 尽量不可变：
# 优先用这份 manifest + episode 自己的 meta.json valid=false 来排除数据，
# 而不是 rm 掉唯一 Raw。下面生成的 delete_bad_episodes.sh 是既有行为，保留但不推荐。
import json as _json
from datetime import datetime as _dt
_rej_dir = paths.rejected / TASK_NAME
_rej_dir.mkdir(parents=True, exist_ok=True)
_rej_manifest = _rej_dir / "rejected_manifest.json"
# 复用上面打印时的同一套判定，保持 manifest 与终端输出一致
_reasons_map = {}
for _ep in sorted(to_delete):
    _r = []
    if _ep in reason_abnormal:
        for _e, _mod, _count in abnormal_length:
            if _e == _ep:
                _r.append(f"bad length: {_mod}={_count}")
                break
    if _ep in reason_misalign:
        _r.append("misalignment")
    if _ep in reason_framedrop:
        _r.append("frame drops")
    _reasons_map[_ep] = _r
with open(_rej_manifest, "w", encoding="utf-8") as _f:
    _json.dump({
        "generated_at": _dt.now().isoformat(timespec="seconds"),
        "generator": "scripts/check_episodes.py",
        "task_name": TASK_NAME,
        "raw_root": str(paths.raw),
        "raw_root_real": str(paths.raw.resolve()) if paths.raw.exists() else None,
        "policy": "Raw 保持不变；此处仅登记索引与原因。排除训练数据请以本文件或 "
                  "episode meta.json 的 valid=false 为准。",
        "rejected_count": len(to_delete),
        "rejected": [{"episode_id": k, "reject_reason": v} for k, v in _reasons_map.items()],
    }, _f, indent=2, ensure_ascii=False)
print(f"💾 Rejected manifest saved to: {_rej_manifest}")

# Write deletion script
delete_script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "delete_bad_episodes.sh")
with open(delete_script, 'w') as f:
    f.write("#!/bin/bash\n")
    f.write("# Delete bad episodes - generated by check_episodes.py\n")
    f.write(f"cd {DATA_DIR}\n")
    f.write("echo 'The following episodes will be deleted:'\n")
    for ep in sorted(to_delete):
        f.write(f"echo '  {ep}'\n")
    f.write("echo ''\n")
    f.write("read -p 'Confirm deletion? (y/N): ' confirm\n")
    f.write('if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then\n')
    for ep in sorted(to_delete):
        f.write(f"    rm -rf {ep}\n")
    f.write('    echo "Deleted successfully."\n')
    f.write('else\n')
    f.write('    echo "Cancelled."\n')
    f.write('fi\n')
os.chmod(delete_script, 0o755)
print(f"💾 Delete script saved to: {delete_script}")

# ─────────────────────────────────────────────────────────────
# 与 manage_episodes audit 完全一致的 PASS/FAIL 汇总。
# 复用同一引擎，避免本脚本和单条审查给出互相矛盾的结论。
# 只读、不写 meta.json —— 写回 valid 是 manage_episodes 的职责。
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("4. PASS / FAIL 汇总（judgement 与 manage_episodes audit 一致）")
print("=" * 80)
from scripts.episode_audit import audit_episode  # noqa: E402

_pass, _fail = [], []
for ep_dir in episode_dirs:
    _r = audit_episode(ep_dir)
    (_pass if _r.passed else _fail).append(_r)

print(f"\nPASS {len(_pass)} 条 / FAIL {len(_fail)} 条")
if _fail:
    print("\nFAIL 明细：")
    for _r in _fail:
        print("  " + _r.render().replace("\n", "\n  "))
print("\n提示：写回 valid 标记请用")
print("  python scripts/manage_episodes.py audit-pending")

print("\n✅ Check complete!")

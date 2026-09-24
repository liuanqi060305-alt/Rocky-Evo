"""人工示范数据管理入口（V2 策略：留存即成功）。

流程（常规只需两步）：
    采一条 -> 当场判断
        失败 -> reject-human：写 rejected log 后立刻删除大文件
        成功 -> 什么都不用做，留在盘上即可
    采完 -> build-train-manifest：自动审查所有未审查条目并出清单

**留存即成功（IMPLICIT_SUCCESS）**：人工失败的 Episode 在采集后立即 reject-human
删除，因此磁盘上留存的数据按定义就是成功示范，不再要求逐条 mark-success。
audit 会把 success 显式写成 true 并标注 success_source="implicit_retained"，
使 meta.json 自解释，不依赖"读代码才知道 null 是什么意思"。

两个概念仍然区分（语义没变，只是 success 不再需要人工输入）：
    success  人工示范任务是否成功。留存即 true；显式 false 才排除
    valid    数据本身是否满足训练质量（audit 判定）
             null=未审查，与 false=已判定坏数据 语义不同

命令：
    python scripts/manage_episodes.py list
    python scripts/manage_episodes.py reject-human <episode_id> --reason operator_failure
    python scripts/manage_episodes.py build-train-manifest     # 自动审查 + 出清单
    python scripts/manage_episodes.py audit <episode_id>       # 单条审查
    python scripts/manage_episodes.py audit-pending            # 只审查不出清单
    python scripts/manage_episodes.py mark-failed <episode_id> # 保留但排除训练
    python scripts/manage_episodes.py mark-success <episode_id> # 显式覆盖，通常不需要
    python scripts/manage_episodes.py delete-invalid <episode_id>   # 需人工确认

审查失败的数据**不会**被自动删除：审查规则本身可能误判，需先人工确认。
delete-invalid 是显式命令，且要求 valid 已为 false。
"""
import argparse
import csv
import datetime
import json
import os
import pathlib
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts import episode_io  # noqa: E402
from scripts.data_paths import paths  # noqa: E402
from scripts.episode_audit import audit_episode  # noqa: E402

REJECTED_LOG = "rejected_log.jsonl"
TRAIN_MANIFEST = "train_manifest.json"
QUALITY_REJECTED = "quality_rejected_manifest.json"

# 留存即成功：失败数据在采集后立即 reject-human 删除，所以盘上的就是成功示范。
# 只有显式写入 success=false（mark-failed）才排除。null 视为 true。
IMPLICIT_SUCCESS = True


def is_success(meta: dict) -> bool:
    """success 判定。留存即成功 —— 只有显式 false 才算失败。

    刻意不写成 `meta.get("success") is True`：那会把 V1 老数据和刚采完还没
    audit 的新数据（success=null）当成失败，而这批数据恰恰是成功示范。
    """
    return meta.get("success") is not False


def _now_iso():
    return datetime.datetime.now().isoformat(timespec="seconds")


def task_name(args) -> str:
    return args.task or os.environ.get("XTRAINER_TASK_NAME", "plug_and_unplug_task")


# ────────────────────────── 安全删除 ──────────────────────────

def assert_safe_to_delete(ep_dir: pathlib.Path, task: str) -> pathlib.Path:
    """删除前的路径闸门。任何一条不满足就抛异常，绝不删。

    防的是：删到 /、删到 datasets 根、删到任务目录、通配符残留、
    符号链接逃逸到 Raw 之外。
    """
    ep = ep_dir.resolve()
    collect = paths.collect_dir(task).resolve()
    raw = paths.raw.resolve()

    if not ep.exists():
        raise SystemExit(f"[拒绝] 目录不存在: {ep}")
    if not ep.is_dir():
        raise SystemExit(f"[拒绝] 不是目录: {ep}")

    # 必须是 <raw>/<task>/collect_data/ 的**直接子目录**
    if ep.parent != collect:
        raise SystemExit(f"[拒绝] 不是 collect_data 的直接子目录\n"
                         f"       目标: {ep}\n       期望父目录: {collect}")
    # 显式排除危险目标
    for danger in (pathlib.Path("/"), raw, collect, collect.parent, ep.parent):
        if ep == danger:
            raise SystemExit(f"[拒绝] 目标是受保护路径: {ep}")
    if len(ep.parts) < 5:
        raise SystemExit(f"[拒绝] 路径层级过浅，疑似危险: {ep}")
    # 通配符/相对路径残留
    name = ep.name
    if any(c in name for c in "*?[]") or name in (".", "..", ""):
        raise SystemExit(f"[拒绝] episode 名含通配符或非法字符: {name!r}")
    # episode 目录名应是采集程序生成的时间戳
    if not name.isdigit():
        raise SystemExit(f"[拒绝] episode 名不是数字时间戳: {name!r}\n"
                         f"       这道检查防止误传任务名/目录名")
    # 内容特征确认：必须长得像一条 episode
    if not (ep / "observation").is_dir() and not (ep / "topImg").is_dir():
        raise SystemExit(f"[拒绝] 目标不含 observation/ 或 topImg/，不像 episode: {ep}")
    return ep


def append_rejected_log(task: str, record: dict) -> pathlib.Path:
    """追加一条删除记录。写失败则抛异常 —— 调用方必须先写日志再删。"""
    log_dir = paths.rejected / task
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / REJECTED_LOG
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())          # 确保落盘，之后才允许删除
    # 回读确认真的写进去了
    with open(log_path, "r", encoding="utf-8") as f:
        if record["episode_id"] not in f.read():
            raise SystemExit(f"[中止] rejected log 回读校验失败，未执行删除: {log_path}")
    # 同时维护一份 CSV，便于人工用表格查看
    csv_path = log_dir / "rejected_log.csv"
    new = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["episode_id", "time", "reason", "source", "success",
                        "num_frames", "size_mb", "note"])
        w.writerow([record["episode_id"], record["time"], record["reason"],
                    record["source"], record.get("success"),
                    record.get("num_frames"), record.get("size_mb"),
                    record.get("note", "")])
    return log_path


def dir_size_mb(p: pathlib.Path) -> float:
    total = 0
    for root, _, files in os.walk(p):
        for fn in files:
            try:
                total += os.path.getsize(os.path.join(root, fn))
            except OSError:
                pass
    return round(total / 1e6, 2)


def _ensure_meta(ep_dir: pathlib.Path, task: str) -> dict:
    """取 meta.json；V1 数据没有则就地补一份最小骨架（不碰采集数据本身）。"""
    meta = episode_io.load_meta(str(ep_dir))
    if meta is not None:
        return meta
    meta = episode_io.build_meta(
        episode_id=ep_dir.name, task_name=task,
        episode_start_time_ns=None, episode_start_monotonic_ns=None,
        data_type="teleop",
        note="meta.json 由 manage_episodes.py 补建（该 episode 采集时为 V1 格式）",
    )
    episode_io.write_meta(str(ep_dir), meta)
    return meta


def _position_view_name(annotation: dict) -> str:
    """Build a readable, path-safe directory name for a position label."""
    name = annotation.get("position_id") or "unlabeled_position"
    description = annotation.get("description")
    if description:
        name = f"{name}__{description}"
    return str(name).replace("/", "_").replace("\\", "_").strip()


def _update_position_view(task: str, episode_id: str, annotation: dict) -> pathlib.Path:
    """Create a human-readable symlink view without renaming raw episode directories."""
    view_root = paths.raw / task / "position_views"
    view_root.mkdir(parents=True, exist_ok=True)

    # When relabeling, remove only old symlinks for this episode. Raw data is never touched.
    for old_link in view_root.glob(f"*/{episode_id}"):
        if old_link.is_symlink():
            old_link.unlink()

    view_dir = view_root / _position_view_name(annotation)
    view_dir.mkdir(parents=True, exist_ok=True)
    link = view_dir / episode_id
    if not os.path.lexists(link):
        link.symlink_to(pathlib.Path("..") / ".." / "collect_data" / episode_id)
    return link


# ────────────────────────── 子命令 ──────────────────────────

def cmd_list(args):
    task = task_name(args)
    collect = paths.collect_dir(task)
    if not collect.is_dir():
        raise SystemExit(f"[ERROR] 找不到 {collect}")
    eps = sorted(d for d in collect.iterdir() if d.is_dir())
    print(f"任务 {task}  共 {len(eps)} 条  ({collect})  策略: 留存即成功")
    # 表头用 ASCII，CJK 是双宽字符，混进定宽列会把对齐搞乱
    print(f"{'episode_id':<18}{'success':<12}{'valid':<10}{'frames':<8}{'schema':<8}{'size':<12}position")
    n_train = 0
    for d in eps:
        m = episode_io.load_meta(str(d)) or {}
        try:
            n = len(episode_io.frame_indices(str(d)))
        except Exception:
            n = 0
        sv = "?"
        try:
            sv = str(episode_io.load_episode(str(d))["source_schema"])
        except Exception:
            pass
        # success 列显示生效值；括号标出是隐式推定还是 meta 里写死的
        raw_s = m.get("success")
        s_txt = "true" if raw_s is True else ("false" if raw_s is False else "true*")
        if is_success(m) and m.get("valid") is not False:
            n_train += 1
        gen = m.get("generalization") or {}
        position = gen.get("description") or gen.get("position_id") or "-"
        print(f"{d.name:<18}{s_txt:<12}{str(m.get('valid')):<10}"
              f"{n:<8}{sv:<8}{str(dir_size_mb(d)) + ' MB':<12}{position}")
    print(f"\n当前可进训练: {n_train} / {len(eps)}（success≠false 且 valid≠false）")
    print("true* = 隐式成功（留存即成功，meta 里 success 仍为 null，audit 后落定）")


def cmd_mark_success(args):
    """显式标记成功。留存即成功策略下通常不需要 —— 仅用于撤销 mark-failed。"""
    task = task_name(args)
    ep_dir = paths.episode_dir(task, args.episode_id)
    if not ep_dir.is_dir():
        raise SystemExit(f"[ERROR] 目录不存在: {ep_dir}")
    meta = _ensure_meta(ep_dir, task)
    meta["success"] = True
    meta["success_source"] = "manual"
    meta["success_marked_at"] = _now_iso()
    if args.note:
        meta["note"] = args.note
    # valid 刻意不动：成功只代表任务完成，数据质量要等 audit
    meta.setdefault("valid", None)
    episode_io.write_meta(str(ep_dir), meta)
    print(f"[mark-success] {args.episode_id}: success=true (manual), "
          f"valid={meta['valid']}")


def cmd_mark_failed(args):
    """标记为人工失败但**保留**数据，排除训练。

    用于「事后才发现这条不该用，但暂时不想删」。要真删就用 reject-human。
    """
    task = task_name(args)
    ep_dir = paths.episode_dir(task, args.episode_id)
    if not ep_dir.is_dir():
        raise SystemExit(f"[ERROR] 目录不存在: {ep_dir}")
    meta = _ensure_meta(ep_dir, task)
    meta["success"] = False
    meta["success_source"] = "manual"
    meta["success_marked_at"] = _now_iso()
    meta["failure_reason"] = args.reason
    if args.note:
        meta["note"] = args.note
    episode_io.write_meta(str(ep_dir), meta)
    print(f"[mark-failed] {args.episode_id}: success=false，已排除训练但数据保留")
    print(f"  要彻底删除请用: reject-human {args.episode_id} --reason {args.reason}")


def cmd_label_position(args):
    """给已采 episode 补写位置泛化标注，不改变 success / valid。"""
    task = task_name(args)
    annotation = {
        "factor": "object_position",
        "object": args.position_object,
        "position_id": args.position_id,
        "description": args.description,
        "reference_frame": args.position_frame,
        "offset_mm": {"x": args.dx_mm, "y": args.dy_mm, "z": args.dz_mm},
        "yaw_deg": args.yaw_deg,
        "split": args.split,
    }
    for episode_id in args.episode_ids:
        ep_dir = paths.episode_dir(task, episode_id)
        if not ep_dir.is_dir():
            raise SystemExit(f"[ERROR] 目录不存在: {ep_dir}")
        meta = _ensure_meta(ep_dir, task)
        meta["generalization"] = annotation
        meta["generalization_marked_at"] = _now_iso()
        episode_io.write_meta(str(ep_dir), meta)
        link = _update_position_view(task, episode_id, annotation)
        print(f"[label-position] {episode_id}: {args.position_id} "
              f"split={args.split} offset_mm={annotation['offset_mm']} yaw={args.yaw_deg}")
        print(f"  view -> {link}")


def cmd_reject_human(args):
    """人工判定失败：先写日志，再删除整个 Episode 目录。"""
    task = task_name(args)
    ep_dir = paths.episode_dir(task, args.episode_id)
    ep = assert_safe_to_delete(ep_dir, task)

    size = dir_size_mb(ep)
    try:
        n_frames = len(episode_io.frame_indices(str(ep)))
    except Exception:
        n_frames = None

    print("即将删除人工失败 demonstration：")
    print(f"  完整路径 : {ep}")
    print(f"  实体位置 : {ep.resolve()}")
    print(f"  帧数     : {n_frames}")
    print(f"  占用     : {size} MB")
    print(f"  原因     : {args.reason}")
    if not args.yes:
        ans = input("确认删除？只有输入 DELETE 才执行: ").strip()
        if ans != "DELETE":
            raise SystemExit("[取消] 未执行任何删除")

    # 顺序至关重要：日志先落盘，失败则中止，绝不先删后记
    record = {
        "episode_id": ep.name,
        "time": _now_iso(),
        "reason": args.reason,
        "source": "human_demo",
        "success": False,
        "task": task,
        "num_frames": n_frames,
        "size_mb": size,
        "note": args.note or "",
        "deleted_path": str(ep),
    }
    log_path = append_rejected_log(task, record)
    print(f"[log] 已记录 -> {log_path}")

    shutil.rmtree(ep)
    if ep.exists():
        raise SystemExit(f"[ERROR] 删除后目录仍存在: {ep}")
    print(f"[deleted] {ep.name} 已删除，释放 {size} MB（记录已保留）")


def cmd_audit(args):
    task = task_name(args)
    ep_dir = paths.episode_dir(task, args.episode_id)
    r = audit_episode(str(ep_dir))
    print(r.render())
    if not args.no_write:
        _write_valid(ep_dir, task, r)
    return 0 if r.passed else 1


def _write_valid(ep_dir: pathlib.Path, task: str, r):
    """把审查结论写回 meta.json。

    顺带把隐式 success 落成显式 true —— 留存即成功策略下 success=null 的含义
    是「留存所以成功」，但 null 本身不自解释。写成 true + success_source 后，
    下游读 meta.json 不必再知道这条策略。已有显式值（人工 mark 过）不覆盖。
    """
    meta = _ensure_meta(ep_dir, task)
    if IMPLICIT_SUCCESS and meta.get("success") is None:
        meta["success"] = True
        meta["success_source"] = "implicit_retained"
    meta["valid"] = bool(r.passed)
    meta["quality_issues"] = r.issue_codes
    meta["quality_warnings"] = r.warnings
    meta["audited_at"] = _now_iso()
    for k in ("num_frames", "duration_s", "fps_measured"):
        if r.info.get(k) is not None:
            meta[k] = r.info[k]
    episode_io.write_meta(str(ep_dir), meta)
    print(f"  -> meta.json: success={meta.get('success')}, valid={meta['valid']}"
          + (f", quality_issues={r.issue_codes}" if r.issue_codes else ""))


def _audit_targets(task: str, force_all: bool = False) -> list:
    """待审查列表：留存（success≠false）且 valid 仍为 null 的 Episode。

    不再要求 success is True —— 留存即成功，刚采完的 success=null 就是待审查
    的正常状态，要求显式 true 会让整批数据永远审不到。
    """
    collect = paths.collect_dir(task)
    if not collect.is_dir():
        raise SystemExit(f"[ERROR] 找不到 {collect}")
    out = []
    for d in sorted(x for x in collect.iterdir() if x.is_dir()):
        m = episode_io.load_meta(str(d)) or {}
        if force_all or (is_success(m) and m.get("valid") is None):
            out.append(d)
    return out


def _run_audit(task: str, targets: list, no_write: bool = False) -> tuple:
    """跑审查，返回 (pass 数, fail 数)。"""
    npass = nfail = 0
    for d in targets:
        r = audit_episode(str(d))
        print(r.render())
        if not no_write:
            _write_valid(d, task, r)
        print()
        npass += r.passed
        nfail += not r.passed
    print(f"审查完成: PASS={npass}  FAIL={nfail}")
    _write_quality_rejected(task)
    return npass, nfail


def cmd_audit_pending(args):
    """审查所有留存且 valid 仍为 null 的 Episode。"""
    task = task_name(args)
    targets = _audit_targets(task, args.all)
    if not targets:
        print("没有待审查的 Episode（留存且 valid=null；--all 可强制全审）")
        return 0
    print(f"待审查 {len(targets)} 条\n")
    _, nfail = _run_audit(task, targets, args.no_write)
    return 0 if nfail == 0 else 1


def _write_quality_rejected(task: str) -> pathlib.Path:
    """汇总 success=true 但 valid=false 的 Episode。文件很小，长期保存。"""
    collect = paths.collect_dir(task)
    items = []
    for d in sorted(x for x in collect.iterdir() if x.is_dir()):
        m = episode_io.load_meta(str(d)) or {}
        if is_success(m) and m.get("valid") is False:
            items.append({
                "episode_id": d.name,
                "quality_issues": m.get("quality_issues", []),
                "audited_at": m.get("audited_at"),
                "num_frames": m.get("num_frames"),
                "note": "任务成功但数据质量不合格；不自动删除，待人工确认",
            })
    out_dir = paths.rejected / task
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / QUALITY_REJECTED
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "task": task,
            "generated_at": _now_iso(),
            "policy": "技术审查失败不自动删除。确认确为坏数据后用 "
                      "manage_episodes.py delete-invalid 显式删除。",
            "count": len(items),
            "episodes": items,
        }, f, indent=2, ensure_ascii=False)
    print(f"[quality] {len(items)} 条质量不合格 -> {out}")
    return out


def cmd_build_train_manifest(args):
    """收录 success != false AND valid != false —— 留存即成功。

    默认会先自动审查所有未审查条目（--no-audit 可跳过）：把「审查」和「出清单」
    合成一步，是因为拆成两步在留存即成功策略下没有意义 —— 没人会想故意拿
    未审查的数据出清单。
    """
    task = task_name(args)
    collect = paths.collect_dir(task)
    if not collect.is_dir():
        raise SystemExit(f"[ERROR] 找不到 {collect}")

    # 先补审：valid=null 的条目在这里被审掉，避免未审查数据直接进清单
    if not args.no_audit:
        targets = _audit_targets(task)
        if targets:
            print(f"发现 {len(targets)} 条未审查，先自动审查\n")
            _run_audit(task, targets)
            print()

    included, excluded, unaudited = [], [], []
    for d in sorted(x for x in collect.iterdir() if x.is_dir()):
        m = episode_io.load_meta(str(d)) or {}
        s, v = m.get("success"), m.get("valid")
        if is_success(m) and v is not False:
            included.append(d.name)
            if v is None:
                unaudited.append(d.name)
        else:
            excluded.append({
                "episode_id": d.name,
                "success": s, "valid": v,
                "reason": "human_failed" if s is False else "quality_rejected",
                "quality_issues": m.get("quality_issues", []),
            })

    out = paths.task_dir(task) / TRAIN_MANIFEST
    doc = {
        "dataset_version": task,
        "generated_at": _now_iso(),
        "generator": "scripts/manage_episodes.py build-train-manifest",
        "policy": "留存即成功（implicit success）：人工失败的 Episode 在采集后立即 "
                  "reject-human 删除，因此磁盘上留存的数据即成功示范。",
        "criteria": "success != false AND valid != false",
        "raw_root": str(paths.raw),
        "raw_root_real": str(paths.raw.resolve()) if paths.raw.exists() else None,
        "episode_count": len(included),
        "episodes": included,
        "unaudited_count": len(unaudited),
        "unaudited": unaudited,
        "excluded_count": len(excluded),
        "excluded": excluded,
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
    print(f"[train-manifest] {len(included)} 条可训练 -> {out}")
    if unaudited:
        print(f"  ⚠ 其中 {len(unaudited)} 条未审查就收录了（--no-audit 的后果）: "
              f"{unaudited[:5]}{'...' if len(unaudited) > 5 else ''}")
    if excluded:
        print(f"  排除 {len(excluded)} 条:")
        for e in excluded:
            print(f"    {e['episode_id']}  {e['reason']}  "
                  f"success={e['success']} valid={e['valid']}"
                  + (f" {e['quality_issues']}" if e["quality_issues"] else ""))
    return out


def cmd_delete_invalid(args):
    """显式删除技术审查失败的数据。要求 valid 已为 false，且需二次确认。"""
    task = task_name(args)
    ep_dir = paths.episode_dir(task, args.episode_id)
    ep = assert_safe_to_delete(ep_dir, task)
    meta = episode_io.load_meta(str(ep)) or {}
    if meta.get("valid") is not False:
        raise SystemExit(f"[拒绝] {args.episode_id} 的 valid={meta.get('valid')}，"
                         f"不是 false。请先 audit 确认确为坏数据。")

    size = dir_size_mb(ep)
    print("即将删除技术审查失败的 Episode：")
    print(f"  完整路径 : {ep}")
    print(f"  质量问题 : {meta.get('quality_issues')}")
    print(f"  占用     : {size} MB")
    if not args.yes:
        if input("确认删除？只有输入 DELETE 才执行: ").strip() != "DELETE":
            raise SystemExit("[取消] 未执行任何删除")

    record = {
        "episode_id": ep.name, "time": _now_iso(),
        "reason": args.reason or "quality_rejected",
        "source": "quality_audit", "success": meta.get("success"),
        "task": task, "num_frames": meta.get("num_frames"), "size_mb": size,
        "note": f"quality_issues={meta.get('quality_issues')}",
        "deleted_path": str(ep),
    }
    log_path = append_rejected_log(task, record)
    print(f"[log] 已记录 -> {log_path}")
    shutil.rmtree(ep)
    if ep.exists():
        raise SystemExit(f"[ERROR] 删除后目录仍存在: {ep}")
    print(f"[deleted] {ep.name} 已删除，释放 {size} MB")


def main():
    ap = argparse.ArgumentParser(description="X-Trainer 人工示范数据管理")
    ap.add_argument("--task", default=None, help="数据集版本（任务目录名）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="列出所有 Episode 及 success/valid 状态")

    p = sub.add_parser("mark-success",
                       help="显式标记成功（留存即成功策略下通常不需要，用于撤销 mark-failed）")
    p.add_argument("episode_id")
    p.add_argument("--note", default=None)

    p = sub.add_parser("mark-failed", help="标记失败但保留数据（排除训练，不删除）")
    p.add_argument("episode_id")
    p.add_argument("--reason", required=True,
                   help="operator_failure / task_failed / wrong_trajectory / misoperation ...")
    p.add_argument("--note", default=None)

    p = sub.add_parser("label-position", help="给已采 Episode 补写位置泛化标注")
    p.add_argument("episode_ids", nargs="+")
    p.add_argument("--position-id", required=True, help="固定位置编号，如 p00_center")
    p.add_argument("--description", default=None, help="供人阅读的位置说明")
    p.add_argument("--position-object", default="socket")
    p.add_argument("--position-frame", default="table_reference")
    p.add_argument("--dx-mm", type=float, default=None)
    p.add_argument("--dy-mm", type=float, default=None)
    p.add_argument("--dz-mm", type=float, default=None)
    p.add_argument("--yaw-deg", type=float, default=None)
    p.add_argument("--split", choices=("train", "validation", "test"), default="train")

    p = sub.add_parser("reject-human", help="人工判失败：记录后删除原始数据")
    p.add_argument("episode_id")
    p.add_argument("--reason", required=True,
                   help="operator_failure / task_failed / wrong_trajectory / misoperation ...")
    p.add_argument("--note", default=None)
    p.add_argument("--yes", action="store_true", help="跳过交互确认（脚本用）")

    p = sub.add_parser("audit", help="审查单条并写回 valid")
    p.add_argument("episode_id")
    p.add_argument("--no-write", action="store_true", help="只看结果不写 meta.json")

    p = sub.add_parser("audit-pending", help="审查所有留存且 valid=null 的条目")
    p.add_argument("--all", action="store_true", help="强制全部重审")
    p.add_argument("--no-write", action="store_true")

    p = sub.add_parser("build-train-manifest",
                       help="自动审查未审查条目并生成训练清单")
    p.add_argument("--no-audit", action="store_true",
                   help="跳过自动审查，只按现有 meta.json 出清单")

    p = sub.add_parser("delete-invalid", help="显式删除已确认的坏数据（valid=false）")
    p.add_argument("episode_id")
    p.add_argument("--reason", default=None)
    p.add_argument("--yes", action="store_true")

    args = ap.parse_args()
    fn = {
        "list": cmd_list, "mark-success": cmd_mark_success,
        "mark-failed": cmd_mark_failed,
        "label-position": cmd_label_position,
        "reject-human": cmd_reject_human, "audit": cmd_audit,
        "audit-pending": cmd_audit_pending,
        "build-train-manifest": cmd_build_train_manifest,
        "delete-invalid": cmd_delete_invalid,
    }[args.cmd]
    rc = fn(args)
    sys.exit(rc if isinstance(rc, int) else 0)


if __name__ == "__main__":
    main()

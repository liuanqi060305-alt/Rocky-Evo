"""单条 Episode 的自动质量审查引擎。

设计意图：把审查逻辑集中在一处，供 manage_episodes.py（单条 PASS/FAIL + 写回
valid）和 check_episodes.py（批量分布概览）共用，避免两套判定标准打架。

阈值来源，全部沿用项目既有约定，不新拍脑袋：
  - 帧数 300-800        -> check_episodes.py:60 既有区间
  - 14 维 state/action  -> DATA_CONTRACT.md
  - 关节 rad / 夹爪 [0,1] -> DATA_AUDIT.md 第 3 节实测
  - 动作跳变 0.17 rad   -> run_inference.py:157 的安全阈值（约 10°）
  - 帧率参考 18.3 Hz    -> DATA_AUDIT.md P0-1 实测

审查只读，绝不修改 Episode 内容。写 meta.json 由调用方负责。
"""
import os
from typing import Any, Dict, List, Optional

import numpy as np

from scripts import episode_io

CAM_DIRS = ("topImg", "leftImg", "rightImg")

# 帧数合理区间，与 check_episodes.py 一致
MIN_FRAMES = 300
MAX_FRAMES = 800
# 单步关节增量阈值（rad），分两级 —— 这两个数不是拍脑袋，理由如下。
#
# 提示级 0.17 rad (≈10°)：run_inference.py:157 的推理期安全阈值。那道闸门防的是
# VLA 模型输出狂暴动作，**不是**人工遥操作的质量标准 —— 操作员本来就会快速移动。
# 实测既有 episode：关节增量 p50=0.44°、p99=7.75°、峰值 10.02°，且峰值处
# (帧415→425) 是 dim3 一段连续快动 (8.58→8.74→10.02→8.61°)，不是孤立尖刺。
# 所以超过它只提示，不判坏数据。
#
# 硬错级 0.35 rad (≈20°)：物理上界的约 2 倍。18.3Hz 下单帧 55ms，Nova 2 关节
# 速度上限约 180°/s -> 单帧理论最大约 9.9°，与实测峰值吻合。真正的数据断裂
# （丢帧造成位置跳跃、数值损坏）会远超此值，才判 FAIL。
ACTION_JUMP_WARN = 0.17
ACTION_JUMP_FAIL = 0.35
# 时间戳间隔上限（秒）。25Hz 目标 / 实测 18.3Hz -> 正常约 0.055s，
# 超过 0.5s 视为异常停顿（留足余量，避免把正常抖动误判为坏数据）
MAX_TS_GAP_S = 0.5
# 夹爪归一化允许范围，留一点浮点余量
GRIPPER_MIN, GRIPPER_MAX = -1e-6, 1.0 + 1e-6
# 关节角合理范围（rad）。±2π 是宽松上界，只用于抓明显离谱的值
JOINT_ABS_LIMIT = 2 * np.pi

GRIPPER_DIMS = (6, 13)


class AuditResult:
    """审查结果。errors 非空即 FAIL；warnings 不影响 PASS。"""

    def __init__(self, episode_id: str):
        self.episode_id = episode_id
        self.errors: List[str] = []      # 硬问题 -> valid=false
        self.warnings: List[str] = []    # 提示，不阻断
        self.issue_codes: List[str] = []  # 机器可读，写进 meta.quality_issues
        self.info: Dict[str, Any] = {}

    def fail(self, code: str, msg: str):
        self.errors.append(msg)
        if code not in self.issue_codes:
            self.issue_codes.append(code)

    def warn(self, msg: str):
        self.warnings.append(msg)

    @property
    def passed(self) -> bool:
        return not self.errors

    def render(self) -> str:
        head = "PASS" if self.passed else "FAIL"
        lines = [f"{self.episode_id}: {head}"]
        for e in self.errors:
            lines.append(f"  - {e}")
        for w in self.warnings:
            lines.append(f"  ! {w}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "result": "PASS" if self.passed else "FAIL",
            "errors": self.errors,
            "warnings": self.warnings,
            "quality_issues": self.issue_codes,
            "info": self.info,
        }


def _frame_ids(d: str, ext: str) -> Optional[List[int]]:
    """目录内按数值排序的帧号。目录不存在返回 None（与「空目录」区分开）。"""
    if not os.path.isdir(d):
        return None
    out = []
    for fn in os.listdir(d):
        if fn.endswith("." + ext):
            try:
                out.append(int(fn[: -(len(ext) + 1)]))
            except ValueError:
                pass
    return sorted(out)


def _gaps(ids: List[int]) -> List[int]:
    """编号缺口。判定方式与 check_episodes.py 一致：首尾之间缺哪些号。"""
    if not ids:
        return []
    return sorted(set(range(ids[0], ids[-1] + 1)) - set(ids))


def _fmt(seq, n=5):
    seq = list(seq)
    s = ", ".join(str(x) for x in seq[:n])
    return s + ("..." if len(seq) > n else "")


def _audit_images(ep_dir: str, r: AuditResult) -> Dict[str, List[int]]:
    """三路图像：目录存在性、数量、编号连续性、空文件、可解码性。"""
    import cv2

    ids_by_cam: Dict[str, List[int]] = {}
    for cam in CAM_DIRS:
        d = os.path.join(ep_dir, cam)
        ids = _frame_ids(d, "jpg")
        if ids is None:
            r.fail("missing_camera_dir", f"缺少目录 {cam}/")
            ids_by_cam[cam] = []
            continue
        if not ids:
            r.fail("empty_camera_dir", f"{cam}/ 内没有 jpg")
            ids_by_cam[cam] = []
            continue
        ids_by_cam[cam] = ids

        gaps = _gaps(ids)
        if gaps:
            r.fail("frame_gap", f"{cam} 编号不连续，缺 {len(gaps)} 帧: {_fmt(gaps)}")

        # 空文件必须逐个查（0 字节文件 imread 也返回 None，但要区分原因）
        empties = []
        for i in ids:
            p = os.path.join(d, f"{i}.jpg")
            try:
                if os.path.getsize(p) == 0:
                    empties.append(i)
            except OSError:
                empties.append(i)
        if empties:
            r.fail("empty_image_file", f"{cam} 有 {len(empties)} 个空文件: {_fmt(empties)}")

        # 解码抽样：首、尾、中间三帧 + 均匀抽至多 20 帧。
        # 全量解码 1500 张要数十秒，抽样足以抓住系统性损坏。
        probe = sorted(set([ids[0], ids[-1], ids[len(ids) // 2]]
                           + ids[:: max(1, len(ids) // 20)]))
        broken, black = [], []
        for i in probe:
            im = cv2.imread(os.path.join(d, f"{i}.jpg"))
            if im is None or im.size == 0:
                broken.append(i)
                continue
            if im.mean() < 1.0:
                black.append(i)
        if broken:
            r.fail("image_decode_error", f"{cam} 有图像无法解码: {_fmt(broken)}")
        if black:
            r.fail("black_image", f"{cam} 抽样出纯黑图像: {_fmt(black)}")

    return ids_by_cam


def _audit_observation(ep_dir: str, r: AuditResult):
    """observation：数量、可读性、必需字段、schema、数值健全性。"""
    d = os.path.join(ep_dir, "observation")
    ids = _frame_ids(d, "pkl")
    if ids is None:
        r.fail("missing_observation_dir", "缺少目录 observation/")
        return None
    if not ids:
        r.fail("empty_observation_dir", "observation/ 内没有 pkl")
        return None

    gaps = _gaps(ids)
    if gaps:
        r.fail("frame_gap", f"observation 编号不连续，缺 {len(gaps)} 帧: {_fmt(gaps)}")

    # 逐帧读原始 pkl 检查字段（load_episode 会归一化掉缺失情况，所以这里读原始）
    import pickle
    schemas, miss_jp, miss_ct, miss_ts, miss_mono, unreadable = set(), [], [], [], [], []
    for i in ids:
        p = os.path.join(d, f"{i}.pkl")
        try:
            with open(p, "rb") as f:
                raw = pickle.load(f)
        except Exception:
            unreadable.append(i)
            continue
        sv = int(raw.get("schema_version", 1))
        schemas.add(sv)
        if "joint_positions" not in raw:
            miss_jp.append(i)
        if "control" not in raw:
            miss_ct.append(i)
        if sv >= 2:
            if "timestamp_ns" not in raw:
                miss_ts.append(i)
            if "monotonic_ns" not in raw:
                miss_mono.append(i)

    if unreadable:
        r.fail("pkl_read_error", f"observation 有 pkl 无法读取: {_fmt(unreadable)}")
    if miss_jp:
        r.fail("missing_joint_positions", f"缺 joint_positions 的帧: {_fmt(miss_jp)}")
    if miss_ct:
        r.fail("missing_control", f"缺 control 的帧: {_fmt(miss_ct)}")
    if miss_ts:
        r.fail("missing_timestamp_ns", f"V2 数据缺 timestamp_ns 的帧: {_fmt(miss_ts)}")
    if miss_mono:
        r.fail("missing_monotonic_ns", f"V2 数据缺 monotonic_ns 的帧: {_fmt(miss_mono)}")

    if not schemas:
        r.fail("unknown_schema", "无法确定 schema_version")
    elif len(schemas) > 1:
        r.fail("mixed_schema", f"同一 episode 内 schema_version 不一致: {sorted(schemas)}")
    else:
        sv = next(iter(schemas))
        r.info["schema_version"] = sv
        if sv not in (1, 2):
            r.fail("unknown_schema", f"未知 schema_version={sv}")
        elif sv == 1:
            # V1 无时间戳，是历史数据，不算坏数据，但正式采集应为 V2
            r.warn("schema_version=1（旧格式，无时间戳）；正式采集应为 V2")

    return ids


def _audit_lengths(ids_by_cam, obs_ids, r: AuditResult):
    """四个模态的帧数与帧号集合一致性。"""
    counts = {c: len(v) for c, v in ids_by_cam.items()}
    counts["observation"] = len(obs_ids) if obs_ids else 0
    r.info["frame_counts"] = counts

    present = [v for v in counts.values() if v]
    if not present:
        return
    if len(set(counts.values())) > 1:
        r.fail("length_mismatch",
               "四模态帧数不一致: " + ", ".join(f"{k}={v}" for k, v in counts.items()))

    # 数量相同也可能帧号错位，必须比集合
    if obs_ids:
        obs_set = set(obs_ids)
        for cam, ids in ids_by_cam.items():
            if not ids:
                continue
            only_img = set(ids) - obs_set
            only_obs = obs_set - set(ids)
            if only_img:
                r.fail("frame_misalignment",
                       f"{cam} 有 {len(only_img)} 帧无对应 observation: {_fmt(sorted(only_img))}")
            if only_obs:
                r.fail("frame_misalignment",
                       f"observation 有 {len(only_obs)} 帧无对应 {cam}: {_fmt(sorted(only_obs))}")

    n = counts.get("observation", 0)
    if n and not (MIN_FRAMES <= n <= MAX_FRAMES):
        r.fail("abnormal_length",
               f"帧数 {n} 超出合理区间 [{MIN_FRAMES}, {MAX_FRAMES}]")


def _audit_values(ep_dir: str, r: AuditResult):
    """数值健全性：NaN/Inf、维度、量纲范围、动作跳变。"""
    try:
        ep = episode_io.load_episode(ep_dir)
    except Exception as e:
        r.fail("episode_load_error", f"无法读取 episode 数值: {e}")
        return None

    state, action = ep["state"], ep["action"]
    r.info["num_frames"] = len(ep["indices"])

    for name, arr in (("state", state), ("action", action)):
        if arr.ndim != 2 or arr.shape[1] != 14:
            r.fail("bad_dimension", f"{name} 形状为 {arr.shape}，应为 (T, 14)")
            continue
        nan_n, inf_n = int(np.isnan(arr).sum()), int(np.isinf(arr).sum())
        if nan_n:
            rows = _fmt(sorted(set(np.where(np.isnan(arr))[0].tolist())))
            r.fail("nan_value", f"{name} 有 {nan_n} 个 NaN，帧: {rows}")
        if inf_n:
            rows = _fmt(sorted(set(np.where(np.isinf(arr))[0].tolist())))
            r.fail("inf_value", f"{name} 有 {inf_n} 个 Inf，帧: {rows}")
        if nan_n or inf_n:
            continue

        # 夹爪维必须在 [0,1]（dynamixel.py:107-114 显式归一化并 clamp）
        for d in GRIPPER_DIMS:
            col = arr[:, d]
            bad = np.where((col < GRIPPER_MIN) | (col > GRIPPER_MAX))[0]
            if len(bad):
                r.fail("gripper_out_of_range",
                       f"{name}[{d}] 夹爪值超出 [0,1]，{len(bad)} 帧，"
                       f"范围 [{col.min():.4f}, {col.max():.4f}]")
        # 关节维用宽松上界只抓离谱值
        arm = [i for i in range(14) if i not in GRIPPER_DIMS]
        bad_joint = np.where(np.abs(arr[:, arm]) > JOINT_ABS_LIMIT)[0]
        if len(bad_joint):
            r.fail("joint_out_of_range",
                   f"{name} 关节角绝对值超过 {JOINT_ABS_LIMIT:.2f} rad，"
                   f"{len(set(bad_joint.tolist()))} 帧")

    # 动作跳变：只看关节维，夹爪本就会突变（开/合）
    if action.ndim == 2 and action.shape[1] == 14 and len(action) > 1:
        arm = [i for i in range(14) if i not in GRIPPER_DIMS]
        d = np.abs(np.diff(action[:, arm], axis=0))
        rowmax = d.max(axis=1)
        r.info["max_action_jump_rad"] = round(float(d.max()), 5) if d.size else None
        hard = np.where(rowmax > ACTION_JUMP_FAIL)[0]
        soft = np.where((rowmax > ACTION_JUMP_WARN) & (rowmax <= ACTION_JUMP_FAIL))[0]
        if len(hard):
            r.fail("action_jump",
                   f"action 关节增量超过 {ACTION_JUMP_FAIL} rad "
                   f"({np.rad2deg(ACTION_JUMP_FAIL):.0f}°) 的位置 {len(hard)} 处，"
                   f"最大 {d.max():.4f} rad，帧: {_fmt(hard.tolist())}")
        elif len(soft):
            r.warn(f"action 关节增量超过 {ACTION_JUMP_WARN} rad "
                   f"({np.rad2deg(ACTION_JUMP_WARN):.0f}°) 的位置 {len(soft)} 处，"
                   f"最大 {rowmax[soft].max():.4f} rad，帧: {_fmt(soft.tolist())}"
                   f" —— 快速人工操作的正常峰值，不判坏数据")
    return ep


def _audit_timestamps(ep, r: AuditResult):
    """V2 时间戳：单调性、重复、倒退、超大间隔；输出时长与实测帧率。"""
    if ep is None:
        return
    if ep["timestamp_ns"] is None and ep["monotonic_ns"] is None:
        # V1 数据无时间戳，上面已给 warning，这里不重复
        r.info["duration_s"] = None
        r.info["fps_measured"] = None
        return

    for key in ("timestamp_ns", "monotonic_ns"):
        arr = ep.get(key)
        if arr is None:
            continue
        d = np.diff(arr.astype(np.int64))
        back = np.where(d < 0)[0]
        dup = np.where(d == 0)[0]
        if len(back):
            r.fail("timestamp_backward",
                   f"{key} 出现倒退 {len(back)} 处，帧: {_fmt(back.tolist())}")
        if len(dup):
            r.fail("timestamp_duplicate",
                   f"{key} 出现重复 {len(dup)} 处，帧: {_fmt(dup.tolist())}")
        big = np.where(d > MAX_TS_GAP_S * 1e9)[0]
        if len(big):
            r.fail("timestamp_gap",
                   f"{key} 出现超过 {MAX_TS_GAP_S}s 的间隔 {len(big)} 处，"
                   f"最大 {d.max()/1e9:.3f}s，帧: {_fmt(big.tolist())}")

    base = ep["monotonic_ns"] if ep["monotonic_ns"] is not None else ep["timestamp_ns"]
    if base is not None and len(base) > 1:
        span = (int(base[-1]) - int(base[0])) / 1e9
        r.info["duration_s"] = round(span, 4)
        r.info["fps_measured"] = round((len(base) - 1) / span, 3) if span > 0 else None


def audit_episode(ep_dir: str) -> AuditResult:
    """审查一条 Episode。只读，不写 meta.json。"""
    r = AuditResult(os.path.basename(os.path.normpath(ep_dir)))
    if not os.path.isdir(ep_dir):
        r.fail("missing_episode_dir", f"目录不存在: {ep_dir}")
        return r

    meta = episode_io.load_meta(ep_dir)
    r.info["has_meta_json"] = meta is not None
    r.info["success"] = (meta or {}).get("success")

    ids_by_cam = _audit_images(ep_dir, r)
    obs_ids = _audit_observation(ep_dir, r)
    _audit_lengths(ids_by_cam, obs_ids, r)
    ep = _audit_values(ep_dir, r) if obs_ids else None
    _audit_timestamps(ep, r)
    return r

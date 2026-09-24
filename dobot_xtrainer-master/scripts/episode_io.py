"""Episode 数据读写统一层（Data Collection V2）。

存在的理由：V2 落盘格式与 V1 不同（去掉了语义错误的字段、新增时间戳），
但磁盘上仍有 V1 数据。所有读取方都应经过本模块，而不是直接 pickle.load，
这样 V1/V2 差异只在这里处理一次。

V1 pkl（旧，5 键）:
    joint_positions  (14,) float64   —— 有效
    control          (14,) float64   —— 有效
    joint_velocities (14,) float64   —— 实为 joint_positions 副本，无效
    ee_pos_quat      (14,) float64   —— 恒零，无效
    gripper_position (2,)  float64   —— 恒 1.0，无效

V2 pkl（新）:
    joint_positions  (14,) float64   —— 有效（含义与 V1 相同）
    control          (14,) float64   —— 有效（含义与 V1 相同）
    timestamp_ns     int             —— time.time_ns()，墙钟，跨进程可比
    monotonic_ns     int             —— time.monotonic_ns()，算间隔用，不受 NTP 跳变影响
    schema_version   int             —— 固定 2

V2 去掉了 joint_velocities / ee_pos_quat / gripper_position 三个键。
它们在 V1 里就不含任何信息（见 DATA_AUDIT.md 第 3 节），继续导出会误导下游。
其"不可用"这一事实由 episode 级 meta.json 的 *_available 标志显式声明。
"""
import glob
import json
import os
import pickle
from typing import Any, Dict, List, Optional

import numpy as np

SCHEMA_VERSION = 2
DATASET_VERSION = "2.0"
META_FILENAME = "meta.json"

# V1 中存在但无有效内容的键。读取时一律剥离，避免下游误用。
_V1_DEAD_KEYS = ("joint_velocities", "ee_pos_quat", "gripper_position")


def detect_schema_version(frame: Dict[str, Any]) -> int:
    """从单帧内容判定 schema 版本。V1 没有 schema_version 键。"""
    return int(frame.get("schema_version", 1))


def load_frame(path: str) -> Dict[str, Any]:
    """读单帧 pkl，归一化成 V2 视图。

    V1 与 V2 都返回同样的键集合：
        joint_positions, control, timestamp_ns, monotonic_ns,
        schema_version, _source_schema
    V1 数据没有时间戳，对应字段返回 None —— 调用方必须能处理 None
    （这是"新增字段允许为空"的落地方式）。
    """
    with open(path, "rb") as f:
        raw = pickle.load(f)

    src = detect_schema_version(raw)
    out = {
        "joint_positions": np.asarray(raw["joint_positions"], dtype=np.float64),
        "control": np.asarray(raw["control"], dtype=np.float64),
        "timestamp_ns": raw.get("timestamp_ns"),      # V1 -> None
        "monotonic_ns": raw.get("monotonic_ns"),      # V1 -> None
        "schema_version": SCHEMA_VERSION,
        "_source_schema": src,
    }
    return out


def frame_indices(episode_dir: str, subdir: str = "observation", ext: str = "pkl") -> List[int]:
    """列出 episode 内的帧号，升序。按数值排序，不是字典序。"""
    pat = os.path.join(episode_dir, subdir, "*." + ext)
    return sorted(int(os.path.basename(p)[: -(len(ext) + 1)]) for p in glob.glob(pat))


def load_episode(episode_dir: str) -> Dict[str, Any]:
    """读整条 episode 的 state/action/时间戳。

    返回:
        indices      : List[int]
        state        : (T, 14) float64
        action       : (T, 14) float64
        timestamp_ns : (T,) int64 或 None（V1 数据）
        monotonic_ns : (T,) int64 或 None（V1 数据）
        source_schema: 1 或 2
    """
    idxs = frame_indices(episode_dir)
    if not idxs:
        raise FileNotFoundError(f"{episode_dir} 下没有 observation/*.pkl")

    frames = [load_frame(os.path.join(episode_dir, "observation", f"{i}.pkl")) for i in idxs]
    ts = [f["timestamp_ns"] for f in frames]
    mono = [f["monotonic_ns"] for f in frames]

    return {
        "indices": idxs,
        "state": np.stack([f["joint_positions"] for f in frames]),
        "action": np.stack([f["control"] for f in frames]),
        "timestamp_ns": np.asarray(ts, dtype=np.int64) if all(t is not None for t in ts) else None,
        "monotonic_ns": np.asarray(mono, dtype=np.int64) if all(t is not None for t in mono) else None,
        "source_schema": frames[0]["_source_schema"],
    }


def load_meta(episode_dir: str) -> Optional[Dict[str, Any]]:
    """读 episode 级 meta.json。V1 数据没有该文件，返回 None。"""
    p = os.path.join(episode_dir, META_FILENAME)
    if not os.path.isfile(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def effective_fps(episode_dir: str, fallback: float = 25.0) -> float:
    """算 episode 的真实帧率。

    优先用 monotonic_ns（不受墙钟跳变影响）；其次 timestamp_ns；
    V1 数据两者都无，退回 meta.json 的 fps_measured，最后才用 fallback。

    这解决 DATA_AUDIT.md P0-1：代码标称 25Hz、实测 18.3Hz。
    """
    try:
        ep = load_episode(episode_dir)
    except Exception:
        return fallback

    for key in ("monotonic_ns", "timestamp_ns"):
        arr = ep.get(key)
        if arr is not None and len(arr) > 1:
            span_s = (int(arr[-1]) - int(arr[0])) / 1e9
            if span_s > 0:
                return (len(arr) - 1) / span_s

    meta = load_meta(episode_dir)
    if meta and meta.get("fps_measured"):
        return float(meta["fps_measured"])
    return fallback


def build_meta(
    episode_id: str,
    task_name: str,
    episode_start_time_ns: int,
    episode_start_monotonic_ns: int,
    left_position=None,
    left_angle=None,
    right_position=None,
    right_angle=None,
    data_type: str = "teleop",
    note: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """构造 episode 级 metadata。

    三个 *_available 标志是本次 V2 升级的核心声明：它们把"这些字段没有真实数据"
    这一事实写进数据本身，而不是只存在于文档里。依据见 DATA_AUDIT.md 第 3 节。

    success / valid 初始为 null —— 采集程序无法自动判定插拔是否成功
    （DATA_AUDIT.md 第 8 节），也不该在主循环里猜。

    下游按「留存即成功」解读 success（见 manage_episodes.py 的 IMPLICIT_SUCCESS）：
    人工失败的 Episode 在采集后立即 reject-human 删除，所以留在盘上的 null
    就是成功示范。这里仍写 null 而不是 true，是因为写 meta 的时刻录制刚结束、
    人还没判断，此时声称 true 就是编造；true 由 audit 落定并附
    success_source="implicit_retained"。null 与 false（已判定失败）语义不同。
    """
    def _l(v):
        return None if v is None else [float(x) for x in np.asarray(v).ravel()]

    meta = {
        "schema_version": SCHEMA_VERSION,
        "dataset_version": DATASET_VERSION,
        "episode_id": episode_id,
        "task_name": task_name,

        # ── 时间 ──
        "episode_start_time": episode_start_time_ns,          # time.time_ns() 墙钟
        "episode_start_time_iso": _ns_to_iso(episode_start_time_ns),
        "episode_start_monotonic_ns": episode_start_monotonic_ns,
        "timestamp": _ns_to_iso(episode_start_time_ns),        # 人可读别名
        "num_frames": None,        # 录制结束时回填
        "duration_s": None,        # 录制结束时回填
        "fps_measured": None,      # 录制结束时回填，取代硬编码 25

        # ── 起始位姿（录制开始时采样一次；不进主循环，避免影响控制频率）──
        "left_position": _l(left_position),    # TCP [X, Y, Z, rx, ry, rz]，位置 mm / 姿态 deg
        "left_angle": _l(left_angle),          # 左臂 7 维关节，rad（第 7 维为归一化夹爪）
        "right_position": _l(right_position),
        "right_angle": _l(right_angle),

        # ── 数据可用性声明（V2 新增，解决审计 P1-4/5/6）──
        "velocity_available": False,   # 无真实关节速度，V2 已不再导出该字段
        "ee_pose_available": False,    # ee_pos_quat 恒零，V2 已不再导出
        "gripper_feedback": False,     # 从手无夹爪反馈；state[6]/[13] 实为上一帧指令

        # ── 人工标注位 ──
        "data_type": data_type,        # teleop / replay / policy_rollout
        "success": None,               # 待人工标注：true / false
        "valid": None,                 # 待人工标注：true / false
        "note": note,

        # ── 采集环境（便于日后复现与排查）──
        "image_layout": {
            "cameras": ["top", "left", "right"],
            "dirs": ["topImg", "leftImg", "rightImg"],
            "resolution": [480, 640, 3],
            "channel_order_on_disk": "BGR",
            "flip": {"top": True, "left": False, "right": True},
        },
        "state_layout": {
            "dim": 14,
            "left_joints": [0, 1, 2, 3, 4, 5],
            "left_gripper": 6,
            "right_joints": [7, 8, 9, 10, 11, 12],
            "right_gripper": 13,
            "joint_unit": "rad",
            "gripper_unit": "normalized[0,1]",
        },
        "action_layout": {
            "dim": 14,
            "semantics": "absolute joint target",
            "same_layout_as_state": True,
        },
    }
    if extra:
        meta.update(extra)
    return meta


def _ns_to_iso(ns) -> Optional[str]:
    if ns is None:
        return None
    import datetime
    return datetime.datetime.fromtimestamp(int(ns) / 1e9).isoformat(timespec="milliseconds")


def write_meta(episode_dir: str, meta: Dict[str, Any]) -> str:
    """把 meta 写到 episode_dir/meta.json。目录不存在时自动创建。"""
    os.makedirs(episode_dir, exist_ok=True)
    p = os.path.join(episode_dir, META_FILENAME)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return p


def finalize_meta(episode_dir: str) -> Optional[Dict[str, Any]]:
    """录制结束后回填 num_frames / duration_s / fps_measured。

    单独一步做，因为这三项只有等 episode 写完才知道。
    meta.json 不存在（例如 V1 数据）时返回 None，不报错。
    """
    meta = load_meta(episode_dir)
    if meta is None:
        return None
    try:
        ep = load_episode(episode_dir)
    except Exception:
        return meta

    meta["num_frames"] = len(ep["indices"])
    arr = ep["monotonic_ns"] if ep["monotonic_ns"] is not None else ep["timestamp_ns"]
    if arr is not None and len(arr) > 1:
        span = (int(arr[-1]) - int(arr[0])) / 1e9
        meta["duration_s"] = round(span, 4)
        meta["fps_measured"] = round((len(arr) - 1) / span, 3) if span > 0 else None
    write_meta(episode_dir, meta)
    return meta

import datetime
import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np


SCHEMA_VERSION = 2

# V1 里存在但不含任何有效信息的键，V2 落盘时剔除。依据 DATA_AUDIT.md 第 3 节：
#   joint_velocities —— dobot.py:200 与 joint_positions 是同一个变量，494/494 帧完全相等
#   ee_pos_quat      —— dobot.py:196 硬编码 np.zeros(7)，494/494 帧全零
#   gripper_position —— dobot.py:108 硬编码 [1.0]，恒为 [1,1]
# 字段名叫 velocity 而内容是 position 会直接误导下游训练，因此不再导出。
# 这三项"不可用"的事实改由 meta.json 的 velocity_available / ee_pose_available /
# gripper_feedback 显式声明。
_DROP_KEYS = ("joint_velocities", "ee_pos_quat", "gripper_position")


def save_frame(
    folder: str,
    timestamp: int,
    obs: Dict[str, np.ndarray],
    action: np.ndarray,
    timestamp_ns: Optional[int] = None,
    monotonic_ns: Optional[int] = None,
) -> None:
    """写单帧 observation+action 到 {folder}{timestamp}.pkl（V2 schema）。

    Args:
        folder: 目标目录，必须以 '/' 结尾（沿用 V1 调用约定）
        timestamp: 帧序号，用作文件名（沿用 V1 语义，不是时间）
        obs: env.get_obs() 的返回
        action: 本帧下发的绝对关节角目标
        timestamp_ns: time.time_ns()，墙钟。由调用方在主循环里采样一次，
            确保 image / state / action 共用同一时间基准
        monotonic_ns: time.monotonic_ns()，算间隔用，不受 NTP 调时影响

    时间戳设计为可选参数：不传时退化成 V1 行为（无时间戳字段），
    保证任何未升级的调用方仍能工作。
    """
    # 不原地改调用方的 dict —— V1 版 `obs["control"] = action` 会污染
    # 调用方持有的 obs 对象。run_control.py 主循环里 obs 会跨帧复用。
    frame = {k: v for k, v in obs.items() if k not in _DROP_KEYS}
    frame["control"] = action
    frame["schema_version"] = SCHEMA_VERSION
    if timestamp_ns is not None:
        frame["timestamp_ns"] = int(timestamp_ns)
    if monotonic_ns is not None:
        frame["monotonic_ns"] = int(monotonic_ns)

    recorded_file = folder + str(timestamp) + ".pkl"
    print(recorded_file)

    with open(recorded_file, "wb") as f:
        pickle.dump(frame, f)

def save_action(recorded_file,action: np.ndarray):
    with open(recorded_file, "ab") as f:
        pickle.dump(action, f)


if __name__ == "__main__":
    # test write
    act = [1,2,3,4,5,6]




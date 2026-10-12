"""用 openpi (pi0.5) 策略服务端驱动 X-Trainer 双臂。

架构：本脚本跑在 conda `x_trainer` (Python 3.8)，负责相机/机械臂 IO；
模型跑在 openpi 的 uv `.venv` (Python 3.11)，由 scripts/serve_policy.py 起
WebSocket 服务端。两边只传 numpy 数组，不共享 Python 依赖 —— 这是绕开
「Dobot 锁 3.8 / openpi 要 3.11」版本互斥的官方做法。

先在智算实例启动 scripts/serve_xtrainer_szu.sh，再从真机电脑建立 SSH
本地端口转发，使服务在本机表现为 127.0.0.1:8000。然后运行本脚本
（需要 launch_nodes.py 已在另一个终端跑着）：
    conda activate x_trainer
    python experiments/run_inference_openpi.py --condition=original --operator=<姓名>

程序不会自动开始运动。按 r 开始一条 rollout，按 s/f 标记成功/失败；运行中
按 h 会中止本条并复位，待机时按 h 只复位。s/f/h 后程序继续运行，按 q 结束，
按 e 停止继续下发动作并结束（需要立即断动力时使用实体急停按钮）。数据和
成功率默认写到 RockyEVO/data/rollouts/。
"""
import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import csv
import ast
import datetime
import json
import pathlib
import pickle
import queue
import re
import select
import termios
import time
import threading
import tty
from dataclasses import dataclass

# OpenCV wheel 自带的 Qt 插件没有打包字体，显式使用系统 DejaVu 字体目录，
# 避免开启 --show-img 时反复打印 QFontDatabase 警告。
os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")

import cv2
import numpy as np
import tyro
import zmq
from openpi_client import image_tools
from openpi_client import websocket_client_policy

from dobot_control.agents import async_policy
from dobot_control.cameras.realsense_camera import RealSenseCamera
from dobot_control.env import RobotEnv
from dobot_control.robots.robot_node import RobotServerError, ZMQClientRobot
from scripts.check_dobot_alarm import (
    CONTROLLERS, ControllerStatus, format_status, parse_error_ids,
    parse_robot_mode,
)
from scripts.manipulate_utils import load_ini_data_camera


CONTROL_PERIOD_S = 0.04
ROBOT_IO_EXCEPTIONS = (RobotServerError, TimeoutError, ConnectionError, zmq.ZMQError)
# 从 24 条成功和 3 条控制器报警的 v2 rollout 离线回放得到；仅是
# 插接前的保守停机征兆，不是经过力传感器标定的接触力阈值。
CONTACT_RIGHT_J3_POSE_DEG = (65.0, 75.0)
CONTACT_RIGHT_J4_POSE_DEG = (-70.0, -60.0)
CONTACT_PREVIOUS_TARGET_LAG_DEG = 1.1
CONTACT_ACTUAL_REVERSAL_DEG = -0.4
CONTACT_MIN_TARGET_DELTA_DEG = -0.4
# 保留原先触发 y 提示的 0.17rad 阈值和 y 之后的平滑插值行为。自动模式只
# 消除人工等待；不放大单步指令，也不改写模型动作。25deg 覆盖当前 20 条成功
# rollout 中观察到的最大值 21.97deg，超出该范围仍会拒绝当前动作。
JOINT_JUMP_WARNING_RAD = 0.17
AUTO_SMOOTH_JOINT_JUMP_MAX_DEG = 25.0

# 原 XYZ 保护要求两臂 Z>42mm。左臂桌面拾取只在 V2 轨迹覆盖的局部盒内
# 允许 30--42mm，右臂保留已有的低位插接特例。其它 XYZ 越界仍会
# 终止当前 rollout。
WORKSPACE_Z_MIN_MM = 42.0
LEFT_PICKUP_Z_FLOOR_MM = 30.0
LEFT_PICKUP_X_RANGE_MM = (-201.0, -158.0)
LEFT_PICKUP_Y_RANGE_MM = (-407.0, -359.0)
LEFT_PICKUP_MAX_DURATION_S = 2.0
RIGHT_INSERTION_Z_FLOOR_MM = 30.0
NOVA2_TOOL_Z_MM = 197.0
# Per-dimension median of frame-0 control targets from the 54 episodes used by
# plug_v1_54eps_full_h100. The former rounded pose put LJ5/RJ5 13.8/8.7 degrees
# outside the demonstration start median and pushed both wrists beyond q01/q99.
TRAINING_INITIAL_TARGET = np.array([
    -1.52949497, 0.06779639, -1.63175606, 0.05962092,
    1.33226192, 1.51796775, 0.96770833,
    1.55728214, -0.00132979, 1.51690270, 0.08103833,
    -1.42004150, -1.64121355, 0.97649740,
], dtype=np.float64)


@dataclass
class Args:
    # 策略服务端地址（scripts/serve_policy.py 默认监听 0.0.0.0:8000）
    remote_host: str = "127.0.0.1"
    remote_port: int = 8000
    # launch_nodes.py 和 run_control.py 当前默认都使用 6002。
    robot_port: int = 6002
    hostname: str = "127.0.0.1"
    # 覆盖服务端的默认 prompt。留空则用训练时 task 字段（prompt_from_task=True）。
    prompt: str = "plug and unplug"
    # 异步模式下表示每执行多少步提交一次后台视觉重规划；同步回退模式下表示
    # 每个动作块开环执行多少步。默认 5 步约 200ms。
    open_loop_horizon: int = 5
    # 后台推理保持控制循环连续；可用 --no-async-inference 回退到原同步实现。
    async_inference: bool = True
    # 新旧动作块交界处对十二个机械臂关节做短时插值，夹爪指令不插值。
    action_blend_steps: int = 3
    # 将对同一时刻的最近几个随机动作块做时间集成，抑制每次重规划时交替反向。
    # decay 越大越信任最新视觉；0.7 对三个块的权重约为 0.57/0.29/0.14。
    temporal_ensemble_chunks: int = 3
    temporal_ensemble_decay: float = 0.7
    # 运动中后台请求超过此时间仍未返回，则安全结束当前 rollout。
    inference_timeout_s: float = 2.0
    # 首次 JAX 编译可能需要 20~30 秒；初始动作在机器人静止时等待。
    initial_inference_timeout_s: float = 60.0
    # 推理只在每次请求策略时取最新帧，不需要采集阶段的 90fps。三路 30fps 可将
    # 同一 USB Hub 的视频流量降到约三分之一，避免长时间推理时 UVC -71/-32。
    camera_fps: int = 30
    # D405 在 30fps 自动曝光时会从训练时约 11ms 增长到约 33ms，造成大面积过曝。
    # 固定 11ms、最低增益可保持接近 90fps 训练数据的成像；设 exposure 为 0 可恢复自动曝光。
    camera_exposure_us: float = 11000.0
    camera_gain: float = 16.0
    max_steps: int = 9000
    show_img: bool = False
    # 干跑：只推理不下发关节指令，用来验证链路和数值范围
    dry_run: bool = False
    # 真机测评数据。留空时写入 <RockyEVO>/data/rollouts/<task_name>/。
    rollout_root: str = ""
    task_name: str = "plug_and_unplug_task"
    # 人可读的 checkpoint 标识，写入每条 rollout 和汇总 CSV。
    checkpoint_label: str = "pi05_xtrainer_full/plug_v1_54eps_full_h100/29999"
    # 当前测试条件，如 original / unseen_position_01。只用于测评记录。
    condition: str = ""
    operator: str = ""
    # 每次按 r 开始新 rollout 前，自动经安全位回到统一初始位。
    reset_before_rollout: bool = True
    save_images: bool = True
    # 默认自动处理历史成功轨迹中固定出现的 y/i 分支，使控制循环不等待人工按键。
    # 调试或 A/B 对照时增加 --interactive-safety-prompts 可恢复原来的 y/i 提示。
    interactive_safety_prompts: bool = False
    # 仅在已验证的右臂插接姿态监测“目标继续前进、实际关节突然反向且已落后目标”。
    # 不改变任何正常动作；可用 --no-contact-guard 做对照。有限样本不能保证拦下所有碰撞。
    contact_guard: bool = True

    def resolved_rollout_root(self) -> pathlib.Path:
        if self.rollout_root:
            return pathlib.Path(self.rollout_root).expanduser().resolve()
        project_root = pathlib.Path(BASE_DIR).resolve().parents[1]
        return project_root / "data" / "rollouts" / self.task_name


class TerminalKeys:
    """无须回车的终端按键读取；退出时恢复终端设置。"""

    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.old = None

    def __enter__(self):
        if not sys.stdin.isatty():
            raise RuntimeError("键盘测评需要在交互式终端运行（stdin 必须是 TTY）")
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def poll(self):
        readable, _, _ = select.select([sys.stdin], [], [], 0)
        if not readable:
            return None
        return sys.stdin.read(1).lower()

    def wait(self):
        while True:
            key = self.poll()
            if key is not None:
                return key
            time.sleep(0.02)

    def __exit__(self, exc_type, exc, tb):
        if self.old is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)


def select_key_input(terminal_key=None, opencv_key=None, accepted_keys=None):
    """合并终端与 OpenCV 按键，并优先保留终端已经读到的有效按键。

    OpenCV 窗口偶尔会留下无关按键码。旧逻辑会用它覆盖同一轮已经从终端读到的
    ``y``，安全提示便会把本条误判为失败。传入 ``accepted_keys`` 时会跳过两个
    输入源中的无关按键，只返回提示明确允许的决定键。
    """
    candidates = []
    if terminal_key:
        candidates.append(str(terminal_key).lower())

    if opencv_key is not None:
        key_code = int(opencv_key) & 0xFF
        if key_code != 0xFF:
            candidates.append(chr(key_code).lower())

    if accepted_keys is not None:
        accepted = {key.lower() for key in accepted_keys}
        candidates = [key for key in candidates if key in accepted]

    return candidates[0] if candidates else None


def _iso_now():
    return datetime.datetime.now().isoformat(timespec="milliseconds")


class RolloutRecorder:
    """异步保存真机 rollout，并维护成功率汇总 CSV/JSON。"""

    CSV_FIELDS = [
        "rollout_id", "session_id", "start_time", "end_time", "result", "success",
        "num_steps", "duration_s", "task_name", "condition", "prompt", "checkpoint",
        "server", "operator", "dry_run", "failure_reason", "rollout_path",
    ]

    def __init__(self, args, server_metadata):
        self.args = args
        self.root = args.resolved_rollout_root()
        self.root.mkdir(parents=True, exist_ok=True)
        self.session_id = datetime.datetime.now().strftime("session_%Y%m%d_%H%M%S")
        self.session_dir = self.root / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.server_metadata = server_metadata
        self.active = False
        self.writer_error = None
        self.summary = {
            "success": 0,
            "failure": 0,
            "evaluated": 0,
            "success_rate": None,
        }
        if (self.root / "rollout_results.csv").exists():
            self.summary = self._update_summary()

    def start(self):
        if self.active:
            raise RuntimeError("已有 rollout 正在记录")
        self.rollout_id = datetime.datetime.now().strftime("rollout_%Y%m%d_%H%M%S_%f")
        self.rollout_dir = self.session_dir / self.rollout_id
        self.obs_dir = self.rollout_dir / "observation"
        self.image_dirs = {
            "top": self.rollout_dir / "topImg",
            "left": self.rollout_dir / "leftImg",
            "right": self.rollout_dir / "rightImg",
        }
        self.obs_dir.mkdir(parents=True)
        if self.args.save_images:
            for d in self.image_dirs.values():
                d.mkdir(parents=True)
        self.start_wall_ns = time.time_ns()
        self.start_mono_ns = time.monotonic_ns()
        self.num_steps = 0
        self.events = []
        self.writer_error = None
        self.write_queue = queue.Queue(maxsize=256)
        self.writer = threading.Thread(target=self._writer_loop, daemon=True)
        self.writer.start()
        self.active = True
        self._write_meta(result=None, failure_reason="")
        return self.rollout_dir

    def add_event(self, event_type, **details):
        if self.active:
            self.events.append({
                "time": _iso_now(),
                "type": event_type,
                **details,
            })

    def _writer_loop(self):
        while True:
            item = self.write_queue.get()
            try:
                if item is None:
                    return
                idx, record, images = item
                with open(self.obs_dir / f"{idx}.pkl", "wb") as f:
                    pickle.dump(record, f, protocol=pickle.HIGHEST_PROTOCOL)
                if self.args.save_images:
                    for name, image in zip(("top", "left", "right"), images):
                        ok = cv2.imwrite(
                            str(self.image_dirs[name] / f"{idx}.jpg"),
                            cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                        )
                        if not ok:
                            raise OSError(f"failed to write {name} image {idx}")
            except Exception as e:
                self.writer_error = f"{type(e).__name__}: {e}"
            finally:
                self.write_queue.task_done()

    def record(
        self,
        state,
        action,
        infer_latency_ms,
        chunk_index,
        images,
        ensemble_sources=1,
    ):
        if not self.active:
            return
        idx = self.num_steps
        record = {
            "schema_version": 1,
            "timestamp_ns": time.time_ns(),
            "monotonic_ns": time.monotonic_ns(),
            "joint_positions": np.asarray(state, dtype=np.float64),
            "control": np.asarray(action, dtype=np.float64),
            "infer_latency_ms": infer_latency_ms,
            "chunk_index": int(chunk_index),
            "ensemble_sources": int(ensemble_sources),
            "data_type": "policy_rollout",
        }
        image_copies = tuple(np.ascontiguousarray(x).copy() for x in images)
        self.write_queue.put((idx, record, image_copies))
        self.num_steps += 1

    def _write_meta(self, result, failure_reason):
        end_ns = time.time_ns() if result is not None else None
        duration = ((time.monotonic_ns() - self.start_mono_ns) / 1e9
                    if result is not None else None)
        meta = {
            "schema_version": 1,
            "data_type": "policy_rollout",
            "rollout_id": self.rollout_id,
            "session_id": self.session_id,
            "task_name": self.args.task_name,
            "condition": self.args.condition,
            "prompt": self.args.prompt,
            "checkpoint": self.args.checkpoint_label,
            "server": f"{self.args.remote_host}:{self.args.remote_port}",
            "server_metadata": self.server_metadata,
            "operator": self.args.operator,
            "dry_run": self.args.dry_run,
            "control_config": {
                "async_inference": self.args.async_inference,
                "replan_interval_steps": self.args.open_loop_horizon,
                "action_blend_steps": self.args.action_blend_steps,
                "temporal_ensemble_chunks": self.args.temporal_ensemble_chunks,
                "temporal_ensemble_decay": self.args.temporal_ensemble_decay,
                "inference_timeout_s": self.args.inference_timeout_s,
                "control_period_s": CONTROL_PERIOD_S,
                "camera_fps": self.args.camera_fps,
                "camera_exposure_us": self.args.camera_exposure_us,
                "camera_gain": self.args.camera_gain,
                "reset_pose": "median_frame0_control_of_54_training_episodes",
                "interactive_safety_prompts": self.args.interactive_safety_prompts,
                "joint_jump_warning_deg": float(
                    np.rad2deg(JOINT_JUMP_WARNING_RAD)
                ),
                "auto_smooth_joint_jump_max_deg": (
                    AUTO_SMOOTH_JOINT_JUMP_MAX_DEG
                ),
                "left_pickup_bounds_mm": {
                    "x": LEFT_PICKUP_X_RANGE_MM,
                    "y": LEFT_PICKUP_Y_RANGE_MM,
                    "z": (LEFT_PICKUP_Z_FLOOR_MM, WORKSPACE_Z_MIN_MM),
                    "x_y_bounds_exclusive": True,
                    "z_lower_exclusive": True,
                    "z_upper_inclusive": True,
                },
                "left_pickup_max_duration_s": LEFT_PICKUP_MAX_DURATION_S,
                "left_action_target_fk": "Nova2_DH_with_197mm_TCP",
                "right_insertion_z_floor_mm": RIGHT_INSERTION_Z_FLOOR_MM,
                "contact_guard": self.args.contact_guard,
                "contact_guard_rule": {
                    "right_j3_pose_deg": CONTACT_RIGHT_J3_POSE_DEG,
                    "right_j4_pose_deg": CONTACT_RIGHT_J4_POSE_DEG,
                    "previous_target_lag_gt_deg": CONTACT_PREVIOUS_TARGET_LAG_DEG,
                    "actual_j3_delta_lt_deg": CONTACT_ACTUAL_REVERSAL_DEG,
                    "target_j3_delta_gt_deg": CONTACT_MIN_TARGET_DELTA_DEG,
                },
            },
            "start_time_ns": self.start_wall_ns,
            "start_time": datetime.datetime.fromtimestamp(self.start_wall_ns / 1e9).isoformat(timespec="milliseconds"),
            "end_time_ns": end_ns,
            "end_time": (_iso_now() if end_ns is not None else None),
            "num_steps": self.num_steps,
            "duration_s": (round(duration, 3) if duration is not None else None),
            "result": result,
            "success": (True if result == "success" else False if result == "failure" else None),
            "failure_reason": failure_reason,
            "events": self.events,
            "writer_error": self.writer_error,
        }
        with open(self.rollout_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False, default=str)
        return meta

    def finish(self, result, failure_reason=""):
        if not self.active:
            return None
        self.write_queue.put(None)
        self.write_queue.join()
        self.writer.join(timeout=10)
        meta = self._write_meta(result=result, failure_reason=failure_reason)
        self._append_csv(meta)
        self.summary = self._update_summary()
        self.active = False
        return meta

    def _append_csv(self, meta):
        path = self.root / "rollout_results.csv"
        row = {
            "rollout_id": meta["rollout_id"],
            "session_id": meta["session_id"],
            "start_time": meta["start_time"],
            "end_time": meta["end_time"],
            "result": meta["result"],
            "success": meta["success"],
            "num_steps": meta["num_steps"],
            "duration_s": meta["duration_s"],
            "task_name": meta["task_name"],
            "condition": meta["condition"],
            "prompt": meta["prompt"],
            "checkpoint": meta["checkpoint"],
            "server": meta["server"],
            "operator": meta["operator"],
            "dry_run": meta["dry_run"],
            "failure_reason": meta["failure_reason"],
            "rollout_path": str(self.rollout_dir),
        }
        new_file = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_FIELDS)
            if new_file:
                writer.writeheader()
            writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())

    def _update_summary(self):
        csv_path = self.root / "rollout_results.csv"
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        evaluated = [r for r in rows if r["result"] in ("success", "failure") and r["dry_run"] == "False"]
        success = sum(r["result"] == "success" for r in evaluated)
        failure = sum(r["result"] == "failure" for r in evaluated)

        by_condition = {}
        for row in evaluated:
            condition = row["condition"] or "(unspecified)"
            stats = by_condition.setdefault(
                condition,
                {"success": 0, "failure": 0, "evaluated": 0, "success_rate": None},
            )
            stats[row["result"]] += 1
            stats["evaluated"] += 1
        for stats in by_condition.values():
            stats["success_rate"] = stats["success"] / stats["evaluated"]

        summary = {
            "updated_at": _iso_now(),
            "success": success,
            "failure": failure,
            "evaluated": len(evaluated),
            "success_rate": (success / len(evaluated) if evaluated else None),
            "by_condition": by_condition,
            "aborted": sum(r["result"] == "aborted" for r in rows),
            "error": sum(r["result"] == "error" for r in rows),
            "dry_run": sum(r["dry_run"] == "True" for r in rows),
            "total_records": len(rows),
            "csv": str(csv_path),
        }
        with open(self.root / "success_rate_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        return summary


# 三路相机最新帧（RGB, uint8, 480x640x3）。用整帧引用交换避免撕裂，
# 理由同 run_control.py:120 的注释。
img_list = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(3)]
cam_fail_count = np.array([0, 0, 0])
cam_last_frame_monotonic = np.zeros(3, dtype=np.float64)
cam_last_error = ["", "", ""]
# 推理过程中不允许用超过这个时间的旧画面继续控制机械臂。
CAMERA_STALE_TIMEOUT_S = 0.75
# 54 条训练数据中三路画面 >=245 的像素中位数均低于 0.11%，最差一路的
# 第 90 百分位约 1%。8% 留出足够场景变化余量，同时能拦住本次 19%~30% 的过曝。
MAX_START_SATURATED_FRACTION = 0.08
thread_run = True


def run_thread_cam(rs_cam, which_cam):
    """持续读取一路相机。

    注意色彩序：RealSenseCamera.read() 已经返回 RGB。采集端 run_control.py 之所以
    再做 [:, :, ::-1] 变回 BGR，是因为紧接着要 cv2.imwrite（imwrite 吃 BGR），存出的
    JPG 颜色是正确的；转换脚本再 imread+BGR2RGB 得到 RGB 存进 LeRobot。
    所以训练数据是 RGB，这里直接用 read() 的输出、不要再翻通道。
    """
    while thread_run:
        try:
            image, _ = rs_cam.read()
            img_list[which_cam] = np.ascontiguousarray(image)
            cam_last_frame_monotonic[which_cam] = time.monotonic()
            cam_last_error[which_cam] = ""
            cam_fail_count[which_cam] = 0
        except Exception as e:
            cam_fail_count[which_cam] += 1
            cam_last_error[which_cam] = str(e)
            print(f"[CAM {which_cam}] read error "
                  f"(consecutive={cam_fail_count[which_cam]}): {e}; "
                  "will keep reconnecting", flush=True)
            # 保持线程存活，使 USB 重新枚举后可自动恢复。主循环通过时间戳拒绝
            # 使用旧帧，断流期间不会继续向机械臂下发模型动作。
            time.sleep(0.5)


def camera_health():
    """返回三路画面是否新鲜及便于操作者排查的状态文字。"""
    now = time.monotonic()
    stale = []
    names = ("top", "left", "right")
    for i, name in enumerate(names):
        age = now - cam_last_frame_monotonic[i]
        if cam_last_frame_monotonic[i] <= 0 or age > CAMERA_STALE_TIMEOUT_S:
            detail = cam_last_error[i] or f"last frame age={age:.1f}s"
            stale.append(f"{name}: {detail}")
    return not stale, "; ".join(stale)


def camera_exposure_health():
    """Check that the initial scene is not far brighter than the training domain."""
    names = ("top", "left", "right")
    reports = []
    overexposed = []
    for name, image in zip(names, img_list):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        mean = float(gray.mean())
        saturated = float(np.mean(gray >= 245))
        reports.append(f"{name}: mean={mean:.1f}, saturated={saturated * 100:.1f}%")
        if saturated > MAX_START_SATURATED_FRACTION:
            overexposed.append(name)
    detail = "; ".join(reports)
    if overexposed:
        return False, detail + f"; overexposed={','.join(overexposed)}"
    return True, detail


def move_smoothly(env, start, target, step=0.001, max_steps=150):
    """在 start -> target 之间插值下发，避免一步跨太大导致机械臂急停。"""
    max_delta = float(np.abs(np.asarray(target) - np.asarray(start)).max())
    steps = min(int(max_delta / step), max_steps)
    for jnt in np.linspace(start, target, max(steps, 1)):
        env.step(jnt, np.array([1, 1]))


def classify_joint_increment(arm_delta):
    """Classify a model step without changing the action itself."""
    max_increment_deg = float(
        np.rad2deg(np.max(np.abs(np.asarray(arm_delta, dtype=np.float64))))
    )
    if max_increment_deg <= float(np.rad2deg(JOINT_JUMP_WARNING_RAD)):
        return "normal", max_increment_deg
    if max_increment_deg <= AUTO_SMOOTH_JOINT_JUMP_MAX_DEG:
        return "smooth", max_increment_deg
    return "reject", max_increment_deg


class JointTrackingGuard:
    """Stop before the next command on the observed right-J3 contact signature.

    This is a detection-only interlock, calibrated against the available v2
    rollouts. It never clips or changes a policy action. It cannot prove that
    contact has not already caused a controller alarm.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.previous_command = None
        self.previous_observed = None

    def observe(self, command, observed):
        command = np.asarray(command, dtype=np.float64).reshape(-1)
        observed = np.asarray(observed, dtype=np.float64).reshape(-1)
        if command.size != 14 or observed.size != 14:
            raise ValueError("contact guard needs 14-D command and observation")
        if not np.isfinite(command).all() or not np.isfinite(observed).all():
            raise ValueError("non-finite robot joint state or policy command")

        previous_command = self.previous_command
        previous_observed = self.previous_observed
        self.previous_command = command.copy()
        self.previous_observed = observed.copy()
        if previous_command is None:
            return False, {}

        right_j3_deg = float(np.rad2deg(observed[9]))
        right_j4_deg = float(np.rad2deg(observed[10]))
        lag_deg = float(np.rad2deg(previous_command[9] - observed[9]))
        actual_delta_deg = float(np.rad2deg(observed[9] - previous_observed[9]))
        target_delta_deg = float(np.rad2deg(command[9] - previous_command[9]))
        triggered = (
            CONTACT_RIGHT_J3_POSE_DEG[0] <= right_j3_deg <= CONTACT_RIGHT_J3_POSE_DEG[1]
            and CONTACT_RIGHT_J4_POSE_DEG[0] <= right_j4_deg <= CONTACT_RIGHT_J4_POSE_DEG[1]
            and lag_deg > CONTACT_PREVIOUS_TARGET_LAG_DEG
            and actual_delta_deg < CONTACT_ACTUAL_REVERSAL_DEG
            and target_delta_deg > CONTACT_MIN_TARGET_DELTA_DEG
        )
        return triggered, {
            "right_j3_deg": right_j3_deg,
            "right_j4_deg": right_j4_deg,
            "previous_target_lag_deg": lag_deg,
            "actual_j3_delta_deg": actual_delta_deg,
            "target_j3_delta_deg": target_delta_deg,
        }


def is_controller_alarm(exc):
    """Recognize the latched left/right controller alarm sent by robot_node."""
    return (
        isinstance(exc, RobotServerError)
        and ("left robot error!" in str(exc).lower()
             or "right robot error!" in str(exc).lower()
             or "192.168.5.1: error!" in str(exc).lower()
             or "192.168.5.2: error!" in str(exc).lower())
    )


def finish_robot_fault(recorder, exc, step=None):
    """Persist a robot I/O fault as a failed physical attempt exactly once."""
    if not recorder.active:
        return None
    kind = "controller_alarm" if is_controller_alarm(exc) else "robot_io_fault"
    reason = f"{kind}: {type(exc).__name__}: {exc}"
    recorder.add_event(
        "robot_controller_alarm" if kind == "controller_alarm" else "robot_io_fault",
        step=step, kind=kind,
        exception_type=type(exc).__name__, message=str(exc),
    )
    return recorder.finish("failure", reason)


def controllers_ready_for_recovery(statuses):
    """Require fresh, parseable, alarm-free ENABLED_IDLE for both arms."""
    def valid_empty_error_reply(response):
        # parse_error_ids() intentionally returns [] on malformed input too;
        # that is insufficient to authorize motion after a controller fault.
        match = re.fullmatch(
            r"\s*0,\{(.+)\},GetErrorID\(\);?\s*", response,
        )
        if match is None:
            return False
        try:
            payload = ast.literal_eval(match.group(1))
        except (SyntaxError, ValueError):
            return False

        def valid_tree(value):
            if isinstance(value, bool):
                return False
            if isinstance(value, int):
                return value == 0
            if isinstance(value, (list, tuple)):
                return all(valid_tree(item) for item in value)
            return False

        return isinstance(payload, (list, tuple)) and valid_tree(payload)

    return (
        len(statuses) == 2
        and {status.name for status in statuses} == {"left", "right"}
        and all(
            status.mode == 5 and not status.error_ids
            and status.mode_response.startswith("0,")
            and valid_empty_error_reply(status.error_response)
            for status in statuses
        )
    )


def read_controller_statuses(robot_client):
    """Query through launch_nodes' existing dashboard sockets (29999 is single-client)."""
    raw = robot_client.get_controller_statuses()
    if not isinstance(raw, dict) or set(raw) != {"left", "right"}:
        raise ValueError("robot server returned incomplete controller statuses")
    statuses = []
    for name in ("left", "right"):
        item = raw[name]
        if not isinstance(item, dict):
            raise ValueError(f"{name} controller status is malformed")
        mode_reply = item.get("mode_response")
        error_reply = item.get("error_response")
        if not isinstance(mode_reply, str) or not isinstance(error_reply, str):
            raise ValueError(f"{name} controller replies are malformed")
        statuses.append(ControllerStatus(
            name=name, ip=CONTROLLERS[name],
            mode=parse_robot_mode(mode_reply),
            error_ids=parse_error_ids(error_reply),
            mode_response=mode_reply, error_response=error_reply,
        ))
    return statuses


def finalize_recorded_rollout(recorder, env, result, reason="", idle_light=True,
                              step=None, robot_fault=None):
    """Finalize once, converting any robot I/O uncertainty into a failure."""
    if result == "success" and robot_fault is None:
        try:
            env.get_obs()
        except ROBOT_IO_EXCEPTIONS as exc:
            robot_fault = exc
    if idle_light and robot_fault is None:
        try:
            set_idle_light(env)
        except Exception as exc:
            robot_fault = exc
    if robot_fault is not None:
        return finish_robot_fault(recorder, robot_fault, step=step), robot_fault
    return recorder.finish(result, reason), None


def _dh_transform(theta, d, a, alpha):
    """Return one standard DH transform for the offline Nova2 FK check."""
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    cos_alpha, sin_alpha = np.cos(alpha), np.sin(alpha)
    return np.array([
        [cos_theta, -sin_theta * cos_alpha, sin_theta * sin_alpha,
         a * cos_theta],
        [sin_theta, cos_theta * cos_alpha, -cos_theta * sin_alpha,
         a * sin_theta],
        [0.0, sin_alpha, cos_alpha, d],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)


def nova2_tcp_position_mm(joint_positions):
    """Compute a Nova2 TCP target offline using the configured 197 mm tool."""
    joints = np.asarray(joint_positions, dtype=np.float64).reshape(-1)
    if joints.size < 6:
        raise ValueError(f"Nova2 FK needs 6 joints, got {joints.size}")
    q0, q1, q2, q3, q4, q5 = joints[:6]
    dh_params = (
        (q0, 0.2234, 0.0, np.pi / 2),
        (q1 - np.pi / 2, 0.0, -0.280, 0.0),
        (q2, 0.0, -0.225, 0.0),
        (q3 - np.pi / 2, 0.1175, 0.0, np.pi / 2),
        (q4, 0.120, 0.0, -np.pi / 2),
        (q5, 0.088, 0.0, 0.0),
    )
    transform = np.eye(4, dtype=np.float64)
    for params in dh_params:
        transform = transform @ _dh_transform(*params)
    tool = np.eye(4, dtype=np.float64)
    tool[2, 3] = NOVA2_TOOL_Z_MM / 1000.0
    return (transform @ tool)[:3, 3] * 1000.0


def is_left_pickup_position(position):
    """Return whether a TCP position is inside the bounded low-Z pickup box."""
    pos = np.asarray(position, dtype=np.float64).reshape(-1)
    if pos.size < 3 or not np.all(np.isfinite(pos[:3])):
        return False
    return (
        LEFT_PICKUP_X_RANGE_MM[0] < pos[0] < LEFT_PICKUP_X_RANGE_MM[1]
        and LEFT_PICKUP_Y_RANGE_MM[0] < pos[1] < LEFT_PICKUP_Y_RANGE_MM[1]
        and LEFT_PICKUP_Z_FLOOR_MM < pos[2] <= WORKSPACE_Z_MIN_MM
    )


def classify_left_target_position(target_position):
    """Reject only non-finite or unsafe low-Z targets from the offline FK."""
    target = np.asarray(target_position, dtype=np.float64).reshape(-1)
    if target.size < 3 or not np.all(np.isfinite(target[:3])):
        return "hard_limit"
    if target[2] > WORKSPACE_Z_MIN_MM:
        # FK is used only to close the one-step low-Z gap. Approximation error
        # must not add a new X/Y restriction to the established high path.
        return "safe"
    return "left_pickup_zone" if is_left_pickup_position(target) else "hard_limit"


def classify_left_action_target(action):
    """Safely classify a pending target without querying or moving the robot."""
    try:
        action_array = np.asarray(action, dtype=np.float64).reshape(-1)
        if action_array.size < 6:
            raise ValueError(
                f"left action target needs at least 6 values, got {action_array.size}"
            )
        target = nova2_tcp_position_mm(action_array[:6])
    except (TypeError, ValueError, IndexError, FloatingPointError, OverflowError):
        return "hard_limit", np.full(3, np.nan, dtype=np.float64)
    if not np.all(np.isfinite(target)):
        return "hard_limit", target
    return classify_left_target_position(target), target


def left_pickup_zone_timed_out(entered_at, now):
    """Pure duration check used by the runtime and deterministic unit tests."""
    return (
        entered_at is not None
        and now - entered_at > LEFT_PICKUP_MAX_DURATION_S
    )


def classify_workspace_position(position):
    """Distinguish the bounded left pickup/right insertion task zones.

    The global bounds match the original safety check exactly. The left-arm
    exception additionally requires the measured TCP to remain inside the
    data-supported pickup box. Simultaneous low-Z violations remain hard.
    """
    pos = np.asarray(position, dtype=np.float64).reshape(-1)
    if pos.size < 9:
        raise ValueError(f"workspace position needs at least 9 values, got {pos.size}")

    checks = (
        ("left_x", -410 < pos[0] < 300),
        ("left_y", -700 < pos[1] < -210),
        ("left_z", pos[2] > WORKSPACE_Z_MIN_MM),
        ("right_x", -250 < pos[6] < 410),
        ("right_y", -700 < pos[7] < -210),
        ("right_z", pos[8] > WORKSPACE_Z_MIN_MM),
    )
    reasons = [name for name, valid in checks if not valid]
    if not reasons:
        return "safe", reasons
    if reasons == ["left_z"] and is_left_pickup_position(pos[:3]):
        return "left_pickup_zone", reasons
    if (reasons == ["right_z"]
            and RIGHT_INSERTION_Z_FLOOR_MM < pos[8] <= WORKSPACE_Z_MIN_MM):
        return "right_insertion_zone", reasons
    return "hard_limit", reasons


def evaluate_action_safety(env, action, check_workspace=True):
    """与原厂 run_inference.py 完全一致的安全边界，阈值未改动。

    附带 workspace 分类，使主循环只自动通过有界的任务低位。
    """
    violations = []
    workspace_detail = None

    # 左臂 -150<J3<0, J4>-35（度）；右臂 150>J3>0, J4<35
    if not ((-2.6 < action[2] < 0 and action[3] > -0.6)
            and (0 < action[9] < 2.6 and action[10] < 0.6)):
        print("[Warn]:The J3 or J4 joints of the robotic arm are out of the safe position!")
        print(action)
        violations.append("joint")

    # 夹爪指尖 XYZ 工作空间限制（mm）
    if check_workspace:
        pos = np.asarray(env.get_XYZrxryrz_state(), dtype=np.float64)
        classification, reasons = classify_workspace_position(pos)
        workspace_detail = {
            "classification": classification,
            "reasons": reasons,
            "position": pos.tolist(),
        }
        if classification != "safe":
            if classification == "hard_limit":
                print("[Warn]:The robot arm XYZ is out of the safe position!")
                print(pos)
            violations.append("workspace")

    return violations, workspace_detail


def check_action_safety(env, action, check_workspace=True):
    """Backward-compatible wrapper returning only the violation names."""
    violations, _ = evaluate_action_safety(env, action, check_workspace)
    return violations


def set_alarm_light(env):
    """Only change the stack light; this is not a controller emergency stop."""
    env.set_do_status([3, 0])  # 黄灯灭
    env.set_do_status([2, 0])  # 绿灯灭
    env.set_do_status([1, 1])  # 红灯亮


def build_observation(state, prompt):
    """组装服务端期望的 observation。

    键名必须和 LeRobotXTrainerDataConfig 的 repack 映射「之后」的名字一致：
    serve_policy.py 不带 repack_transforms，所以 DataConfig 里那层 RepackTransform
    只在训练读盘时生效，推理时得由客户端直接发 observation/xxx 这套键名。

    图像在这里就 resize 到 224x224：服务端的 ResizeImages 用的是同一个
    resize_with_pad，且该函数对已是目标尺寸的输入直接返回，所以数值等价，
    但每步上行流量从约 2.7MB 降到约 0.3MB。
    """
    return {
        "observation/state": state,
        "observation/top_image": image_tools.resize_with_pad(img_list[0], 224, 224),
        "observation/left_image": image_tools.resize_with_pad(img_list[1], 224, 224),
        "observation/right_image": image_tools.resize_with_pad(img_list[2], 224, 224),
        "prompt": prompt,
    }


def move_to_initial_pose(env):
    """经安全位回到与训练采集一致的初始位。"""
    safe_left = np.deg2rad([-90, 30, -110, 20, 90, 90, 0])
    safe_right = np.deg2rad([90, -30, 110, -20, -90, -90, 0])
    move_smoothly(
        env,
        env.get_obs()["joint_positions"],
        np.concatenate([safe_left, safe_right]),
    )
    time.sleep(1)
    move_smoothly(
        env,
        env.get_obs()["joint_positions"],
        TRAINING_INITIAL_TARGET,
    )


def set_idle_light(env):
    # Stack-light wiring: DO1=red, DO2=orange/yellow, DO3=green.
    env.set_do_status([1, 0])
    env.set_do_status([3, 0])
    env.set_do_status([2, 1])


def set_running_light(env):
    env.set_do_status([1, 0])
    env.set_do_status([2, 0])
    env.set_do_status([3, 1])


def main(args):
    global thread_run
    if args.open_loop_horizon < 1:
        raise ValueError("--open-loop-horizon must be at least 1")
    if args.action_blend_steps < 0:
        raise ValueError("--action-blend-steps cannot be negative")
    if args.temporal_ensemble_chunks < 1:
        raise ValueError("--temporal-ensemble-chunks must be at least 1")
    if args.temporal_ensemble_decay < 0:
        raise ValueError("--temporal-ensemble-decay cannot be negative")
    if args.inference_timeout_s <= 0:
        raise ValueError("--inference-timeout-s must be positive")
    if args.initial_inference_timeout_s <= 0:
        raise ValueError("--initial-inference-timeout-s must be positive")
    if args.camera_fps <= 0:
        raise ValueError("--camera-fps must be positive")
    if args.camera_exposure_us < 0:
        raise ValueError("--camera-exposure-us cannot be negative")
    if args.camera_exposure_us > 0 and args.camera_gain < 0:
        raise ValueError("--camera-gain cannot be negative")

    manual_exposure_us = (
        args.camera_exposure_us if args.camera_exposure_us > 0 else None
    )
    manual_gain = args.camera_gain if manual_exposure_us is not None else None

    thread_run = True
    env = None
    robot_client = None
    recorder = None
    emergency = False

    # 先验证服务端，连接失败时不启动相机，也不接管机械臂。
    client = websocket_client_policy.WebsocketClientPolicy(
        host=args.remote_host,
        port=args.remote_port,
    )
    server_metadata = client.get_server_metadata()
    print(f"policy server metadata: {server_metadata}")
    policy_runner = (
        async_policy.AsyncPolicyRunner(client) if args.async_inference else None
    )
    action_ensemble = async_policy.TemporalActionEnsembler(
        max_chunks=args.temporal_ensemble_chunks,
        decay=args.temporal_ensemble_decay,
    )

    # 分辨率与翻转方向和采集脚本一致；帧率单独降低以保证三路长期稳定。
    camera_dict = load_ini_data_camera()
    cams = []
    try:
        cams.append(RealSenseCamera(
            flip=True, device_id=camera_dict["top"], fps=args.camera_fps,
            allow_hardware_reset=False,
            exposure_us=manual_exposure_us, gain=manual_gain,
        ))
        cams.append(RealSenseCamera(
            flip=False, device_id=camera_dict["left"], fps=args.camera_fps,
            allow_hardware_reset=False,
            exposure_us=manual_exposure_us, gain=manual_gain,
        ))
        cams.append(RealSenseCamera(
            flip=True, device_id=camera_dict["right"], fps=args.camera_fps,
            allow_hardware_reset=False,
            exposure_us=manual_exposure_us, gain=manual_gain,
        ))
    except BaseException:
        # 第三路初始化失败时也立即释放已经启动的前两路，否则同一终端重试时
        # 它们可能继续占用 UVC 接口，造成误判为硬件故障。
        for camera in cams:
            camera.close()
        raise
    threads = [
        threading.Thread(target=run_thread_cam, args=(camera, i), daemon=True)
        for i, camera in enumerate(cams)
    ]
    for thread in threads:
        thread.start()
    time.sleep(2)
    print("camera thread init success...")

    robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)
    env = RobotEnv(robot_client)
    set_idle_light(env)
    print("robot init success (orange light = idle)...")

    recorder = RolloutRecorder(args, server_metadata)
    print(f"rollout root: {recorder.root}")

    show_canvas = np.zeros((480, 640 * 3, 3), dtype=np.uint8)
    running = False
    quit_requested = False
    state = None
    last_action = None
    first = True
    action_chunk = None
    chunk_idx = 0
    total_steps = 0
    infer_latency_ms = None
    ignore_workspace_safety = False
    active_left_pickup_zone = False
    left_pickup_zone_entered_at = None
    rollout_generation = 0
    pending_request_id = None
    pending_request_started = None
    next_replan_step = 0
    contact_guard = JointTrackingGuard()
    faulted = False
    needs_manual_reset = False

    def finish_rollout(result, reason="", idle_light=True, robot_fault=None):
        nonlocal running, action_chunk, chunk_idx, rollout_generation
        nonlocal pending_request_id, pending_request_started
        nonlocal faulted, needs_manual_reset, emergency
        # Stop the control/record loop and change the light immediately when the
        # operator marks the result. recorder.finish() may still need a moment
        # to flush already queued frames to disk, but no new frame is accepted.
        running = False
        action_chunk = None
        chunk_idx = 0
        action_ensemble.clear()
        # Any result that arrives after the operator ends this rollout belongs
        # to the previous generation and must never control the next rollout.
        rollout_generation += 1
        pending_request_id = None
        pending_request_started = None
        meta, robot_fault = finalize_recorded_rollout(
            recorder, env, result, reason, idle_light, total_steps, robot_fault,
        )
        if robot_fault is not None:
            result = "failure"
            faulted = True
            needs_manual_reset = True
            emergency = True
            print(f"[ROBOT FAULT] {type(robot_fault).__name__}: {robot_fault}",
                  flush=True)
            print_fault_guidance()
        if meta is not None:
            print(
                f"[RESULT] {result.upper()} | steps={meta['num_steps']} | "
                f"duration={meta['duration_s']}s | {recorder.rollout_dir}",
                flush=True,
            )
            print(f"[TABLE] {recorder.root / 'rollout_results.csv'}", flush=True)
            if result in ("success", "failure") and not args.dry_run:
                summary = recorder.summary
                rate = summary["success_rate"] * 100
                print(
                    f"[成功率] {summary['success']}/{summary['evaluated']}，"
                    f"成功率 {rate:.1f}%",
                    flush=True,
                )

    def print_ready():
        if faulted:
            print("[HOLD] Clear contact/alarm, then c CHECK, h RESET.", flush=True)
        else:
            print("[READY] h RESET, then r START the next rollout.", flush=True)

    def print_fault_guidance():
        print(
            "[HOLD] No further motion commands will be sent. Remove the physical "
            "interference first. Check controller alarms with "
            "python scripts/check_dobot_alarm.py; clear only after inspection. "
            "A latched launch_nodes.py alarm requires restarting that server. "
            "Then press c for a read-only health check, h to reset, or q to quit. "
            "Use the physical emergency-stop button if motion or force persists.",
            flush=True,
        )

    def enter_robot_fault(exc):
        """Fail closed after robot I/O failed; never retry the same command."""
        nonlocal faulted, needs_manual_reset, emergency
        faulted = True
        needs_manual_reset = True
        emergency = True  # A controller alarm also rejects stack-light RPCs.
        if recorder.active:
            finish_rollout("failure", idle_light=False, robot_fault=exc)
        else:
            print(f"[ROBOT FAULT] {type(exc).__name__}: {exc}", flush=True)
            print_fault_guidance()

    def check_recovery():
        """Read both controller alarms and robot state; never clear/enable/move."""
        nonlocal faulted, emergency
        try:
            statuses = read_controller_statuses(robot_client)
            for status in statuses:
                print("[RECOVERY] " + format_status(status), flush=True)
            if not controllers_ready_for_recovery(statuses):
                print("[RECOVERY] Both arms must report ENABLED_IDLE with no alarms.",
                      flush=True)
                return
            env.get_obs()
        except Exception as exc:
            print(f"[RECOVERY] Robot still unavailable: {exc}", flush=True)
            return
        faulted = False
        emergency = False
        print("[RECOVERY] Controller and robot server checks passed. This cannot prove "
              "the plugs are separated; press h only after clearing the scene.",
              flush=True)

    def reset_robot():
        """Return to the standard initial pose without creating a rollout."""
        nonlocal needs_manual_reset
        set_idle_light(env)
        if args.dry_run:
            if policy_runner is None or not policy_runner.is_busy():
                client.reset()
            print("[RESET] dry-run policy state reset; robot was not moved.", flush=True)
            return
        print("[RESET] Returning to the standard initial pose...", flush=True)
        move_to_initial_pose(env)
        if policy_runner is None or not policy_runner.is_busy():
            client.reset()
        needs_manual_reset = False
        print("[RESET] Complete (orange light = idle).", flush=True)

    def start_rollout():
        nonlocal running, state, last_action, first
        nonlocal action_chunk, chunk_idx, total_steps, ignore_workspace_safety
        nonlocal active_left_pickup_zone, left_pickup_zone_entered_at
        nonlocal infer_latency_ms, rollout_generation
        nonlocal pending_request_id, pending_request_started, next_replan_step
        if needs_manual_reset:
            print("[WAIT] Previous contact/alarm requires an explicit h RESET first.",
                  flush=True)
            return
        if policy_runner is not None:
            # A result from a just-finished rollout normally completes before
            # the operator starts again. Refuse to overlap websocket calls if
            # it is still outstanding.
            if not policy_runner.wait_until_idle(args.inference_timeout_s):
                print("[WAIT] previous policy request is still running; press r again shortly.",
                      flush=True)
                return
            policy_runner.discard_completed()
        healthy, detail = camera_health()
        if not healthy:
            print(f"[WAIT] rollout not started; camera unavailable: {detail}", flush=True)
            return
        exposure_ok, exposure_detail = camera_exposure_health()
        print(f"[CAM EXPOSURE] {exposure_detail}", flush=True)
        if not exposure_ok:
            print(
                "[WAIT] rollout not started; images are overexposed relative to training data.",
                flush=True,
            )
            return
        if args.reset_before_rollout and not args.dry_run:
            reset_robot()
            healthy, detail = camera_health()
            if not healthy:
                print(f"[WAIT] rollout not started after reset; camera unavailable: {detail}",
                      flush=True)
                return
            exposure_ok, exposure_detail = camera_exposure_health()
            print(f"[CAM EXPOSURE] after reset: {exposure_detail}", flush=True)
            if not exposure_ok:
                print("[WAIT] rollout not started after reset; images are overexposed.",
                      flush=True)
                return
        obs = env.get_obs()
        obs["joint_positions"][6] = 1.0
        obs["joint_positions"][13] = 1.0
        state = obs["joint_positions"].astype(np.float32)
        last_action = state.copy()
        first = True
        action_chunk = None
        chunk_idx = 0
        action_ensemble.clear()
        total_steps = 0
        ignore_workspace_safety = False
        active_left_pickup_zone = False
        left_pickup_zone_entered_at = None
        contact_guard.reset()
        pending_request_id = None
        pending_request_started = None
        rollout_generation += 1
        # reset_robot() already resets policy state. When automatic robot reset
        # is disabled (or in dry-run), reset the policy explicitly here.
        if not args.reset_before_rollout or args.dry_run:
            client.reset()

        if policy_runner is not None:
            print("[INFER] Preparing initial action chunk while robot is stationary...",
                  flush=True)
            try:
                request_id = policy_runner.submit(
                    build_observation(state.copy(), args.prompt),
                    generation=rollout_generation,
                    request_step=0,
                )
                if request_id is None:
                    raise RuntimeError("policy worker unexpectedly busy before rollout")
                result = policy_runner.wait(
                    request_id,
                    timeout_s=args.initial_inference_timeout_s,
                )
                if result.error is not None:
                    raise RuntimeError(f"initial policy inference failed: {result.error}")
                action_chunk, chunk_idx = async_policy.prepare_action_chunk(
                    result.output["actions"],
                    steps_elapsed=0,
                    last_action=last_action,
                    blend_steps=0,
                )
                action_ensemble.add(action_chunk, request_step=0)
                infer_latency_ms = result.latency_ms
                next_replan_step = max(1, args.open_loop_horizon)
                print(
                    f"[infer:init] {infer_latency_ms:.0f}ms  chunk={action_chunk.shape}",
                    flush=True,
                )
            except Exception as exc:
                rollout_generation += 1
                set_idle_light(env)
                print(f"[INFER] rollout not started: {exc}", flush=True)
                return

            healthy, detail = camera_health()
            if not healthy:
                rollout_generation += 1
                set_idle_light(env)
                print(f"[WAIT] rollout not started after initial inference: {detail}",
                      flush=True)
                return
        path = recorder.start()
        running = True
        set_running_light(env)
        print(f"[START] rollout recording (green light): {path}", flush=True)

    def wait_for_safety_key(keys, message, accepted_keys):
        """安全提示期间同时接收终端和 OpenCV 预览窗口按键。"""
        print(message, flush=True)
        while True:
            terminal_key = keys.poll()
            cv_key = None
            if args.show_img:
                cv2.rectangle(show_canvas, (0, 0), (1920, 42), (0, 0, 0), -1)
                cv2.putText(
                    show_canvas,
                    message,
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 80, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("openpi", show_canvas)
                cv_key = cv2.waitKey(20)
            key = select_key_input(
                terminal_key,
                cv_key,
                accepted_keys=accepted_keys,
            )
            if key is not None:
                return key
            time.sleep(0.02)

    print("\n-------------- X-Trainer real-robot evaluation --------------")
    print("r: start + record     s: success     f: failure")
    print("h: abort if running + reset/home     q: finish program")
    print("e: stop policy commands + exit; use the physical E-stop for immediate stop")
    print("green = running/recording; orange = idle/recording stopped")
    print(
        "policy control: "
        + ("asynchronous closed-loop" if args.async_inference else "synchronous fallback")
    )
    if args.interactive_safety_prompts:
        print("safety handling: interactive prompts (y joint smoothing / i XYZ override)")
    else:
        print(
            "safety handling: automatic smoothing in the validated range; "
            "unexpected limits stop the current rollout"
        )
    print("q during a rollout records ABORTED and does not enter the success rate.")
    print("contact guard: " + ("on (v2 right-J3 reversal pattern)" if args.contact_guard
                                else "off"))
    print("A robot alarm or contact warning holds motion until c CHECK and h RESET.")
    print("Press r when the scene and robot are ready.\n")

    try:
        with TerminalKeys() as keys:
            while not quit_requested:
                for i, thread in enumerate(threads):
                    if not thread.is_alive():
                        raise RuntimeError(f"camera thread {i} died")

                terminal_key = keys.poll()
                cv_key = None
                if args.show_img:
                    show_canvas[:, :640] = img_list[0][:, :, ::-1]
                    show_canvas[:, 640:1280] = img_list[1][:, :, ::-1]
                    show_canvas[:, 1280:1920] = img_list[2][:, :, ::-1]
                    if faulted:
                        status_text = "ROBOT HOLD  |  clear contact/alarm  c CHECK  h RESET  q QUIT"
                        status_color = (0, 0, 255)
                    elif running:
                        status_text = (
                            f"RUNNING  step={total_steps}  |  "
                            "s SUCCESS  f FAILURE  h ABORT+RESET  q QUIT"
                        )
                        status_color = (40, 220, 40)
                    else:
                        summary = recorder.summary
                        rate = (
                            summary["success_rate"] * 100
                            if summary["success_rate"] is not None
                            else 0.0
                        )
                        status_text = (
                            "IDLE (ORANGE)  |  r START  h RESET  q QUIT  |  "
                            f"SUCCESS {summary['success']}/{summary['evaluated']} "
                            f"({rate:.1f}%)"
                        )
                        status_color = (0, 220, 255)
                    cv2.rectangle(show_canvas, (0, 0), (1920, 42), (0, 0, 0), -1)
                    cv2.putText(
                        show_canvas,
                        status_text,
                        (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        status_color,
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow("openpi", show_canvas)
                    cv_key = cv2.waitKey(1)
                key = select_key_input(
                    terminal_key,
                    cv_key,
                    accepted_keys={"r", "s", "f", "h", "q", "e", "c"},
                )

                if faulted:
                    if key == "c":
                        check_recovery()
                    elif key in ("q", "e"):
                        quit_requested = True
                    elif key in ("r", "h"):
                        print("[HOLD] Clear the scene and alarm, then press c CHECK first.",
                              flush=True)
                    else:
                        time.sleep(0.02)
                    continue

                if not running:
                    if key == "r":
                        try:
                            start_rollout()
                        except ROBOT_IO_EXCEPTIONS as exc:
                            enter_robot_fault(exc)
                    elif key == "h":
                        try:
                            reset_robot()
                        except ROBOT_IO_EXCEPTIONS as exc:
                            enter_robot_fault(exc)
                    elif key == "q":
                        quit_requested = True
                    elif key == "e":
                        emergency = True
                        try:
                            set_alarm_light(env)
                        except Exception as light_exc:
                            print(f"[ALARM LIGHT FAILED] {light_exc}", flush=True)
                        print("[STOP] Policy commands ended. Use the physical E-stop if needed.",
                              flush=True)
                        quit_requested = True
                    else:
                        time.sleep(0.02)
                    continue

                if key == "s":
                    finish_rollout("success")
                    print_ready()
                    continue
                if key == "f":
                    finish_rollout("failure", "operator_marked_failure")
                    print_ready()
                    continue
                if key == "h":
                    finish_rollout("aborted", "operator_reset")
                    if not faulted:
                        try:
                            reset_robot()
                        except ROBOT_IO_EXCEPTIONS as exc:
                            enter_robot_fault(exc)
                    continue
                if key == "q":
                    finish_rollout("aborted", "operator_quit")
                    quit_requested = True
                    continue
                if key == "e":
                    emergency = True
                    finish_rollout("failure", "operator_stop", idle_light=False)
                    try:
                        set_alarm_light(env)
                    except Exception as light_exc:
                        print(f"[ALARM LIGHT FAILED] {light_exc}", flush=True)
                    print("[STOP] Policy commands ended. Use the physical E-stop if needed.",
                          flush=True)
                    quit_requested = True
                    continue

                healthy, detail = camera_health()
                if not healthy:
                    print(f"[CAMERA] current rollout stopped; stale image: {detail}", flush=True)
                    finish_rollout("failure", "camera_stale")
                    print_ready()
                    continue

                tic = time.monotonic()
                ensemble_sources = 1
                if policy_runner is not None:
                    policy_error = None
                    for result in policy_runner.poll():
                        # Discard completed requests from an earlier rollout or
                        # an already superseded request.
                        if (result.generation != rollout_generation
                                or result.request_id != pending_request_id):
                            continue
                        pending_request_id = None
                        pending_request_started = None
                        if result.error is not None:
                            policy_error = result.error
                            break
                        try:
                            elapsed_steps = max(0, total_steps - result.request_step)
                            action_chunk, chunk_idx = async_policy.prepare_action_chunk(
                                result.output["actions"],
                                steps_elapsed=elapsed_steps,
                                last_action=last_action,
                                blend_steps=args.action_blend_steps,
                            )
                            action_ensemble.add(
                                action_chunk,
                                request_step=result.request_step,
                            )
                            infer_latency_ms = result.latency_ms
                            print(
                                f"[infer:async] {infer_latency_ms:.0f}ms  "
                                f"skip={chunk_idx}  chunk={action_chunk.shape}",
                                flush=True,
                            )
                        except Exception as exc:
                            policy_error = exc
                            break

                    if policy_error is not None:
                        print(f"[INFER] background request failed: {policy_error}", flush=True)
                        finish_rollout("failure", "policy_inference_error")
                        print_ready()
                        continue

                    if (pending_request_id is not None
                            and pending_request_started is not None
                            and time.monotonic() - pending_request_started
                            > args.inference_timeout_s):
                        print(
                            f"[INFER] request timeout > {args.inference_timeout_s:.1f}s; "
                            "stopping this rollout.",
                            flush=True,
                        )
                        finish_rollout("failure", "policy_inference_timeout")
                        print_ready()
                        continue

                    if (pending_request_id is None
                            and not policy_runner.is_busy()
                            and total_steps >= next_replan_step):
                        pending_request_id = policy_runner.submit(
                            build_observation(state.copy(), args.prompt),
                            generation=rollout_generation,
                            request_step=total_steps,
                        )
                        if pending_request_id is not None:
                            pending_request_started = time.monotonic()
                            next_replan_step = (
                                total_steps + max(1, args.open_loop_horizon)
                            )

                    try:
                        action, action_index, ensemble_sources = (
                            action_ensemble.action_for_step(total_steps)
                        )
                    except IndexError:
                        print("[INFER] action buffer exhausted before a new chunk arrived.",
                              flush=True)
                        finish_rollout("failure", "action_buffer_exhausted")
                        print_ready()
                        continue
                elif action_chunk is None or chunk_idx >= min(
                    args.open_loop_horizon, len(action_chunk)
                ):
                    infer_start = time.time()
                    result = client.infer(build_observation(state, args.prompt))
                    infer_latency_ms = (time.time() - infer_start) * 1000
                    action_chunk = np.asarray(result["actions"], dtype=np.float64)
                    chunk_idx = 0
                    print(
                        f"[infer:sync] {infer_latency_ms:.0f}ms  chunk={action_chunk.shape}",
                        flush=True,
                    )
                    if action_chunk.ndim != 2 or action_chunk.shape[1] != 14:
                        raise RuntimeError(
                            f"unexpected action shape {action_chunk.shape}, want (H, 14)"
                        )
                    if len(action_chunk) == 0:
                        raise RuntimeError("policy returned an empty action chunk")

                if policy_runner is None:
                    action_index = chunk_idx
                    action = action_chunk[action_index].copy()
                    chunk_idx += 1
                action[6] = np.clip(action[6], 0.0, 1.0)
                action[13] = np.clip(action[13], 0.0, 1.0)

                if args.dry_run:
                    recorder.record(
                        state=state,
                        action=action,
                        infer_latency_ms=infer_latency_ms,
                        chunk_index=action_index,
                        images=tuple(img_list),
                        ensemble_sources=ensemble_sources,
                    )
                    print(
                        f"[dry] t={total_steps} action={np.round(action, 4)}",
                        flush=True,
                    )
                    state = action.astype(np.float32)
                    last_action = action.copy()
                    total_steps += 1
                    if total_steps >= args.max_steps:
                        finish_rollout("aborted", "dry_run_max_steps")
                        continue
                    time.sleep(CONTROL_PERIOD_S)
                    continue

                # 对两侧六个机械臂关节取绝对增量，夹爪维不参与角度判断。
                arm_delta = np.concatenate(
                    (action[0:6] - last_action[0:6], action[7:13] - last_action[7:13])
                )
                requires_smooth_move = False
                joint_step_class, max_increment_deg = classify_joint_increment(
                    arm_delta
                )
                if joint_step_class != "normal":
                    print(
                        "Note! Joint increment larger than 10 degrees:",
                        np.round(arm_delta, 4),
                    )
                    if args.interactive_safety_prompts:
                        decision = wait_for_safety_key(
                            keys,
                            "JOINT JUMP >10deg | y CONTINUE | f FAIL",
                            accepted_keys={"y", "f"},
                        )
                    else:
                        decision = (
                            "y" if joint_step_class == "smooth" else "f"
                        )
                    if decision == "y":
                        requires_smooth_move = True
                        recorder.add_event(
                            "joint_increment_override",
                            step=total_steps,
                            max_increment_deg=max_increment_deg,
                            decision_source=(
                                "interactive" if args.interactive_safety_prompts
                                else "automatic"
                            ),
                        )
                        if args.interactive_safety_prompts:
                            print(
                                "[OVERRIDE] y accepted; executing the existing "
                                "smooth joint move.",
                                flush=True,
                            )
                        else:
                            print(
                                "[SMOOTH] Joint transition accepted in the validated "
                                f"range ({max_increment_deg:.2f}deg); executing the "
                                "existing smooth move.",
                                flush=True,
                            )
                    else:
                        recorder.add_event(
                            "joint_increment_rejected",
                            step=total_steps,
                            max_increment_deg=max_increment_deg,
                            decision_source=(
                                "interactive" if args.interactive_safety_prompts
                                else "automatic"
                            ),
                        )
                        if not args.interactive_safety_prompts:
                            print(
                                "[SAFETY STOP] Joint transition exceeds the "
                                f"validated {AUTO_SMOOTH_JOINT_JUMP_MAX_DEG:.0f}deg "
                                "automatic-smoothing range.",
                                flush=True,
                            )
                        finish_rollout(
                            "failure", "joint_increment_rejected"
                        )
                        print_ready()
                        continue

                left_target_classification = None
                left_target_position = None
                if (not args.interactive_safety_prompts
                        and not ignore_workspace_safety):
                    left_target_classification, left_target_position = (
                        classify_left_action_target(action)
                    )
                    if left_target_classification == "hard_limit":
                        recorder.add_event(
                            "left_action_target_rejected",
                            step=total_steps,
                            action=np.asarray(action).tolist(),
                            target_position_mm=left_target_position.tolist(),
                        )
                        print(
                            "[SAFETY STOP] Pending left-arm low-Z TCP target is "
                            "outside the bounded pickup zone: "
                            + np.array2string(
                                left_target_position, precision=3
                            ),
                            flush=True,
                        )
                        finish_rollout(
                            "failure", "left_target_safety_boundary"
                        )
                        print_ready()
                        continue

                try:
                    violations, workspace_detail = evaluate_action_safety(
                        env,
                        action,
                        check_workspace=not ignore_workspace_safety,
                    )
                except ROBOT_IO_EXCEPTIONS as exc:
                    enter_robot_fault(exc)
                    continue
                if "joint" in violations:
                    # The rejected action has not been sent to the robot, so end
                    # this rollout and return to idle instead of killing the
                    # whole evaluation process. The operator can press h to
                    # reset and then start the next rollout.
                    finish_rollout("failure", "joint_safety_boundary")
                    print_ready()
                    continue
                if "workspace" in violations:
                    classification = workspace_detail["classification"]
                    automatic_left_pickup = (
                        not args.interactive_safety_prompts
                        and classification == "left_pickup_zone"
                    )
                    if automatic_left_pickup:
                        now = time.monotonic()
                        if not active_left_pickup_zone:
                            active_left_pickup_zone = True
                            left_pickup_zone_entered_at = now
                            recorder.add_event(
                                "workspace_safety_override",
                                step=total_steps,
                                action=np.asarray(action).tolist(),
                                workspace=workspace_detail,
                                decision_source="automatic",
                                scope="bounded_left_pickup_zone",
                                target_classification=left_target_classification,
                                target_position_mm=left_target_position.tolist(),
                            )
                            print(
                                "[TASK ZONE] Expected left-arm pickup position "
                                "accepted; bounded XYZ checks remain active.",
                                flush=True,
                            )
                        elif left_pickup_zone_timed_out(
                                left_pickup_zone_entered_at, now):
                            recorder.add_event(
                                "workspace_safety_rejected",
                                step=total_steps,
                                action=np.asarray(action).tolist(),
                                workspace=workspace_detail,
                                decision_source="automatic",
                                reason="left_pickup_zone_timeout",
                                duration_s=now - left_pickup_zone_entered_at,
                            )
                            print(
                                "[SAFETY STOP] Left arm remained in the low-Z "
                                f"pickup zone for more than "
                                f"{LEFT_PICKUP_MAX_DURATION_S:.1f}s.",
                                flush=True,
                            )
                            finish_rollout(
                                "failure", "left_pickup_zone_timeout"
                            )
                            print_ready()
                            continue
                    else:
                        active_left_pickup_zone = False
                        left_pickup_zone_entered_at = None
                        if args.interactive_safety_prompts:
                            decision = wait_for_safety_key(
                                keys,
                                "XYZ LIMIT | i IGNORE THIS ROLLOUT | f FAIL",
                                accepted_keys={"i", "f"},
                            )
                        else:
                            decision = (
                                "i"
                                if classification == "right_insertion_zone"
                                else "f"
                            )
                        if decision == "i":
                            # Keep the established right-zone and explicit-i
                            # semantics: later XYZ RPCs are skipped for this
                            # rollout. Automatic left pickup never takes this path.
                            ignore_workspace_safety = True
                            recorder.add_event(
                                "workspace_safety_override",
                                step=total_steps,
                                action=np.asarray(action).tolist(),
                                workspace=workspace_detail,
                                decision_source=(
                                    "interactive"
                                    if args.interactive_safety_prompts
                                    else "automatic"
                                ),
                            )
                            if args.interactive_safety_prompts:
                                print(
                                    "[OVERRIDE] i accepted; XYZ workspace checks are "
                                    "disabled for this rollout. Joint limits remain active.",
                                    flush=True,
                                )
                            else:
                                print(
                                    "[TASK ZONE] Expected right-arm insertion height "
                                    "accepted; continuing without a key press. "
                                    "Joint limits remain active.",
                                    flush=True,
                                )
                        else:
                            recorder.add_event(
                                "workspace_safety_rejected",
                                step=total_steps,
                                action=np.asarray(action).tolist(),
                                workspace=workspace_detail,
                                decision_source=(
                                    "interactive"
                                    if args.interactive_safety_prompts
                                    else "automatic"
                                ),
                            )
                            if not args.interactive_safety_prompts:
                                print(
                                    "[SAFETY STOP] Unexpected XYZ boundary: "
                                    + ", ".join(workspace_detail["reasons"]),
                                    flush=True,
                                )
                            finish_rollout(
                                "failure", "workspace_safety_boundary"
                            )
                            print_ready()
                            continue
                else:
                    active_left_pickup_zone = False
                    left_pickup_zone_entered_at = None

                # 检测只读历史命令和反馈，不修改正常 policy 动作。异常模式出现时
                # 停止发送下一步，并要求人工检查；不自动 ClearError/复位。
                if first or requires_smooth_move:
                    contact_guard.reset()
                elif args.contact_guard:
                    try:
                        contact_suspected, contact_detail = contact_guard.observe(
                            action, state,
                        )
                    except ValueError as exc:
                        enter_robot_fault(exc)
                        continue
                    if contact_suspected:
                        recorder.add_event(
                            "suspected_insertion_contact", step=total_steps,
                            **contact_detail,
                        )
                        finish_rollout("failure", "suspected_insertion_contact")
                        faulted = True
                        needs_manual_reset = True
                        print(
                            "[CONTACT HOLD] Right J3 reversed while the target kept "
                            "advancing. The next action was not sent. Remove any "
                            "interference, check controller alarms, then press c CHECK "
                            "and h RESET. Use the physical E-stop if force persists.",
                            flush=True,
                        )
                        continue

                # 记录尝试下发的动作；若此后机器人拒绝命令，fault event 会注明失败。
                recorder.record(
                    state=state,
                    action=action,
                    infer_latency_ms=infer_latency_ms,
                    chunk_index=action_index,
                    images=tuple(img_list),
                    ensemble_sources=ensemble_sources,
                )

                try:
                    if first or requires_smooth_move:
                        move_smoothly(env, last_action, action, max_steps=100)
                        first = False

                    last_action = action.copy()
                    obs = env.step(action, np.array([1, 1]))
                except ROBOT_IO_EXCEPTIONS as exc:
                    enter_robot_fault(exc)
                    continue
                obs["joint_positions"][6] = action[6]
                obs["joint_positions"][13] = action[13]
                state = obs["joint_positions"].astype(np.float32)
                total_steps += 1

                if total_steps >= args.max_steps:
                    finish_rollout("failure", "max_steps_reached")
                    continue

                elapsed = time.monotonic() - tic
                if elapsed < CONTROL_PERIOD_S:
                    time.sleep(CONTROL_PERIOD_S - elapsed)
    except KeyboardInterrupt:
        if recorder.active:
            finish_rollout("aborted", "keyboard_interrupt")
        print("\nManual stop (Ctrl+C)")
    except Exception as exc:
        if recorder is not None and recorder.active:
            recorder.finish("error", f"{type(exc).__name__}: {exc}")
        if env is not None:
            # Whether delivery succeeds or not, do not send later idle-light
            # commands through the same failed transport in the finally block.
            emergency = True
            try:
                set_alarm_light(env)
            except Exception as stop_exc:
                # Keep the first exception visible.  A dead robot server cannot
                # deliver a software E-stop and used to mask the actual timeout
                # with a secondary ZMQ EFSM error here.
                print(
                    f"[ALARM LIGHT FAILED] {type(stop_exc).__name__}: {stop_exc}. "
                    "Use the physical emergency-stop button if needed.",
                    flush=True,
                )
        raise
    finally:
        thread_run = False
        if policy_runner is not None:
            policy_runner.close()
        for thread in threads:
            thread.join(timeout=1)
        for camera in cams:
            camera.close()
        if env is not None and not emergency:
            try:
                set_idle_light(env)
            except Exception as light_exc:
                print(
                    f"[IDLE LIGHT FAILED] {type(light_exc).__name__}: {light_exc}",
                    flush=True,
                )
        if robot_client is not None:
            robot_client.close()
        cv2.destroyAllWindows()
        print("Evaluation stopped.")

if __name__ == "__main__":
    try:
        main(tyro.cli(Args))
    except KeyboardInterrupt:
        print("\n手动停止 (Ctrl+C)")
    finally:
        thread_run = False
        cv2.destroyAllWindows()

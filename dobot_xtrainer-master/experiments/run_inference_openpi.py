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
按 e 急停并结束。数据和成功率默认写到 RockyEVO/data/rollouts/。
"""
import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import csv
import datetime
import json
import pathlib
import pickle
import queue
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
from openpi_client import image_tools
from openpi_client import websocket_client_policy

from dobot_control.agents import async_policy
from dobot_control.cameras.realsense_camera import RealSenseCamera
from dobot_control.env import RobotEnv
from dobot_control.robots.robot_node import ZMQClientRobot
from scripts.manipulate_utils import load_ini_data_camera


CONTROL_PERIOD_S = 0.04
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


def check_action_safety(env, action, check_workspace=True):
    """与原厂 run_inference.py 完全一致的安全边界，阈值未改动。

    返回触发的保护类型；workspace 可由操作者显式覆盖，joint 不允许覆盖。
    """
    violations = []

    # 左臂 -150<J3<0, J4>-35（度）；右臂 150>J3>0, J4<35
    if not ((-2.6 < action[2] < 0 and action[3] > -0.6)
            and (0 < action[9] < 2.6 and action[10] < 0.6)):
        print("[Warn]:The J3 or J4 joints of the robotic arm are out of the safe position!")
        print(action)
        violations.append("joint")

    # 夹爪指尖 XYZ 工作空间限制（mm）
    if check_workspace:
        pos = env.get_XYZrxryrz_state()
        if not ((-410 < pos[0] < 300 and -700 < pos[1] < -210 and pos[2] > 42)
                and (-250 < pos[6] < 410 and -700 < pos[7] < -210 and pos[8] > 42)):
            print("[Warn]:The robot arm XYZ is out of the safe position!")
            print(pos)
            violations.append("workspace")

    return violations


def emergency_stop(env):
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
    rollout_generation = 0
    pending_request_id = None
    pending_request_started = None
    next_replan_step = 0

    def finish_rollout(result, reason="", idle_light=True):
        nonlocal running, action_chunk, chunk_idx, rollout_generation
        nonlocal pending_request_id, pending_request_started
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
        if idle_light:
            try:
                set_idle_light(env)
            except Exception as light_exc:
                print(
                    f"[IDLE LIGHT FAILED] {type(light_exc).__name__}: {light_exc}",
                    flush=True,
                )
        meta = recorder.finish(result, reason)
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
        print("[READY] h RESET, then r START the next rollout.", flush=True)

    def reset_robot():
        """Return to the standard initial pose without creating a rollout."""
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
        print("[RESET] Complete (orange light = idle).", flush=True)

    def start_rollout():
        nonlocal running, state, last_action, first
        nonlocal action_chunk, chunk_idx, total_steps, ignore_workspace_safety
        nonlocal infer_latency_ms, rollout_generation
        nonlocal pending_request_id, pending_request_started, next_replan_step
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

    def wait_for_safety_key(keys, message):
        """安全提示期间同时接收终端和 OpenCV 预览窗口按键。"""
        print(message, flush=True)
        while True:
            key = keys.poll()
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
                cv_key = cv2.waitKey(20) & 0xFF
                if cv_key != 255:
                    key = chr(cv_key).lower()
            if key is not None:
                return key
            time.sleep(0.02)

    print("\n-------------- X-Trainer real-robot evaluation --------------")
    print("r: start + record     s: success     f: failure")
    print("h: abort if running + reset/home     q: finish program")
    print("e: emergency stop + finish program")
    print("green = running/recording; orange = idle/recording stopped")
    print(
        "policy control: "
        + ("asynchronous closed-loop" if args.async_inference else "synchronous fallback")
    )
    print("i: ignore XYZ workspace limit only when the override prompt is shown")
    print("q during a rollout records ABORTED and does not enter the success rate.")
    print("Press r when the scene and robot are ready.\n")

    try:
        with TerminalKeys() as keys:
            while not quit_requested:
                for i, thread in enumerate(threads):
                    if not thread.is_alive():
                        raise RuntimeError(f"camera thread {i} died")

                key = keys.poll()
                if args.show_img:
                    show_canvas[:, :640] = img_list[0][:, :, ::-1]
                    show_canvas[:, 640:1280] = img_list[1][:, :, ::-1]
                    show_canvas[:, 1280:1920] = img_list[2][:, :, ::-1]
                    if running:
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
                    cv_key = cv2.waitKey(1) & 0xFF
                    if cv_key != 255:
                        key = chr(cv_key).lower()

                if not running:
                    if key == "r":
                        start_rollout()
                    elif key == "h":
                        reset_robot()
                    elif key == "q":
                        quit_requested = True
                    elif key == "e":
                        emergency_stop(env)
                        emergency = True
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
                    reset_robot()
                    continue
                if key == "q":
                    finish_rollout("aborted", "operator_quit")
                    quit_requested = True
                    continue
                if key == "e":
                    emergency_stop(env)
                    emergency = True
                    finish_rollout("failure", "emergency_stop", idle_light=False)
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
                if np.max(np.abs(arm_delta)) > 0.17:
                    print(
                        "Note! Joint increment larger than 10 degrees:",
                        np.round(arm_delta, 4),
                    )
                    decision = wait_for_safety_key(
                        keys,
                        "JOINT JUMP >10deg | y CONTINUE | other FAIL",
                    )
                    if decision == "y":
                        requires_smooth_move = True
                    else:
                        finish_rollout(
                            "failure", "joint_increment_rejected"
                        )
                        print_ready()
                        continue

                violations = check_action_safety(
                    env,
                    action,
                    check_workspace=not ignore_workspace_safety,
                )
                if "joint" in violations:
                    # The rejected action has not been sent to the robot, so end
                    # this rollout and return to idle instead of killing the
                    # whole evaluation process. The operator can press h to
                    # reset and then start the next rollout.
                    finish_rollout("failure", "joint_safety_boundary")
                    print_ready()
                    continue
                if "workspace" in violations:
                    decision = wait_for_safety_key(
                        keys,
                        "XYZ LIMIT | i IGNORE THIS ROLLOUT | other FAIL",
                    )
                    if decision == "i":
                        ignore_workspace_safety = True
                        recorder.add_event(
                            "workspace_safety_override",
                            step=total_steps,
                            action=np.asarray(action).tolist(),
                        )
                        print(
                            "[OVERRIDE] XYZ workspace limit disabled for this rollout; "
                            "joint limits remain active.",
                            flush=True,
                        )
                    else:
                        finish_rollout(
                            "failure", "workspace_safety_boundary"
                        )
                        print_ready()
                        continue

                # 通过安全检查后再落盘，保证记录中的 control 确实被下发。
                recorder.record(
                    state=state,
                    action=action,
                    infer_latency_ms=infer_latency_ms,
                    chunk_index=action_index,
                    images=tuple(img_list),
                    ensemble_sources=ensemble_sources,
                )

                if first or requires_smooth_move:
                    move_smoothly(env, last_action, action, max_steps=100)
                    first = False

                last_action = action.copy()
                obs = env.step(action, np.array([1, 1]))
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
                emergency_stop(env)
            except Exception as stop_exc:
                # Keep the first exception visible.  A dead robot server cannot
                # deliver a software E-stop and used to mask the actual timeout
                # with a secondary ZMQ EFSM error here.
                print(
                    f"[EMERGENCY STOP FAILED] {type(stop_exc).__name__}: {stop_exc}. "
                    "Use the physical emergency-stop button.",
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

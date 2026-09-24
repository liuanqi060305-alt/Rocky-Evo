import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import cv2
import time
from dataclasses import dataclass
import pathlib
from typing import Literal, Optional
import numpy as np
import tyro
import threading
from dobot_control.agents.agent import BimanualAgent
from scripts.format_obs import save_frame
from scripts import episode_io
from scripts import data_paths
from dobot_control.env import RobotEnv
from dobot_control.robots.robot_node import ZMQClientRobot
from scripts.function_util import mismatch_data_write, wait_period, log_write, mk_dir
from scripts.manipulate_utils import robot_pose_init, pose_check, dynamic_approach, obs_action_check, servo_action_check, load_ini_data_hands, set_light, load_ini_data_camera
from dobot_control.agents.dobot_agent import DobotAgent
from dobot_control.cameras.realsense_camera import RealSenseCamera
import datetime
from pathlib import Path
import requests

@dataclass
class Args:
    robot_port: int = 6002
    hostname: str = "127.0.0.1"
    show_img: bool = False
    # Raw 数据根目录。留空(None)时按统一配置层 scripts/data_paths.py 解析：
    #   环境变量 XTRAINER_RAW_DATA_ROOT > config/data_paths.yaml > 项目 data/raw
    # 项目 data/raw 是指向本工程 datasets/ 的软链接，因此默认行为与改造前一致
    # （数据仍落在 <本工程>/datasets/），只是不再写死。
    #
    # 原实现是 `save_data_path = str(Path(__file__)...)`，无类型标注 ->
    # 在 dataclass 里不算 field -> tyro 从不暴露它，路径完全无法覆盖，
    # 且随「从哪个仓库副本启动」而变。加上标注后才成为真正的 CLI 参数。
    save_data_path: Optional[str] = None
    project_name: str = "plug_and_unplug_task"
    # 录制结束后自动为该条 episode 生成三路拼接视频 replay.mp4（后台线程，不阻塞控制循环）
    auto_video: bool = True
    # 位置泛化标注。position_id 留空表示普通采集；设置后自动写进每条 meta.json。
    # 这些字段只用于数据平衡和泛化评估，不会写进模型的语言 prompt。
    position_id: Optional[str] = None
    position_description: Optional[str] = None
    position_object: str = "socket"
    position_frame: str = "table_reference"
    position_dx_mm: Optional[float] = None
    position_dy_mm: Optional[float] = None
    position_dz_mm: Optional[float] = None
    position_yaw_deg: Optional[float] = None
    position_split: Literal["train", "validation", "test"] = "train"

    def resolved_save_dir(self) -> str:
        """返回本次采集的 Raw 根目录，末尾带 '/'（沿用既有拼接约定）。"""
        root = self.save_data_path
        if root is None:
            root = str(data_paths.paths.raw)
        return os.path.join(str(pathlib.Path(root).expanduser()), "")

    def generalization_annotation(self) -> Optional[dict]:
        """返回写入 episode meta.json 的位置泛化标注。"""
        if self.position_id is None:
            return None
        return {
            "factor": "object_position",
            "object": self.position_object,
            "position_id": self.position_id,
            "description": self.position_description,
            "reference_frame": self.position_frame,
            "offset_mm": {
                "x": self.position_dx_mm,
                "y": self.position_dy_mm,
                "z": self.position_dz_mm,
            },
            "yaw_deg": self.position_yaw_deg,
            "split": self.position_split,
        }


# Thread button: [lock or nor, servo or not, record or not]
# 0: lock, 1: unlock
# 0: stop servo, 1: servo
# 0: stop recording, 1: recording
what_to_do = np.array(([0, 0, 0], [0, 0, 0]))
dt_time = np.array([20240507161455])
using_sensor_protection = False
is_falling = np.array([0])

def button_monitor_realtime(agent):
    """
    按键监听线程 — 与原始 3_just_buttonA.py 完全一致的 delta 检测
    - Button A 短按 (<0.5s)：切换锁定/解锁
    - Button A 长按 (>1s) ：切换从手跟随 开/关
    - Button B           ：切换录像 开/关
    """
    import traceback

    last_keys_status = np.zeros((2, 3), dtype=int)
    start_press_status = np.zeros((2, 2), dtype=int)
    keys_press_count = np.zeros((2, 3), dtype=int)
    tic_btn = np.zeros((2, 2))  # per-hand, per-button press start time

    print(f"[BTN] thread started, waiting for first key read...", flush=True)

    while not is_falling[0]:
        try:
            now_keys = agent.get_keys().astype(int)
            dev_keys = now_keys - last_keys_status

            # --- Button A (index 0) ---
            for i in range(2):
                if dev_keys[i, 0] == -1:  # press
                    tic_btn[i, 0] = time.time()
                    start_press_status[i, 0] = 1
                if dev_keys[i, 0] == 1 and start_press_status[i, 0]:  # release
                    start_press_status[i, 0] = 0
                    toc = time.time()
                    held = toc - tic_btn[i, 0]
                    if held < 0.5:
                        keys_press_count[i, 0] += 1
                        what_to_do[i, 0] = keys_press_count[i, 0] % 2
                        label = "UNLOCK" if what_to_do[i, 0] else "LOCK"
                        print(f"[BTN] A[{i}] SHORT={held*1000:.0f}ms -> {label}  what_to_do={what_to_do.tolist()}", flush=True)
                    elif held > 1:
                        keys_press_count[i, 1] += 1
                        what_to_do[i, 1] = keys_press_count[i, 1] % 2
                        label = "SERVO ON" if what_to_do[i, 1] else "SERVO OFF"
                        print(f"[BTN] A[{i}] LONG={held*1000:.0f}ms -> {label}  what_to_do={what_to_do.tolist()}", flush=True)

            # --- Button B (index 1) ---
            for i in range(2):
                if dev_keys[i, 1] == -1:  # press
                    tic_btn[i, 1] = time.time()
                    start_press_status[i, 1] = 1
                if dev_keys[i, 1] == 1 and start_press_status[i, 1]:  # release
                    start_press_status[i, 1] = 0
                    keys_press_count[0, 2] += 1
                    what_to_do[0, 2] = keys_press_count[0, 2] % 2
                    if what_to_do[0, 2]:
                        now_time = datetime.datetime.now()
                        dt_time[0] = int(now_time.strftime("%Y%m%d%H%M%S"))
                        print(f"[BTN] B REC START [{dt_time[0]}]", flush=True)
                    else:
                        print(f"[BTN] B REC STOP", flush=True)

            last_keys_status = now_keys.copy()

            # 传感器保护
            if using_sensor_protection:
                for i in range(2):
                    if now_keys[i, 2] and what_to_do[i, 0]:
                        agent.set_torque(2, True)
                        is_falling[0] = 1

        except Exception as e:
            print(f"[BTN] ERROR: {e}", flush=True)
            traceback.print_exc()
            time.sleep(0.1)


# Thread: camera
# img_list 用 python list 存「整帧数组的引用」，相机线程私下构造好一整帧后只做一次
# 元素赋值。list 元素赋值是单条 STORE_SUBSCR 字节码，GIL 下原子、不释放 GIL，
# 读者取到的数组对象此后不再被改动，因此永远读到自洽的一帧。
#
# 旧写法 np.array([...]) 建的是 float64 (3,480,640,3) 大数组，写入是「切片赋值 +
# uint8->float64 转换」，numpy 转换循环会释放 GIL；读者 img_list[0] 拿到的是指向同一
# 块内存的视图，cv2 编码期间相机线程仍在改写 → 上下半幅来自不同时刻。实测撕裂率 0.3%
# （引用交换方案同条件下 0%）。改成 uint8 顺带省掉每帧 2.52ms 的 dtype 转换，
# 内存占用从 22MB 降到 2.8MB。
img_list = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(3)]


# 每路相机连续失败次数。read() 内部已重试5轮+重建pipeline，这里再兜一层，
# 避免一次 USB 抖动直接杀死线程、进而让主循环 assert 崩掉整个程序。
cam_fail_count = np.array([0, 0, 0])
CAM_FAIL_LIMIT = 10


def run_thread_cam(rs_cam, which_cam):
    while 1:
        try:
            # 三路相机统一：读原图 + 通道翻转(RGB->BGR)，不做裁剪/缩放
            image_cam, _ = rs_cam.read()
            # ascontiguousarray 同时完成两件事：把 [:, :, ::-1] 的负步长视图实体化成
            # 独立连续副本（read() 返回的视图指向 librealsense 帧缓冲，会被回收），
            # 以及为下面的引用交换准备一个此后不再被改动的数组。
            image_cam = np.ascontiguousarray(image_cam[:, :, ::-1])

            # 原先这里还有一次 cv2.imencode 写入 npy_list，但 npy_list 在本文件内
            # 只写不读（存盘走 cv2.imwrite(img_list[n])，自己重新编码），属纯浪费，
            # 每帧 1.80ms。已连同 npy_list/npy_len_list 一起删除。
            # 注意：gui/ 下有同名变量，是 GUI 自己的独立副本，不受此改动影响。
            img_list[which_cam] = image_cam
            cam_fail_count[which_cam] = 0
        except Exception as e:
            cam_fail_count[which_cam] += 1
            print(f"[CAM {which_cam}] read error "
                  f"{cam_fail_count[which_cam]}/{CAM_FAIL_LIMIT}: {e}", flush=True)
            if cam_fail_count[which_cam] >= CAM_FAIL_LIMIT:
                print(f"[CAM {which_cam}] giving up, thread exiting", flush=True)
                raise
            time.sleep(0.1)


def _encode_episode_video(traj_dir, fps=25.0):
    """把一条 episode 的三路 jpg 拼成 replay.mp4，存在该 episode 目录下。

    复用 scripts/replay_video.py 的 frame_indices / load_stitched，保证和离线
    回放工具产出的视频格式完全一致（1920x480 横向拼接、绿色相机名、白色帧号）。
    传入的 fps 仅作 fallback：V2 数据会用真实时间戳算出的实测帧率覆盖它。
    """
    from scripts.replay_video import frame_indices, load_stitched, CAMS, W, H
    # V2：帧率由真实时间戳算出，不再用硬编码 25。
    # 审计实测真实采集是 18.3Hz，按 25 写出的视频快 1.37 倍，会误导人工检视。
    # V1 数据无时间戳时 effective_fps 自动退回传入的 fallback。
    try:
        measured = episode_io.effective_fps(traj_dir, fallback=fps)
        if measured and measured > 0:
            fps = measured
    except Exception as e:
        print(f"[VIDEO] 帧率推算失败，用 {fps}: {e}", flush=True)
    out_path = os.path.join(traj_dir, "replay.mp4")
    try:
        idxs = frame_indices(traj_dir)
    except SystemExit as e:      # frame_indices 找不到 topImg 时会 raise SystemExit
        print(f"[VIDEO] skip {traj_dir}: {e}", flush=True)
        return
    if not idxs:
        print(f"[VIDEO] skip {traj_dir}: 没有帧", flush=True)
        return
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (W * len(CAMS), H))
    if not writer.isOpened():
        print(f"[VIDEO] 无法创建 {out_path}", flush=True)
        return
    try:
        for i in idxs:
            writer.write(load_stitched(traj_dir, i))
    finally:
        writer.release()
    size_mb = os.path.getsize(out_path) / 1e6 if os.path.exists(out_path) else 0
    print(f"[VIDEO] {out_path}  {len(idxs)} 帧  {size_mb:.1f} MB", flush=True)


def spawn_episode_video(traj_dir, fps=25.0):
    """在后台守护线程里生成视频，不阻塞控制循环。

    编码 1500 帧（约一分钟数据）需要数秒，放在主循环里会打断 25Hz 节拍，
    因此必须异步。daemon=True：程序退出时不等它，未编完的视频可事后用
    scripts/batch_replay_video.py 补。
    """
    def _work():
        try:
            _encode_episode_video(traj_dir, fps)
        except Exception as e:
            print(f"[VIDEO] {traj_dir} 生成失败: {type(e).__name__}: {e}", flush=True)
    threading.Thread(target=_work, daemon=True).start()


def write_episode_meta_start(env, save_dir, ep_id, task_name, action, generalization=None):
    """录制开始时写 meta.json。失败只告警，绝不影响采集。

    起始位姿在这里采样一次即可满足元数据需求，不必进主循环 —— 逐帧调
    get_XYZrxryrz_state() 会给 25Hz 控制回路增加额外 TCP 往返，违反
    "不改变控制频率"的约束。
    """
    ep_dir = os.path.join(save_dir, str(ep_id))
    try:
        pos = np.asarray(env.get_XYZrxryrz_state(), dtype=float)  # 14 维：左7 + 右7
        left_pos, right_pos = pos[:7].tolist(), pos[7:].tolist()
    except Exception as e:
        print(f"[META] 读取 TCP 位姿失败，left/right_position 留空: {e}", flush=True)
        left_pos, right_pos = None, None
    try:
        act = np.asarray(action, dtype=float)
        meta = episode_io.build_meta(
            episode_id=str(ep_id),
            task_name=task_name,
            episode_start_time_ns=time.time_ns(),
            episode_start_monotonic_ns=time.monotonic_ns(),
            left_position=left_pos,
            left_angle=act[:7],
            right_position=right_pos,
            right_angle=act[7:],
            data_type="teleop",
            extra={"generalization": generalization} if generalization else None,
        )
        episode_io.write_meta(ep_dir, meta)
        print(f"[META] 已写入 {ep_dir}/meta.json", flush=True)
    except Exception as e:
        print(f"[META] 写 meta.json 失败（采集继续）: {e}", flush=True)


def finalize_episode_meta(ep_dir):
    """录制结束回填统计字段。失败只告警。"""
    try:
        m = episode_io.finalize_meta(ep_dir)
        if m:
            print(f"[META] {ep_dir}: {m.get('num_frames')} 帧, "
                  f"{m.get('duration_s')}s, 实测 {m.get('fps_measured')} fps", flush=True)
    except Exception as e:
        print(f"[META] 回填 meta.json 失败: {e}", flush=True)


def dh_transformation_matrix(theta, d, a, alpha):
    """
    Create the DH transformation matrix
    """
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)
    return np.array([
        [cos_theta, -sin_theta * cos_alpha, sin_theta * sin_alpha, a * cos_theta],
        [sin_theta, cos_theta * cos_alpha, -cos_theta * sin_alpha, a * sin_theta],
        [0, sin_alpha, cos_alpha, d],
        [0, 0, 0, 1]
    ])

def claw_width(coef):
    """
    Calculate the claw width
    """
    claw_servo = 2.3818 - coef * 1.5401
    cos_claw_servo = np.cos(claw_servo)
    claw_wid = 0.03 * cos_claw_servo + 0.5 * np.sqrt(0.0036 * cos_claw_servo ** 2 + 0.0028)
    return claw_wid

def forward_kinematics(q0, q1, q2, q3, q4, q5, y, r_type):
    """
    Compute the forward kinematics
    """
    if r_type == "Nova 2":
        dh_params = [
            (q0, 0.2234, 0, np.pi / 2),
            (q1 - np.pi / 2, 0, -0.280, 0),
            (q2, 0, -0.225, 0),
            (q3 - np.pi / 2, 0.1175, 0, np.pi / 2),
            (q4, 0.120, 0, -np.pi / 2),
            (q5, 0.088, 0, 0)
        ]
    if r_type == "Nova 5":
        dh_params = [
            (q0, 0.240, 0, np.pi / 2),
            (q1 - np.pi / 2, 0, -0.400, 0),
            (q2, 0, -0.330, 0),
            (q3 - np.pi / 2, 0.135, 0, np.pi / 2),
            (q4, 0.120, 0, -np.pi / 2),
            (q5, 0.088, 0, 0)
        ]

    t = np.eye(4)
    for params in dh_params:
        t = np.dot(t, dh_transformation_matrix(*params))
    t_tool = np.eye(4)
    t_tool[:3, 3] = np.array([0, y, 0.2])
    t_final = np.dot(t, t_tool)
    pos = t_final[:3, 3]
    return pos


def calculate_vel_pos(action, last_action, total_time, r_type):
    """
    Calculate the velocity for forward kinematics
    """
    claw_left = claw_width(action[6])
    claw_right = claw_width(action[13])

    positions = {}
    vel = {}

    for side in ['left', 'right']:
        for paw in ['left', 'right']:
            coef = 1 if paw == 'left' else -1
            claw = claw_left if side == 'left' else claw_right
            claw *= coef

            current_fk = forward_kinematics(*action[0:6] if side == 'left' else action[7:13], claw, r_type)
            last_fk = forward_kinematics(*last_action[0:6] if side == 'left' else last_action[7:13], claw, r_type)

            positions[f'{side}_{paw}'] = current_fk
            vel[f'{side}_{paw}'] = (current_fk - last_fk) / total_time

    return positions, vel

# Check that the positions is within a safe zone
def is_within_safe_position(position, x_range, y_range, z_min):
    return x_range[0] <= position[0] <= x_range[1] and \
           y_range[0] <= position[1] <= y_range[1] and \
           position[2] > z_min


def check_pose_protection(positions, vel, what_to_do):
    protect_err = False
    warnings = []

    delta_left_left = vel['left_left']
    delta_left_right = vel['left_right']
    delta_right_left = vel['right_left']
    delta_right_right = vel['right_right']

    positions_mm = {key: value * 1000 for key, value in positions.items()}
    # Define a safe zone
    # left arm (jaw tip position) limit:  300>x>-450  -750<Y<-210  z>42;
    # right arm (jaw tip position) limit:  450>x>-250  -750<Y<-210  z>42;
    x_range_left = (-450, 300)
    x_range_right = (-250, 450)
    y_range = (-750, -160)
    z_range_left = 42
    z_range_right = 42

    if what_to_do[0, 1]:  # The left hand is in sync
        # Z direction speed limit -1 m/s
        if delta_left_left[2] < -2 or delta_left_right[2] < -2:
            warnings.append("[Warn]:The left robot speed of the TCP is moving too fast!")
            warnings.append(f"delta_left_left: {delta_left_left[2]}")
            protect_err = True
        # Left arm working space limitation
        positions_to_check = ['left_left', 'left_right']
        x_ranges = [x_range_left, x_range_left]
        z_ranges = [z_range_left, z_range_left]
        if not all(is_within_safe_position(positions_mm[pos], x_range, y_range, z_range)
                   for pos, x_range, z_range in zip(positions_to_check, x_ranges, z_ranges)):
            warnings.append("[Warn]:The left arm is out of the safe zone!")
            protect_err = True

    if what_to_do[1, 1]:  # The right hand is in sync
        # Z direction speed limit -1 m/s
        if delta_right_left[2] < -1 or delta_right_right[2] < -1:
            warnings.append("[Warn]:The right robot speed of the TCP is moving too fast!")
            warnings.append(f"delta_right_left: {delta_right_left[2]}")
            protect_err = True
        # Right arm working space limitation
        positions_to_check = ['right_left', 'right_right']
        x_ranges = [x_range_right, x_range_right]
        z_ranges = [z_range_right, z_range_right]
        if not all(is_within_safe_position(positions_mm[pos], x_range, y_range, z_range)
                   for pos, x_range, z_range in zip(positions_to_check, x_ranges, z_ranges)):
            warnings.append("[Warn]:The right arm is out of the safe zone!")
            protect_err = True

    for warning in warnings:
        print(warning)

    return protect_err

def check_joint_safety(action):
    protect_err = False
    if not (action[2] < 0):
        print("[Warn]:The J3 joints of the robotic arm are out of the safe position! ")
        print(action)
        protect_err = True
    if not (action[9] > 0):
        print("[Warn]:The J3 joints of the robotic arm are out of the safe position! ")
        print(action)
        protect_err = True
    return protect_err

def get_firmware_version_satisfied(robot_ip):
    try:
        response = requests.post("http://"+robot_ip+":22000/settings/version")
        if response.status_code == 200:
            rt_version = response.text.split("{")[1].split("\n\t")[3].split(":")[1].split("\"")[1].split("-")[0].split(".")
            rt_num = "".join(rt_version)
            return 1, int(rt_num)
        else:
            print("Failed to retrieve the version webpage")
            return 0, 0
    except Exception as e:
        print(e)
        return 0, 0

def get_robot_type(robot_ip):
    response = requests.post("http://"+robot_ip+":22000/properties/controllerType")
    if response.status_code == 200:
        print("The type of robot is:", eval(response.text)["name"])
        return eval(response.text)["name"]
    else:
        print("Failed to obtain the type of robot")
        return None

def check_firmware_version():
    left_version = get_firmware_version_satisfied("192.168.5.1")
    right_version = get_firmware_version_satisfied("192.168.5.2")
    if left_version[1] < 3581 or left_version[1]>=4000:
        print("[ERROR]Left hand error[192.168.5.1]:firmware version requires V3 and must >=3.5.8.1 (found: %s),please check and update"%left_version[1])
        return False
    if right_version[1] < 3581 or right_version[1]>=4000:
        print("[ERROR]Right hand error[192.168.5.1]:firmware version requires V3 and must >=3.5.8.1 (found: {%s}),please check and update"% right_version[1])
        return False
    return True

# 资源登记表：daemon 线程被强杀不会清理硬件，退出时由 _shutdown() 统一优雅释放
_cleanup_targets = {"cameras": [], "robot_client": None}


def _shutdown():
    """进程退出前尽力释放硬件资源（相机 pipeline / ZMQ socket）。best-effort，失败不影响退出。"""
    for cam in _cleanup_targets["cameras"]:
        try:
            cam.close()
        except Exception as e:
            print(f"[cleanup] camera pipeline stop failed: {e}")
    rc = _cleanup_targets["robot_client"]
    if rc is not None:
        try:
            rc.close()
        except Exception as e:
            print(f"[cleanup] robot_client close failed: {e}")


def main(args):
    # create dataset file path
    raw_root = args.resolved_save_dir()
    save_dir = raw_root+args.project_name+"/collect_data"
    print(f"[DATA] Raw 根目录: {raw_root}")
    print(f"[DATA]   实体位置: {pathlib.Path(raw_root).resolve()}")
    print(f"[DATA]   本次写入: {save_dir}")
    mk_dir(save_dir)

    # camera init
    camera_dict = load_ini_data_camera()
    rs1 = RealSenseCamera(flip=True, device_id=camera_dict["top"])
    _cleanup_targets["cameras"].append(rs1)
    rs2 = RealSenseCamera(flip=False, device_id=camera_dict["left"])
    _cleanup_targets["cameras"].append(rs2)
    rs3 = RealSenseCamera(flip=True, device_id=camera_dict["right"])
    _cleanup_targets["cameras"].append(rs3)
    thread_cam_top = threading.Thread(target=run_thread_cam, args=(rs1, 0))
    thread_cam_left = threading.Thread(target=run_thread_cam, args=(rs2, 1))
    thread_cam_right = threading.Thread(target=run_thread_cam, args=(rs3, 2))
    thread_cam_top.daemon = True
    thread_cam_left.daemon = True
    thread_cam_right.daemon = True
    thread_cam_top.start()
    thread_cam_left.start()
    thread_cam_right.start()
    show_canvas = np.zeros((480, 640*3, 3), dtype=np.uint8)
    time.sleep(2)
    print("camera thread init success...")

    # agent init
    _, hands_dict = load_ini_data_hands()
    left_agent = DobotAgent(which_hand="LEFT", dobot_config=hands_dict["HAND_LEFT"])
    right_agent = DobotAgent(which_hand="RIGHT", dobot_config=hands_dict["HAND_RIGHT"])
    agent = BimanualAgent(left_agent, right_agent)

    # pose init
    print("Waiting to connect the robot...")
    robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)
    _cleanup_targets["robot_client"] = robot_client
    print("If the robot fails to initialize successfully after 5 seconds,please check that the robot network is connected correctly and make sure TCP/IP mode is turned!")
    if check_firmware_version()==False:
        return
    robot_type = get_robot_type("192.168.5.1")
    env = RobotEnv(robot_client)
    env.set_do_status([1, 0])
    env.set_do_status([2, 0])
    env.set_do_status([3, 0])
    robot_pose_init(env)
    start_servo = False
    curr_light = "dark"
    print("robot init success....")

    # button status init
    last_status = np.array(([0, 0, 0], [0, 0, 0]))  # init lock
    thread_button = threading.Thread(target=button_monitor_realtime, args=(agent, ))
    thread_button.daemon = True
    thread_button.start()
    print("button thread init success...")

    print("-------------------------Ok, let's start------------------------")
    idx = 0
    safe_limit = 0
    total_time = 0.04

    while 1:
        tic = time.time()

        assert thread_cam_top.is_alive(), "Error: please check the top camera!"
        assert thread_cam_left.is_alive(), "Error: please check the left camera!"
        assert thread_cam_right.is_alive(), "Error: please check the right camera!"
        assert not is_falling, "sensor   detection!"

        action = agent.act({})
        dev_what_to_do = what_to_do.copy()-last_status
        last_status = what_to_do.copy()

        # debug: print action delta if any button state changed
        if dev_what_to_do.any():
            print(f"[MAIN] dev={dev_what_to_do.tolist()}  what_to_do={what_to_do.tolist()}", flush=True)

        # button A: short press event. lock and unlock
        for i in range(2):
            if dev_what_to_do[i, 0] != 0:
                print(f"[MAIN] set_torque hand[{i}] = {not what_to_do[i,0]} (lock={not what_to_do[i,0]})", flush=True)
                agent.set_torque(i, not what_to_do[i, 0])

        # button A: long press event. servo or not
        if dev_what_to_do[0, 1] == 1 or dev_what_to_do[1, 1] == 1:
            # pose check between main hand and the follower
            print("dynamic approach")
            for i in range(2):
                if what_to_do[i, 1]:
                    agent.set_torque(i, True)
            flag_in = np.array([what_to_do[0, 1], what_to_do[1, 1]])
            last_action = dynamic_approach(env, agent, flag_in)
            for i in range(2):
                if what_to_do[i, 0]:
                    if what_to_do[i, 1]:
                        agent.set_torque(i, False)
            start_servo = True
            obs = env.get_obs()
            if curr_light != "green":
                curr_light = set_light(env, "yellow", 1)

        if dev_what_to_do[0, 1] == -1 or dev_what_to_do[1, 1] == -1:
            # long press toggled servo OFF: explicitly disable torque
            print(f"[MAIN] servo OFF, disabling torque", flush=True)
            for i in range(2):
                if dev_what_to_do[i, 1] == -1:
                    agent.set_torque(i, False)
            flag_in = np.array([what_to_do[0, 1], what_to_do[1, 1]])
            if what_to_do[0, 1] == 0 and what_to_do[1, 1] == 0:
                set_light(env, "green", 0)

        if (what_to_do[0, 1] or what_to_do[1, 1]) and start_servo:
            action = agent.act({})
            err3, action = servo_action_check(action, last_action, flag_in)
            assert err3 != 0, set_light(env, "red", 1)

            # ×××××××××××××××××××××××××××××Security protection×××××××××××××××××××××××××××××××××××××××××××
            # [Note]: Modify the protection parameters in this section carefully !
            # protect_err = [False, False]
            # if (safe_limit < 1):
            #     safe_limit = safe_limit + 1
            # else:
            #     positions, vel = calculate_vel_pos(action, last_action, total_time, robot_type)
            #     protect_err[0] = check_pose_protection(positions, vel, what_to_do)
            #     protect_err[1] = check_joint_safety(action)
            # if any(protect_err):
            #     set_light(env, "red", 1)
            #     time.sleep(1)
            #     exit()
            # ×××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××

            # button B: recording or not
            if dev_what_to_do[0, 2] == 1:
                curr_light = set_light(env, "green", 1)
                # 录制开始 -> 写 episode 级 meta.json。
                # 放在按键边沿而非主循环内：这里每条 episode 只执行一次，
                # get_XYZrxryrz_state() 的一次 TCP 读取不会影响控制频率。
                write_episode_meta_start(env, save_dir, dt_time[0],
                                        args.project_name, action,
                                        args.generalization_annotation())
            elif dev_what_to_do[0, 2] == -1:
                curr_light = set_light(env, "yellow", 1)
                # 录制刚停止，该条 episode 已写完 -> 后台生成 replay.mp4。
                # 必须在这里就把目录路径算出来传进去：dt_time[0] 会在下次按 B 时被改写。
                if args.auto_video:
                    spawn_episode_video(save_dir + f"/{dt_time[0]}")
                # 回填 num_frames / duration_s / fps_measured（只有写完才知道）
                finalize_episode_meta(save_dir + f"/{dt_time[0]}")
            if what_to_do[0, 2] == 1:
                # 单一时间基准：本帧只采样一次时间，图像/state/action 共用。
                # 分别取时间会让三者带上互不相同的戳，无法用于对齐分析。
                # time_ns 是墙钟（跨进程可比、可转日历时间），monotonic_ns 单调
                # 递增不受 NTP 调时影响，算帧间隔用它才可靠 —— 两个都存。
                frame_time_ns = time.time_ns()
                frame_mono_ns = time.monotonic_ns()

                idx += 1
                left_dir = save_dir + f"/{dt_time[0]}/leftImg/"
                right_dir = save_dir + f"/{dt_time[0]}/rightImg/"
                top_dir = save_dir + f"/{dt_time[0]}/topImg/"
                mk_dir(right_dir)
                mk_dir(top_dir)
                if mk_dir(left_dir):
                    idx = 0
                cv2.imwrite(top_dir + f"{idx}.jpg", img_list[0])
                cv2.imwrite(left_dir + f"{idx}.jpg", img_list[1])
                cv2.imwrite(right_dir + f"{idx}.jpg", img_list[2])

                obs_dir = save_dir + f"/{dt_time[0]}/observation/"
                mk_dir(obs_dir)
                save_frame(obs_dir, idx, obs, action,
                           timestamp_ns=frame_time_ns, monotonic_ns=frame_mono_ns)

            obs = env.step(action, flag_in)
            obs["joint_positions"][6] = action[6]
            obs["joint_positions"][13] = action[13]
            last_action = action
        else:
            start_servo = False
            safe_limit = 0

        # img show
        if args.show_img:
            show_canvas[:, :640] = np.asarray(img_list[0], dtype="uint8")
            show_canvas[:, 640:640 * 2] = np.asarray(img_list[1], dtype="uint8")
            show_canvas[:, 640 * 2:640 * 3] = np.asarray(img_list[2], dtype="uint8")
            cv2.imshow("0", show_canvas)
            cv2.waitKey(1)

        toc = time.time()
        total_time = toc-tic
        # throttle to ~20Hz to avoid CPU hogging and give button thread time
        if total_time < 0.04:
            time.sleep(0.04 - total_time)


if __name__ == "__main__":
    try:
        main(tyro.cli(Args))
    except KeyboardInterrupt:
        print("\n手动停止 (Ctrl+C)，正在退出...")
    finally:
        _shutdown()

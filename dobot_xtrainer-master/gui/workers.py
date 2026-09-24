"""Background worker threads for long-running operations."""
import sys
import os
import time
import traceback
import subprocess
import configparser
from pathlib import Path
import numpy as np

from PyQt6.QtCore import QThread, pyqtSignal

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


# ── Step 1: Port Scan ────────────────────────────────────────────

class PortScanWorker(QThread):
    """Scan all serial ports."""
    ports_found = pyqtSignal(list)
    log_message = pyqtSignal(str)
    error = pyqtSignal(str)

    def run(self):
        try:
            import serial.tools.list_ports as serial_stl
            self.log_message.emit("正在扫描串口...")
            com_list = []
            ports = list(serial_stl.comports())
            for p in ports:
                if "USB" in p.device:
                    com_list.append(p.device)
                if "ACM" in p.device:
                    com_list.append(p.device)

            # Grant permissions
            ini_path = os.path.join(
                os.path.dirname(__file__), "..", "scripts", "dobot_config", "dobot_settings.ini"
            )
            ini = configparser.ConfigParser()
            ini.read(ini_path)
            passwd = os.environ.get("XTRAINER_SUDO_PASSWORD") or ini.get(
                "COMPUTER", "passcode", fallback=""
            )
            if not passwd:
                raise RuntimeError(
                    "未配置串口授权密码：请设置 XTRAINER_SUDO_PASSWORD，"
                    "或在本机 dobot_settings.ini 的 [COMPUTER] 中填写 passcode"
                )

            for port in com_list:
                subprocess.run(
                    ["sudo", "-S", "chmod", "777", port],
                    input=passwd + "\n",
                    text=True,
                    capture_output=True,
                    check=True,
                )

            self.log_message.emit(f"发现 {len(com_list)} 个端口: {com_list}")
            self.ports_found.emit(com_list)
        except Exception as e:
            self.error.emit(str(e))
            traceback.print_exc()


class PortConfigWorker(QThread):
    """Auto-configure hands and grippers to ports (replicates 1_find_port.py)."""
    config_complete = pyqtSignal(dict)
    log_message = pyqtSignal(str)
    warning = pyqtSignal(str)

    def run(self):
        try:
            from scripts.manipulate_utils import load_ini_data_hands, load_ini_data_gripper
            from dobot_control.dynamixel.driver import DynamixelDriver
            from dobot_control.gripper.dobot_gripper import DobotGripper
            from scripts.function_util import scan_port

            ini_path = os.path.join(
                os.path.dirname(__file__), "..", "scripts", "dobot_config", "dobot_settings.ini"
            )
            ini_file, hands_dict = load_ini_data_hands()
            port_list = scan_port()
            self.log_message.emit(f"可用端口: {port_list}")

            if len(port_list) < 4:
                self.warning.emit(f"至少需要 4 个端口，当前只有 {len(port_list)} 个")

            result = {}
            baud_rate_list = [2000000, 1000000]

            # Find hand ports
            drivers_to_close = []
            for which_hand in hands_dict.keys():
                found = False
                for _port in list(port_list):
                    for _baud_rate in baud_rate_list:
                        try:
                            self.log_message.emit(f"尝试 {which_hand} @ {_port} ({_baud_rate})")
                            driver = DynamixelDriver(
                                ids=hands_dict[which_hand].joint_ids,
                                append_id=hands_dict[which_hand].append_id,
                                port=_port,
                                baudrate=_baud_rate,
                            )
                            drivers_to_close.append(driver)
                            port_list.remove(_port)
                            self.log_message.emit(f"成功: {which_hand} -> {_port} @ {_baud_rate}")
                            ini_file.set(which_hand, "port", _port)
                            ini_file.set(which_hand, "baud_rate", str(_baud_rate))
                            result[which_hand] = _port
                            found = True
                            break
                        except Exception:
                            continue
                    if found:
                        break

            # Release all Dynamixel ports used during scanning
            for d in drivers_to_close:
                try:
                    d.close()
                except Exception:
                    pass
            drivers_to_close.clear()

            self.log_message.emit(f"剩余端口(夹爪): {port_list}")

            # Find gripper ports
            _, gripper_dict = load_ini_data_gripper()
            for which_gripper in gripper_dict.keys():
                found = False
                for _port in list(port_list):
                    try:
                        self.log_message.emit(f"尝试 {which_gripper} @ {_port}")
                        gripper = DobotGripper(
                            port=_port,
                            servo_pos=gripper_dict[which_gripper].pos,
                            id_name=gripper_dict[which_gripper].id_name,
                        )
                        port_list.remove(_port)
                        self.log_message.emit(f"成功: {which_gripper} -> {_port}")
                        ini_file.set(which_gripper, "port", _port)
                        result[which_gripper] = _port
                        found = True
                        break
                    except Exception as e:
                        self.warning.emit(f"{which_gripper} 失败: {e}")

            # Save
            with open(ini_path, "w+") as f:
                ini_file.write(f)

            self.config_complete.emit(result)
        except Exception as e:
            self.warning.emit(str(e))
            traceback.print_exc()


# ── Step 2: Offset Calibration ───────────────────────────────────

class OffsetReadWorker(QThread):
    """Read current joint offsets (replicates 2_get_offset.py)."""
    offsets_ready = pyqtSignal(dict)
    error = pyqtSignal(str)
    log_message = pyqtSignal(str)

    def run(self):
        driver = None
        try:
            from dobot_control.dynamixel.driver import DynamixelDriver
            from scripts.manipulate_utils import load_ini_data_hands
            from scripts.function_util import scan_port

            # Critical: set port permissions before opening (like original 2_get_offset.py)
            scan_port()

            ini_path = os.path.join(
                os.path.dirname(__file__), "..", "scripts", "dobot_config", "dobot_settings.ini"
            )
            ini_file, hands_dict = load_ini_data_hands()

            gripper_ids = {"HAND_LEFT": [8], "HAND_RIGHT": [18]}
            result = {}

            for which_hand in hands_dict.keys():
                cfg = hands_dict[which_hand]
                self.log_message.emit(f"读取 {which_hand} 关节角度...")
                driver = DynamixelDriver(
                    ids=cfg.joint_ids + gripper_ids[which_hand],
                    append_id=cfg.append_id,
                    port=cfg.port,
                    baudrate=cfg.baud_rate,
                )
                pos_joint = driver.get_joints()
                curr_joints = pos_joint[:6]
                dev_pos = [
                    float(round(float(curr_joints[i]) - cfg.start_joints[i] * cfg.joint_signs[i], 2))
                    for i in range(6)
                ]
                gripper_on = int(np.rad2deg(pos_joint[-1]) - 0.2)
                gripper_close = int(np.rad2deg(pos_joint[-1]) + 30)

                result[which_hand] = {
                    "curr_joints": [round(np.rad2deg(j), 2) for j in curr_joints],
                    "offsets": dev_pos,
                    "gripper_config": [gripper_ids[which_hand][0], gripper_close, gripper_on],
                }

                # Write to ini
                ini_file.set(
                    which_hand, "joint_offsets",
                    str(dev_pos).replace("[", "").replace("]", ""),
                )
                ini_file.set(
                    which_hand, "gripper_config",
                    str(result[which_hand]["gripper_config"]).replace("[", "").replace("]", ""),
                )

                # Release port immediately after reading (avoid port conflict)
                if driver:
                    try:
                        driver.close()
                        driver = None
                    except Exception:
                        pass

            with open(ini_path, "w+") as f:
                ini_file.write(f)

            self.log_message.emit("偏移量已保存")
            self.offsets_ready.emit(result)
        except Exception as e:
            self.error.emit(str(e))
            traceback.print_exc()
        finally:
            if driver:
                try:
                    driver.close()
                except Exception:
                    pass


# ── Step 3: Camera Stream ────────────────────────────────────────

class CameraStreamWorker(QThread):
    """Stream 3 RealSense cameras and emit stitched frames."""
    frame_ready = pyqtSignal(np.ndarray)
    camera_error = pyqtSignal(str, str)  # camera_name, message
    fps_update = pyqtSignal(float)

    def __init__(self, camera_dict=None, parent=None):
        super().__init__(parent)
        self.camera_dict = camera_dict
        self._running = False
        self._frame = np.zeros((480, 1920, 3), dtype=np.uint8)
        self._paused = False

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    def stop(self):
        self._running = False

    def latest_frame(self) -> np.ndarray:
        return self._frame.copy()

    def run(self):
        try:
            from dobot_control.cameras.realsense_camera import RealSenseCamera
            from scripts.manipulate_utils import load_ini_data_camera
        except Exception:
            self.camera_error.emit("import", "无法导入相机模块")
            return

        if self.camera_dict is None:
            try:
                self.camera_dict = load_ini_data_camera()
            except Exception:
                self.camera_dict = {}

        cameras = []
        cam_names = ["top", "left", "right"]
        flips = [True, False, True]

        for i, name in enumerate(cam_names):
            try:
                device_id = self.camera_dict.get(name)
                if not device_id:
                    self.camera_error.emit(name, f"未配置设备 ID")
                    continue
                cam = RealSenseCamera(flip=flips[i], device_id=device_id)
                cameras.append(cam)
            except Exception as e:
                self.camera_error.emit(name, str(e))

        if not cameras:
            self.camera_error.emit("all", "没有可用的相机")
            return

        self._running = True
        last_fps_time = time.time()
        frame_count = 0
        canvas = np.zeros((480, 1920, 3), dtype=np.uint8)

        while self._running:
            if self._paused:
                time.sleep(0.05)
                continue

            try:
                for i in range(min(len(cameras), 3)):
                    _img, _ = cameras[i].read()
                    _img = _img[:, :, ::-1]  # RGB -> BGR
                    col = int(640 * i)
                    canvas[:, col:col + 640] = np.asarray(_img, dtype="uint8")

                self._frame = canvas.copy()
                self.frame_ready.emit(canvas.copy())

                frame_count += 1
                now = time.time()
                if now - last_fps_time >= 1.0:
                    fps = frame_count / (now - last_fps_time)
                    self.fps_update.emit(fps)
                    frame_count = 0
                    last_fps_time = now

            except Exception as e:
                self.camera_error.emit("stream", str(e))
                time.sleep(0.1)

        # Clean up
        for cam in cameras:
            try:
                cam._pipeline.stop()
            except Exception:
                pass


# ── Step 4: ZMQ Server ───────────────────────────────────────────

class ZMQServerWorker(QThread):
    """Run ZMQ robot server (replicates launch_nodes.py)."""
    server_started = pyqtSignal()
    server_stopped = pyqtSignal()
    log_message = pyqtSignal(str)
    robot_error = pyqtSignal(str, str)  # ip, message
    status_update = pyqtSignal(str, str)  # key, status

    def __init__(self, port=6002, host="127.0.0.1", parent=None):
        super().__init__(parent)
        self.port = port
        self.host = host
        self._server = None
        self._stop_flag = False

    def stop_server(self):
        self._stop_flag = True
        if self._server:
            try:
                self._server.stop()
            except Exception:
                pass

    def run(self):
        try:
            from dobot_control.robots.dobot import DobotRobot
            from dobot_control.robots.robot import BimanualRobot
            from dobot_control.robots.robot_node import ZMQServerRobot

            self.log_message.emit("正在连接左臂 (192.168.5.1)...")
            self.status_update.emit("left_robot", "offline")
            self.status_update.emit("right_robot", "offline")

            robot_l = DobotRobot(robot_ip="192.168.5.1", robot_number=2)
            self.log_message.emit("左臂已连接")
            self.status_update.emit("left_robot", "online")

            self.log_message.emit("正在连接右臂 (192.168.5.2)...")
            robot_r = DobotRobot(robot_ip="192.168.5.2", robot_number=2)
            self.log_message.emit("右臂已连接")
            self.status_update.emit("right_robot", "online")

            robot = BimanualRobot(robot_l, robot_r)
            self._server = ZMQServerRobot(robot, port=self.port, host=self.host)
            self.log_message.emit(f"ZMQ 服务已启动: tcp://{self.host}:{self.port}")
            self.status_update.emit("zmq", "online")

            self.server_started.emit()
            self._server.serve()

        except Exception as e:
            self.log_message.emit(f"服务启动失败: {e}")
            self.robot_error.emit("server", str(e))
            self.status_update.emit("zmq", "offline")
            traceback.print_exc()
        finally:
            self.status_update.emit("zmq", "offline")
            self.status_update.emit("left_robot", "offline")
            self.status_update.emit("right_robot", "offline")
            self.server_stopped.emit()


# ── Step 5: Button Monitor ───────────────────────────────────────

class ButtonMonitorWorker(QThread):
    """Monitor master hand buttons in background (replicates button_monitor_realtime)."""
    button_event = pyqtSignal(int, int, str)  # hand_index, button, event_desc
    status_update = pyqtSignal(list)          # what_to_do flattened
    error = pyqtSignal(str)

    def __init__(self, agent, what_to_do, dt_time, is_falling, parent=None):
        super().__init__(parent)
        self.agent = agent
        self.what_to_do = what_to_do
        self.dt_time = dt_time
        self.is_falling = is_falling
        self._running = False

    def stop(self):
        self._running = False

    def run(self):
        import datetime
        self._running = True
        last_keys_status = np.zeros((2, 3), dtype=int)
        start_press_status = np.zeros((2, 2), dtype=int)
        keys_press_count = np.zeros((2, 3), dtype=int)
        tic_btn = np.zeros((2, 2))

        while self._running and not self.is_falling[0]:
            try:
                now_keys = self.agent.get_keys().astype(int)
                dev_keys = now_keys - last_keys_status

                # Button A
                for i in range(2):
                    if dev_keys[i, 0] == -1:
                        tic_btn[i, 0] = time.time()
                        start_press_status[i, 0] = 1
                    if dev_keys[i, 0] == 1 and start_press_status[i, 0]:
                        start_press_status[i, 0] = 0
                        held = time.time() - tic_btn[i, 0]
                        if held < 0.5:
                            keys_press_count[i, 0] += 1
                            self.what_to_do[i, 0] = keys_press_count[i, 0] % 2
                            label = "UNLOCK" if self.what_to_do[i, 0] else "LOCK"
                            hand = "左" if i == 0 else "右"
                            self.button_event.emit(i, 0, f"{hand}手短按A → {label}")
                    elif dev_keys[i, 0] == 1:
                        held = time.time() - tic_btn[i, 0]
                        if held > 1:
                            keys_press_count[i, 1] += 1
                            self.what_to_do[i, 1] = keys_press_count[i, 1] % 2
                            label = "SERVO ON" if self.what_to_do[i, 1] else "SERVO OFF"
                            hand = "左" if i == 0 else "右"
                            self.button_event.emit(i, 0, f"{hand}手长按A → {label}")

                # Button B
                for i in range(2):
                    if dev_keys[i, 1] == -1:
                        tic_btn[i, 1] = time.time()
                        start_press_status[i, 1] = 1
                    if dev_keys[i, 1] == 1 and start_press_status[i, 1]:
                        start_press_status[i, 1] = 0
                        keys_press_count[0, 2] += 1
                        self.what_to_do[0, 2] = keys_press_count[0, 2] % 2
                        if self.what_to_do[0, 2]:
                            now_time = datetime.datetime.now()
                            self.dt_time[0] = int(now_time.strftime("%Y%m%d%H%M%S"))
                            self.button_event.emit(0, 1, f"录像开始 [{self.dt_time[0]}]")
                        else:
                            self.button_event.emit(0, 1, "录像停止")

                last_keys_status = now_keys.copy()
                self.status_update.emit(self.what_to_do.tolist())
                time.sleep(0.01)

            except Exception as e:
                self.error.emit(str(e))
                time.sleep(0.1)


# ── Step 5: Collection Main Loop ─────────────────────────────────

class CollectionMainWorker(QThread):
    """Main data collection loop (replicates run_control.py main())."""
    status_update = pyqtSignal(dict)       # collection state fields
    frame_saved = pyqtSignal(int)          # frame count
    error = pyqtSignal(str)
    finished = pyqtSignal()
    log_message = pyqtSignal(str)
    light_changed = pyqtSignal(str)        # red/yellow/green
    button_event = pyqtSignal(int, int, str)  # hand, button, desc
    button_status = pyqtSignal(list)          # what_to_do flattened

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self.config = config  # save_data_path, project_name, what_to_do, dt_time,
                              # is_falling, img_list, npy_list, npy_len_list
        self._running = False

    def stop(self):
        self._running = False

    def run(self):
        try:
            from dobot_control.agents.dobot_agent import DobotAgent
            from dobot_control.agents.agent import BimanualAgent
            from dobot_control.robots.robot_node import ZMQClientRobot
            from dobot_control.env import RobotEnv
            from scripts.manipulate_utils import (
                load_ini_data_hands, robot_pose_init,
                dynamic_approach, servo_action_check, set_light,
            )
            from scripts.format_obs import save_frame
            from scripts.function_util import mk_dir, wait_period
            import requests

            what_to_do = self.config["what_to_do"]
            dt_time = self.config["dt_time"]
            is_falling = self.config["is_falling"]
            img_list = self.config["img_list"]
            save_data_path = self.config["save_data_path"]
            project_name = self.config["project_name"]
            robot_port = self.config.get("robot_port", 6002)
            hostname = self.config.get("hostname", "127.0.0.1")

            # Init agents (reuse from TeleopWorker if provided)
            agent = self.config.get("agent")
            if agent is not None:
                self.log_message.emit("使用已初始化的主手...")
            else:
                self.log_message.emit("初始化主手...")
                _, hands_dict = load_ini_data_hands()
                left_agent = DobotAgent(which_hand="LEFT", dobot_config=hands_dict["HAND_LEFT"])
                right_agent = DobotAgent(which_hand="RIGHT", dobot_config=hands_dict["HAND_RIGHT"])
                agent = BimanualAgent(left_agent, right_agent)

            # Start button monitor thread (internal, like run_control.py)
            import threading as thr
            import datetime as dt_mod

            def _button_monitor():
                """Internal button monitor — identical to run_control.py logic."""
                last_keys = np.zeros((2, 3), dtype=int)
                press_start = np.zeros((2, 2), dtype=int)
                press_count = np.zeros((2, 3), dtype=int)
                tic = np.zeros((2, 2))
                while self._running and not is_falling[0]:
                    try:
                        now_keys = agent.get_keys().astype(int)
                        dev = now_keys - last_keys
                        # Button A
                        for i in range(2):
                            if dev[i, 0] == -1:
                                tic[i, 0] = time.time()
                                press_start[i, 0] = 1
                            if dev[i, 0] == 1 and press_start[i, 0]:
                                press_start[i, 0] = 0
                                held = time.time() - tic[i, 0]
                                if held < 0.5:
                                    press_count[i, 0] += 1
                                    what_to_do[i, 0] = press_count[i, 0] % 2
                                    label = "解锁" if what_to_do[i, 0] else "锁定"
                                    hand = "左" if i == 0 else "右"
                                    self.button_event.emit(i, 0, f"{hand}手短按A → {label}")
                                elif held > 1:
                                    press_count[i, 1] += 1
                                    what_to_do[i, 1] = press_count[i, 1] % 2
                                    label = "伺服ON" if what_to_do[i, 1] else "伺服OFF"
                                    hand = "左" if i == 0 else "右"
                                    self.button_event.emit(i, 0, f"{hand}手长按A → {label}")
                        # Button B
                        for i in range(2):
                            if dev[i, 1] == -1:
                                tic[i, 1] = time.time()
                                press_start[i, 1] = 1
                            if dev[i, 1] == 1 and press_start[i, 1]:
                                press_start[i, 1] = 0
                                press_count[0, 2] += 1
                                what_to_do[0, 2] = press_count[0, 2] % 2
                                if what_to_do[0, 2]:
                                    now = dt_mod.datetime.now()
                                    dt_time[0] = int(now.strftime("%Y%m%d%H%M%S"))
                                    self.button_event.emit(0, 1, f"录像开始 [{dt_time[0]}]")
                                else:
                                    self.button_event.emit(0, 1, "录像停止")
                        last_keys = now_keys.copy()
                        self.button_status.emit(what_to_do.tolist())
                        time.sleep(0.01)
                    except Exception as e:
                        self.log_message.emit(f"按键读取错误: {e}")
                        time.sleep(0.1)

            btn_thread = thr.Thread(target=_button_monitor, daemon=True)
            btn_thread.start()
            self.log_message.emit("按键监听线程已启动")

            # Connect to robot
            self.log_message.emit("连接机器人...")
            robot_client = ZMQClientRobot(port=robot_port, host=hostname)

            # Check firmware
            try:
                r = requests.post("http://192.168.5.1:22000/settings/version", timeout=3)
                if r.status_code == 200:
                    self.log_message.emit("固件版本检查通过")
            except Exception:
                self.log_message.emit("固件检查跳过")

            # Get robot type
            robot_type = "Nova 2"
            try:
                r = requests.post("http://192.168.5.1:22000/properties/controllerType", timeout=3)
                if r.status_code == 200:
                    robot_type = eval(r.text)["name"]
                    self.log_message.emit(f"机器人型号: {robot_type}")
            except Exception:
                pass

            env = RobotEnv(robot_client)
            env.set_do_status([1, 0])
            env.set_do_status([2, 0])
            env.set_do_status([3, 0])

            self.log_message.emit("执行姿态初始化...")
            robot_pose_init(env)
            self.log_message.emit("初始化完成，等待按键操作...")

            last_status = np.zeros((2, 3), dtype=int)
            start_servo = False
            last_action = None
            idx = 0
            safe_limit = 0
            total_time = 0.04

            save_dir = os.path.join(save_data_path, project_name, "collect_data_scene1")
            mk_dir(save_dir)

            self._running = True
            self.status_update.emit({"state": "ready", "msg": "等待操作..."})

            while self._running and not is_falling[0]:
                tic = time.time()

                action = agent.act({})
                dev_what_to_do = what_to_do.copy() - last_status
                last_status = what_to_do.copy()

                # Lock/unlock
                for i in range(2):
                    if dev_what_to_do[i, 0] != 0:
                        agent.set_torque(i, not what_to_do[i, 0])

                # Servo ON
                if dev_what_to_do[0, 1] == 1 or dev_what_to_do[1, 1] == 1:
                    self.log_message.emit("dynamic approach...")
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
                    set_light(env, "yellow", 1)
                    self.light_changed.emit("yellow")

                # Servo OFF
                if dev_what_to_do[0, 1] == -1 or dev_what_to_do[1, 1] == -1:
                    self.log_message.emit("servo OFF")
                    for i in range(2):
                        if dev_what_to_do[i, 1] == -1:
                            agent.set_torque(i, False)
                    if what_to_do[0, 1] == 0 and what_to_do[1, 1] == 0:
                        set_light(env, "green", 0)
                        self.light_changed.emit("dark")

                # Servo tracking
                if (what_to_do[0, 1] or what_to_do[1, 1]) and start_servo:
                    action = agent.act({})
                    flag_in = np.array([what_to_do[0, 1], what_to_do[1, 1]])
                    err3, action = servo_action_check(action, last_action, flag_in)
                    if err3 != 1:
                        set_light(env, "red", 1)
                        self.light_changed.emit("red")
                        self.error.emit("Servo action check failed")
                        break

                    # Record
                    if dev_what_to_do[0, 2] == 1:
                        set_light(env, "green", 1)
                        self.light_changed.emit("green")
                    elif dev_what_to_do[0, 2] == -1:
                        set_light(env, "yellow", 1)
                        self.light_changed.emit("yellow")

                    if what_to_do[0, 2] == 1:
                        idx += 1
                        ts = str(dt_time[0])
                        left_dir = os.path.join(save_dir, ts, "leftImg")
                        right_dir = os.path.join(save_dir, ts, "rightImg")
                        top_dir = os.path.join(save_dir, ts, "topImg")
                        mk_dir(right_dir)
                        mk_dir(top_dir)
                        if mk_dir(left_dir):
                            idx = 0
                        cv2 = __import__("cv2")
                        cv2.imwrite(os.path.join(top_dir, f"{idx}.jpg"), img_list[0])
                        cv2.imwrite(os.path.join(left_dir, f"{idx}.jpg"), img_list[1])
                        cv2.imwrite(os.path.join(right_dir, f"{idx}.jpg"), img_list[2])

                        obs_dir = os.path.join(save_dir, ts, "observation")
                        mk_dir(obs_dir)
                        save_frame(obs_dir, idx, obs, action)
                        self.frame_saved.emit(idx)

                    obs = env.step(action, flag_in)
                    if "joint_positions" in obs:
                        obs["joint_positions"][6] = action[6]
                        obs["joint_positions"][13] = action[13]
                    last_action = action
                else:
                    start_servo = False
                    safe_limit = 0

                total_time = time.time() - tic
                if total_time < 0.04:
                    time.sleep(0.04 - total_time)

        except Exception as e:
            self.error.emit(str(e))
            traceback.print_exc()
        finally:
            self.log_message.emit("采集线程已退出")
            self.finished.emit()


# ── Step 6: Dataset Scan ─────────────────────────────────────────

class DatasetScanWorker(QThread):
    """Scan collected datasets."""
    datasets_found = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, data_path, parent=None):
        super().__init__(parent)
        self.data_path = data_path

    def run(self):
        try:
            collect_dir = os.path.join(self.data_path, "collect_data_scene1")
            if not os.path.isdir(collect_dir):
                self.datasets_found.emit([])
                return
            dirs = sorted(os.listdir(collect_dir))
            result = []
            for d in dirs:
                p = os.path.join(collect_dir, d)
                if os.path.isdir(p):
                    obs = os.path.join(p, "observation")
                    count = len([f for f in os.listdir(obs) if f.endswith(".pkl")]) if os.path.isdir(obs) else 0
                    result.append({"name": d, "path": p, "frames": count})
            self.datasets_found.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class FrameCountWorker(QThread):
    """Count frames in dataset."""
    count_complete = pyqtSignal(int, int)  # total, max_per_episode
    error = pyqtSignal(str)

    def __init__(self, data_path, parent=None):
        super().__init__(parent)
        self.data_path = data_path

    def run(self):
        try:
            collect_dir = os.path.join(self.data_path, "collect_data_scene1")
            if not os.path.isdir(collect_dir):
                self.count_complete.emit(0, 0)
                return
            total = 0
            max_frames = 0
            for d in sorted(os.listdir(collect_dir)):
                obs = os.path.join(collect_dir, d, "observation")
                if os.path.isdir(obs):
                    count = len([f for f in os.listdir(obs) if f.endswith(".pkl")])
                    total += count
                    if count > max_frames:
                        max_frames = count
            self.count_complete.emit(total, max_frames)
        except Exception as e:
            self.error.emit(str(e))


class HDF5ConversionWorker(QThread):
    """Convert raw data to HDF5 (replicates script_collect2train.py)."""
    progress = pyqtSignal(int, int)         # current, total
    conversion_complete = pyqtSignal(str)   # output path
    log_message = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, root_dir, dataset_name, episodes, parent=None):
        super().__init__(parent)
        self.root_dir = root_dir
        self.dataset_name = dataset_name
        self.episodes = episodes

    def run(self):
        try:
            import glob
            import pickle
            import cv2
            import h5py

            collect_dir = os.path.join(self.root_dir, self.dataset_name, "collect_data_scene1")
            train_dir = os.path.join(self.root_dir, self.dataset_name, "train_data")
            os.makedirs(train_dir, exist_ok=True)

            for ep_idx, ep in enumerate(self.episodes):
                ep_path = os.path.join(collect_dir, ep["name"])
                obs_dir = os.path.join(ep_path, "observation")
                top_dir = os.path.join(ep_path, "topImg")
                left_dir = os.path.join(ep_path, "leftImg")
                right_dir = os.path.join(ep_path, "rightImg")

                pkl_files = sorted(glob.glob(os.path.join(obs_dir, "*.pkl")))
                if not pkl_files:
                    continue

                qpos_list, action_list, img_dict = [], [], {"top": [], "left": [], "right": []}

                for pkl_f in pkl_files:
                    try:
                        with open(pkl_f, "rb") as f:
                            data = pickle.load(f)
                        qpos = data.get("joint_positions", np.zeros(14))
                        action = data.get("control", np.zeros(14))
                        qpos_list.append(qpos)
                        action_list.append(action)
                    except Exception:
                        continue

                n = len(qpos_list)
                # Load images
                for cam_name, cam_dir in [("top", top_dir), ("left", left_dir), ("right", right_dir)]:
                    imgs = []
                    for i in range(1, n + 1):
                        impath = os.path.join(cam_dir, f"{i}.jpg")
                        if os.path.isfile(impath):
                            img = cv2.imread(impath)
                            if img is not None:
                                _, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
                                imgs.append(enc.tobytes())
                            else:
                                imgs.append(b"")
                        else:
                            imgs.append(b"")
                    img_dict[cam_name] = imgs

                # Write HDF5
                h5_path = os.path.join(train_dir, f"episode_{ep_idx}.hdf5")
                with h5py.File(h5_path, "w") as hf:
                    obs_grp = hf.create_group("observations")
                    obs_grp.create_dataset("qpos", data=np.array(qpos_list))
                    obs_grp.create_dataset("qvel", data=np.zeros_like(np.array(qpos_list)))
                    img_grp = obs_grp.create_group("images")
                    for cam in ["top", "left", "right"]:
                        img_grp.create_dataset(cam, data=np.array(img_dict[cam]))
                    hf.create_dataset("action", data=np.array(action_list))

                self.log_message.emit(f"已转换 episode {ep_idx}: {ep['name']} ({n} 帧) -> {h5_path}")
                self.progress.emit(ep_idx + 1, len(self.episodes))

            self.conversion_complete.emit(train_dir)
        except Exception as e:
            self.error.emit(str(e))
            traceback.print_exc()

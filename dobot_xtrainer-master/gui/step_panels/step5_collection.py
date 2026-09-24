"""Step 5: Data collection (core functionality). Two-phase: connect → collect."""
import os
import time
import threading
import datetime as dt_mod
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QTextEdit, QLabel, QLineEdit, QGroupBox, QGridLayout,
    QMessageBox,
)
from PyQt6.QtCore import QThread, pyqtSignal

from gui.camera_widget import CameraWidget
from gui.workers import CameraStreamWorker, CollectionMainWorker


# ── Phase 1: Teleoperation test (no cameras, no recording) ──────

class TeleopWorker(QThread):
    """Initialize master hands + button monitor, allow free teleop testing."""
    log_message = pyqtSignal(str)
    button_event = pyqtSignal(int, int, str)   # hand, button, desc
    button_status = pyqtSignal(list)           # what_to_do flattened
    connected = pyqtSignal(object)             # agent instance
    error = pyqtSignal(str)

    def __init__(self, what_to_do, dt_time, is_falling, parent=None):
        super().__init__(parent)
        self.what_to_do = what_to_do
        self.dt_time = dt_time
        self.is_falling = is_falling
        self._running = False

    def stop(self):
        self._running = False
        # Close serial ports to release them for other steps
        for agent_attr in ['_left_agent', '_right_agent']:
            try:
                ag = getattr(self, agent_attr, None)
                if ag and hasattr(ag, '_robot') and hasattr(ag._robot, '_driver'):
                    ag._robot._driver.close()
            except Exception:
                pass

    def run(self):
        try:
            from dobot_control.agents.dobot_agent import DobotAgent
            from dobot_control.agents.agent import BimanualAgent
            from scripts.manipulate_utils import load_ini_data_hands
            from scripts.function_util import scan_port

            # Set port permissions first
            scan_port()

            self.log_message.emit("正在初始化主手...")
            _, hands_dict = load_ini_data_hands()
            self._left_agent = DobotAgent(which_hand="LEFT", dobot_config=hands_dict["HAND_LEFT"])
            self._right_agent = DobotAgent(which_hand="RIGHT", dobot_config=hands_dict["HAND_RIGHT"])
            agent = BimanualAgent(self._left_agent, self._right_agent)
            self.log_message.emit("主手连接成功 ✅")
            self.log_message.emit("现在可以操作主手测试遥操作：短按A=锁定/解锁，长按A=舵机跟随")

            self.connected.emit(agent)
            self._running = True

            # Button monitor (same logic as run_control.py)
            last_keys = np.zeros((2, 3), dtype=int)
            press_start = np.zeros((2, 2), dtype=int)
            press_count = np.zeros((2, 3), dtype=int)
            tic = np.zeros((2, 2))

            while self._running and not self.is_falling[0]:
                try:
                    now_keys = agent.get_keys().astype(int)
                    dev = now_keys - last_keys

                    for i in range(2):
                        if dev[i, 0] == -1:  # A pressed
                            tic[i, 0] = time.time()
                            press_start[i, 0] = 1
                        if dev[i, 0] == 1 and press_start[i, 0]:  # A released
                            press_start[i, 0] = 0
                            held = time.time() - tic[i, 0]
                            hand = "左" if i == 0 else "右"
                            if held < 0.5:
                                press_count[i, 0] += 1
                                self.what_to_do[i, 0] = press_count[i, 0] % 2
                                label = "解锁" if self.what_to_do[i, 0] else "锁定"
                                agent.set_torque(i, not self.what_to_do[i, 0])
                                self.button_event.emit(i, 0, f"{hand}手短按A → {label}")
                            elif held > 1:
                                press_count[i, 1] += 1
                                self.what_to_do[i, 1] = press_count[i, 1] % 2
                                label = "伺服ON" if self.what_to_do[i, 1] else "伺服OFF"
                                agent.set_torque(i, self.what_to_do[i, 1])
                                self.button_event.emit(i, 0, f"{hand}手长按A → {label}")

                    for i in range(2):
                        if dev[i, 1] == -1:  # B pressed
                            tic[i, 1] = time.time()
                            press_start[i, 1] = 1
                        if dev[i, 1] == 1 and press_start[i, 1]:  # B released
                            press_start[i, 1] = 0
                            press_count[0, 2] += 1
                            self.what_to_do[0, 2] = press_count[0, 2] % 2
                            if self.what_to_do[0, 2]:
                                now = dt_mod.datetime.now()
                                self.dt_time[0] = int(now.strftime("%Y%m%d%H%M%S"))
                                self.button_event.emit(0, 1, f"录像开始 [{self.dt_time[0]}]")
                            else:
                                self.button_event.emit(0, 1, "录像停止")

                    last_keys = now_keys.copy()
                    self.button_status.emit(self.what_to_do.tolist())
                    time.sleep(0.01)

                except Exception as e:
                    self.log_message.emit(f"按键读取错误: {e}")
                    time.sleep(0.1)

        except Exception as e:
            self.error.emit(str(e))
        finally:
            # Always release serial ports when exiting
            self.stop()


class Step5Collection(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.mw = main_window

        # Workers
        self._teleop_worker = None    # Phase 1
        self._cam_worker = None       # Phase 2
        self._coll_worker = None      # Phase 2
        self._agent = None            # Shared agent from Phase 1

        # States
        self._connected = False       # Phase 1 done
        self._collecting = False      # Phase 2 active
        self._frame_count = 0

        # Shared arrays
        self.what_to_do = np.zeros((2, 3), dtype=int)
        self.dt_time = np.array([20240701000000])
        self.is_falling = np.array([0])
        self.img_list = [
            np.zeros((480, 640, 3), dtype=np.uint8),
            np.zeros((480, 640, 3), dtype=np.uint8),
            np.zeros((480, 640, 3), dtype=np.uint8),
        ]
        self.npy_list = [np.zeros(480 * 640 * 3), np.zeros(480 * 640 * 3), np.zeros(480 * 640 * 3)]
        self.npy_len_list = np.array([0, 0, 0])

        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        title = QLabel("步骤 5: 遥操作 & 数据采集")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #cdd6f4;")
        layout.addWidget(title)

        # Project name
        proj_row = QHBoxLayout()
        proj_row.addWidget(QLabel("项目名称:"))
        self.proj_input = QLineEdit("plug-and-unplug-task")
        self.proj_input.setStyleSheet(
            "background-color: #181825; color: #cdd6f4; border: 1px solid #45475a; "
            "padding: 4px; border-radius: 4px;"
        )
        self.proj_input.setMaximumWidth(300)
        proj_row.addWidget(self.proj_input)
        proj_row.addStretch()
        layout.addLayout(proj_row)

        # ── Phase 1: Teleop ──
        group1 = QGroupBox("阶段 1: 连接主手（遥操作测试）")
        group1.setStyleSheet(self._group_style())
        g1 = QVBoxLayout()

        g1_desc = QLabel("先连接主手，可以自由测试锁定/解锁和舵机跟随，无需启动相机和录制。")
        g1_desc.setStyleSheet("color: #a6adc8; font-size: 11px;")
        g1.addWidget(g1_desc)

        btn1_row = QHBoxLayout()
        self.connect_btn = QPushButton("🔌 连接主手")
        self.connect_btn.setMinimumHeight(38)
        self.connect_btn.clicked.connect(self._toggle_connect)
        self.connect_btn.setStyleSheet(self._btn_style())
        btn1_row.addWidget(self.connect_btn)

        self.teleop_status = QLabel("未连接")
        self.teleop_status.setStyleSheet("color: #f38ba8; font-weight: bold;")
        btn1_row.addWidget(self.teleop_status)
        btn1_row.addStretch()
        g1.addLayout(btn1_row)

        group1.setLayout(g1)
        layout.addWidget(group1)

        # ── Status indicators ──
        status_group = QGroupBox("双手状态")
        status_group.setStyleSheet(self._group_style())
        grid = QGridLayout()
        for i, name in enumerate(["左手", "右手"]):
            grid.addWidget(QLabel(f"{name}:"), i, 0)
            lock = QLabel("🔒 锁定")
            lock.setStyleSheet("color: #f38ba8; font-weight: bold;")
            servo = QLabel("⚙ 未跟随")
            servo.setStyleSheet("color: #6c7086;")
            grid.addWidget(lock, i, 1)
            grid.addWidget(servo, i, 2)
            setattr(self, f"lock_{i}", lock)
            setattr(self, f"servo_{i}", servo)

        self.rec_label = QLabel("⏹ 未录制")
        self.rec_label.setStyleSheet("color: #f38ba8; font-weight: bold;")
        grid.addWidget(QLabel("录制状态:"), 2, 0)
        grid.addWidget(self.rec_label, 2, 1)

        self.frame_label = QLabel("帧数: 0")
        self.frame_label.setStyleSheet("color: #a6adc8;")
        grid.addWidget(QLabel("帧计数:"), 3, 0)
        grid.addWidget(self.frame_label, 3, 1)

        self.light_label = QLabel("💡 灭")
        self.light_label.setStyleSheet("color: #6c7086;")
        grid.addWidget(QLabel("指示灯:"), 4, 0)
        grid.addWidget(self.light_label, 4, 1)

        status_group.setLayout(grid)
        layout.addWidget(status_group)

        # ── Phase 2: Collection ──
        group2 = QGroupBox("阶段 2: 数据采集")
        group2.setStyleSheet(self._group_style())
        g2 = QVBoxLayout()

        g2_desc = QLabel("连接主手后就绪。开始采集后启动相机、连接从臂、触发录制。")
        g2_desc.setStyleSheet("color: #a6adc8; font-size: 11px;")
        g2.addWidget(g2_desc)

        btn2_row = QHBoxLayout()
        self.start_btn = QPushButton("▶ 开始采集")
        self.start_btn.setMinimumHeight(40)
        self.start_btn.setEnabled(False)
        self.start_btn.clicked.connect(self._toggle_collection)
        self.start_btn.setStyleSheet(self._btn_style_green())
        btn2_row.addWidget(self.start_btn)

        self.stop_btn = QPushButton("⏹ 紧急停止")
        self.stop_btn.setMinimumHeight(40)
        self.stop_btn.clicked.connect(self._emergency_stop)
        self.stop_btn.setStyleSheet(self._btn_style_red())
        self.stop_btn.setEnabled(False)
        btn2_row.addWidget(self.stop_btn)
        btn2_row.addStretch()
        g2.addLayout(btn2_row)

        group2.setLayout(g2)
        layout.addWidget(group2)

        # Camera preview
        self.camera_widget = CameraWidget(960, 240)
        layout.addWidget(self.camera_widget)

        # Log
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(130)
        self.log.setStyleSheet("background-color: #11111b; color: #a6e3a1; font-family: monospace;")
        layout.addWidget(self.log)

        hint = QLabel("操作说明: ①连接主手 → ②自由遥操作测试 → ③按「开始采集」进入录制模式")
        hint.setStyleSheet("color: #585b70; font-size: 11px;")
        layout.addWidget(hint)

        self.setLayout(layout)

    # ── Styles ──
    def _btn_style(self):
        return """
            QPushButton {background-color: #45475a; color: #cdd6f4; border: 1px solid #585b70;
                border-radius: 6px; padding: 6px 20px; font-size: 13px;}
            QPushButton:hover {background-color: #585b70;}
            QPushButton:disabled {background-color: #313244; color: #6c7086;}
        """
    def _btn_style_green(self):
        return """
            QPushButton {background-color: #2a6b3f; color: #cdd6f4; border: 1px solid #3a8b4f;
                border-radius: 6px; padding: 6px 20px; font-size: 13px;}
            QPushButton:hover {background-color: #3a8b4f;}
            QPushButton:disabled {background-color: #313244; color: #6c7086;}
        """
    def _btn_style_red(self):
        return """
            QPushButton {background-color: #7a3030; color: #cdd6f4; border: 1px solid #b04040;
                border-radius: 6px; padding: 6px 20px; font-size: 13px;}
            QPushButton:hover {background-color: #b04040;}
            QPushButton:disabled {background-color: #313244; color: #6c7086;}
        """
    def _group_style(self):
        return """
            QGroupBox {color: #cdd6f4; font-weight: bold; border: 1px solid #45475a;
                border-radius: 8px; margin-top: 10px; padding: 12px;}
        """

    def append_log(self, msg: str):
        self.log.append(msg)

    def on_enter(self):
        pass

    # ── Phase 1: Connect ──
    def _toggle_connect(self):
        if self._connected:
            self._disconnect_teleop()
        else:
            self._connect_teleop()

    def _connect_teleop(self):
        self.connect_btn.setEnabled(False)
        self.connect_btn.setText("连接中...")
        self.append_log("正在连接主手...")
        self.what_to_do[:, :] = 0
        self.is_falling[0] = 0

        self._teleop_worker = TeleopWorker(self.what_to_do, self.dt_time, self.is_falling)
        self._teleop_worker.log_message.connect(self.append_log)
        self._teleop_worker.button_event.connect(self._on_button_event)
        self._teleop_worker.button_status.connect(self._on_button_status)
        self._teleop_worker.connected.connect(self._on_teleop_connected)
        self._teleop_worker.error.connect(lambda e: self.append_log(f"❌ {e}"))
        self._teleop_worker.finished.connect(self._on_teleop_finished)
        self._teleop_worker.start()

    def _on_teleop_connected(self, agent):
        self._agent = agent
        self._connected = True
        self.connect_btn.setText("🔌 断开主手")
        self.connect_btn.setEnabled(True)
        self.teleop_status.setText("✅ 已连接 · 可自由操作")
        self.teleop_status.setStyleSheet("color: #a6e3a1; font-weight: bold;")
        self.start_btn.setEnabled(True)  # unlock Phase 2
        self.mw.update_status("left_hand", "online")
        self.mw.update_status("right_hand", "online")
        self.append_log("主手已就绪，请短按A解锁后长按A测试遥操作")

    def _disconnect_teleop(self):
        self.append_log("正在断开主手...")
        if self._teleop_worker:
            self._teleop_worker.stop()
            self._teleop_worker.quit()
            self._teleop_worker.wait(3000)
            self._teleop_worker = None
        self._connected = False
        self._agent = None
        self.connect_btn.setText("🔌 连接主手")
        self.teleop_status.setText("未连接")
        self.teleop_status.setStyleSheet("color: #f38ba8; font-weight: bold;")
        self.start_btn.setEnabled(False)
        self.mw.update_status("left_hand", "offline")
        self.mw.update_status("right_hand", "offline")

    def _on_teleop_finished(self):
        if self._connected:
            self._disconnect_teleop()

    # ── Phase 2: Collection ──
    def _toggle_collection(self):
        if self._collecting:
            self._stop_collection()
        else:
            self._start_collection()

    def _start_collection(self):
        if not self.mw.state.get("server_running"):
            QMessageBox.warning(self, "提示", "请先在步骤 4 启动 ZMQ 服务！")
            return
        if not self._connected:
            QMessageBox.warning(self, "提示", "请先点击「连接主手」！")
            return

        project_name = self.proj_input.text().strip() or "plug-and-unplug-task"
        self.mw.state["project_name"] = project_name
        self.is_falling[0] = 0
        self._frame_count = 0

        # Start camera
        self.append_log("启动相机...")
        camera_dict = self.mw.state.get("camera_dict")
        if camera_dict is None:
            from scripts.manipulate_utils import load_ini_data_camera
            camera_dict = load_ini_data_camera()

        self._cam_worker = CameraStreamWorker(camera_dict)
        self._cam_worker.frame_ready.connect(self.camera_widget.update_frame)
        self._cam_worker.camera_error.connect(lambda n, m: self.append_log(f"[{n}] {m}"))
        self._cam_worker.start()
        self.camera_widget.start(30)
        self.mw.update_status("camera", "online")

        # Start collection worker
        config = {
            "save_data_path": self.mw.state["save_data_path"],
            "project_name": project_name,
            "what_to_do": self.what_to_do,
            "dt_time": self.dt_time,
            "is_falling": self.is_falling,
            "img_list": self.img_list,
            "npy_list": self.npy_list,
            "npy_len_list": self.npy_len_list,
            "robot_port": 6002,
            "hostname": "127.0.0.1",
            "agent": self._agent,  # Pass the already-initialized agent!
        }

        self._coll_worker = CollectionMainWorker(config)
        self._coll_worker.log_message.connect(self.append_log)
        self._coll_worker.frame_saved.connect(self._on_frame)
        self._coll_worker.light_changed.connect(self._on_light)
        self._coll_worker.error.connect(lambda e: self.append_log(f"❌ {e}"))
        self._coll_worker.finished.connect(self._on_collection_finished)
        self._coll_worker.start()

        self.mw.update_status("recording", "yellow")
        self._collecting = True
        self.start_btn.setText("⏹ 停止采集")
        self.stop_btn.setEnabled(True)
        self.connect_btn.setEnabled(False)  # can't disconnect while collecting
        self.append_log(f"数据采集已启动 - 项目: {project_name}")

    def _stop_collection(self):
        self.append_log("正在停止采集...")
        self.is_falling[0] = 1

        if self._coll_worker:
            self._coll_worker.stop()
            self._coll_worker.quit()
            self._coll_worker.wait(5000)
            self._coll_worker = None

        if self._cam_worker:
            self._cam_worker.stop()
            self._cam_worker.quit()
            self._cam_worker.wait(3000)
            self._cam_worker = None

        self.camera_widget.stop()
        self._cleanup()

    def _emergency_stop(self):
        self.append_log("⚠ 紧急停止！")
        self.is_falling[0] = 1
        if self._coll_worker:
            self._stop_collection()
        self.what_to_do[:, :] = 0
        self._on_button_status([[0, 0, 0], [0, 0, 0]])

    def _cleanup(self):
        self._collecting = False
        self.start_btn.setText("▶ 开始采集")
        self.stop_btn.setEnabled(False)
        self.connect_btn.setEnabled(True)
        self.mw.update_status("recording", "offline")
        self.mw.update_status("camera", "offline")
        self._coll_worker = None
        self._cam_worker = None

    # ── Button status UI update ──
    def _on_button_event(self, hand, btn, desc):
        self.append_log(f"[按键] {desc}")

    def _on_button_status(self, status_list):
        if len(status_list) < 2:
            return
        for i in range(2):
            what = status_list[i]
            lock = getattr(self, f"lock_{i}", None)
            servo = getattr(self, f"servo_{i}", None)
            if lock:
                if what[0]:
                    lock.setText("🔓 解锁")
                    lock.setStyleSheet("color: #a6e3a1; font-weight: bold;")
                else:
                    lock.setText("🔒 锁定")
                    lock.setStyleSheet("color: #f38ba8; font-weight: bold;")
            if servo:
                if what[1]:
                    servo.setText("⚙ 跟随中")
                    servo.setStyleSheet("color: #a6e3a1; font-weight: bold;")
                else:
                    servo.setText("⚙ 未跟随")
                    servo.setStyleSheet("color: #6c7086;")
        if len(status_list[0]) > 2:
            if status_list[0][2]:
                self.rec_label.setText("⏺ 录制中")
                self.rec_label.setStyleSheet("color: #a6e3a1; font-weight: bold;")
            else:
                self.rec_label.setText("⏹ 未录制")
                self.rec_label.setStyleSheet("color: #f38ba8; font-weight: bold;")

    def _on_frame(self, count):
        self._frame_count = count
        self.frame_label.setText(f"帧数: {count}")

    def _on_light(self, color):
        labels = {"red": "💡 红灯 ⚠", "yellow": "💡 黄灯", "green": "💡 绿灯 🎬", "dark": "💡 灭"}
        self.light_label.setText(labels.get(color, f"💡 {color}"))
        color_map = {"red": "#f38ba8", "yellow": "#f9e2af", "green": "#a6e3a1", "dark": "#6c7086"}
        self.light_label.setStyleSheet(f"color: {color_map.get(color, '#6c7086')}; font-weight: bold;")
        self.mw.update_status("recording", "green" if color == "green" else "yellow" if color == "yellow" else "red" if color == "red" else "offline")

    def _on_collection_finished(self):
        self._cleanup()
        self.append_log("采集线程已退出")

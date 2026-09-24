"""Step 4: ZMQ Robot Server control."""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QTextEdit, QLabel, QGroupBox, QGridLayout, QMessageBox,
)
from PyQt6.QtCore import Qt

from gui.workers import ZMQServerWorker


class Step4Server(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.mw = main_window
        self._worker = None
        self._running = False

        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)

        title = QLabel("步骤 4: 启动 ZMQ 服务")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #cdd6f4;")
        layout.addWidget(title)

        desc = QLabel("连接左右从臂机器人并启动 ZMQ 服务，供数据采集程序调用。\n确保机器人已开机并连接网络 (192.168.5.1 / 192.168.5.2)。")
        desc.setStyleSheet("color: #a6adc8; font-size: 12px; margin-bottom: 8px;")
        layout.addWidget(desc)

        # Server status
        status_group = QGroupBox("服务状态")
        status_group.setStyleSheet(self._group_style())
        grid = QGridLayout()
        grid.setSpacing(10)

        self.server_label = QLabel("ZMQ 服务: 未启动")
        self.server_label.setStyleSheet("color: #f38ba8; font-weight: bold;")
        grid.addWidget(QLabel("ZMQ 服务:"), 0, 0)
        grid.addWidget(self.server_label, 0, 1)

        self.left_robot_label = QLabel("192.168.5.1 (左臂): 未连接")
        self.left_robot_label.setStyleSheet("color: #6c7086;")
        grid.addWidget(QLabel("左臂:"), 1, 0)
        grid.addWidget(self.left_robot_label, 1, 1)

        self.right_robot_label = QLabel("192.168.5.2 (右臂): 未连接")
        self.right_robot_label.setStyleSheet("color: #6c7086;")
        grid.addWidget(QLabel("右臂:"), 2, 0)
        grid.addWidget(self.right_robot_label, 2, 1)

        self.port_label = QLabel("端口: 6002")
        self.port_label.setStyleSheet("color: #a6adc8;")
        grid.addWidget(QLabel("端口:"), 3, 0)
        grid.addWidget(self.port_label, 3, 1)

        status_group.setLayout(grid)
        layout.addWidget(status_group)

        # Buttons
        btn_row = QHBoxLayout()
        self.start_btn = QPushButton("🚀 启动 ZMQ 服务")
        self.start_btn.setMinimumHeight(40)
        self.start_btn.clicked.connect(self._toggle)
        self.start_btn.setStyleSheet(self._btn_style())

        self.next_btn = QPushButton("✅ 继续")
        self.next_btn.setMinimumHeight(40)
        self.next_btn.setEnabled(False)
        self.next_btn.clicked.connect(self._go_next)
        self.next_btn.setStyleSheet(self._btn_style())

        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.next_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # Log
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setStyleSheet("background-color: #11111b; color: #a6e3a1; font-family: monospace;")
        layout.addWidget(self.log)

        self.setLayout(layout)

    def _btn_style(self):
        return """
            QPushButton {
                background-color: #45475a; color: #cdd6f4;
                border: 1px solid #585b70; border-radius: 6px;
                padding: 6px 20px; font-size: 13px;
            }
            QPushButton:hover { background-color: #585b70; }
            QPushButton:disabled { background-color: #313244; color: #6c7086; }
        """

    def _group_style(self):
        return """
            QGroupBox {
                color: #cdd6f4; font-weight: bold;
                border: 1px solid #45475a; border-radius: 8px;
                margin-top: 10px; padding: 12px;
            }
        """

    def append_log(self, msg: str):
        self.log.append(msg)

    def on_enter(self):
        pass

    def _toggle(self):
        if self._running:
            self._stop_server()
        else:
            self._start_server()

    def _start_server(self):
        self.start_btn.setEnabled(False)
        self.start_btn.setText("启动中...")
        self.append_log("正在启动 ZMQ 服务...")
        self.mw.update_status("left_robot", "offline")
        self.mw.update_status("right_robot", "offline")
        self.mw.update_status("zmq", "offline")

        self._worker = ZMQServerWorker(port=6002)
        self._worker.log_message.connect(self.append_log)
        self._worker.status_update.connect(self._on_status)
        self._worker.server_started.connect(self._on_started)
        self._worker.server_stopped.connect(self._on_stopped)
        self._worker.robot_error.connect(lambda ip, m: self.append_log(f"[{ip}] 错误: {m}"))
        self._worker.start()

    def _stop_server(self):
        self.start_btn.setEnabled(False)
        self.start_btn.setText("停止中...")
        self.append_log("正在停止 ZMQ 服务...")
        if self._worker:
            self._worker.stop_server()
            self._worker.quit()
            self._worker.wait(5000)
            self._worker = None

        self._running = False
        self._update_ui_stopped()

    def _on_status(self, key, status):
        self.mw.update_status(key, status)

    def _on_started(self):
        self._running = True
        self.start_btn.setText("⏹ 停止服务")
        self.start_btn.setEnabled(True)
        self.server_label.setText("ZMQ 服务: 运行中 ✅")
        self.server_label.setStyleSheet("color: #a6e3a1; font-weight: bold;")
        self.left_robot_label.setText("192.168.5.1 (左臂): 已连接 ✅")
        self.left_robot_label.setStyleSheet("color: #a6e3a1;")
        self.right_robot_label.setText("192.168.5.2 (右臂): 已连接 ✅")
        self.right_robot_label.setStyleSheet("color: #a6e3a1;")
        self.next_btn.setEnabled(True)
        self.mw.state["server_running"] = True
        self.mw.update_status("zmq", "online")
        self.mw.update_status("left_robot", "online")
        self.mw.update_status("right_robot", "online")

    def _on_stopped(self):
        self._running = False
        self._update_ui_stopped()

    def _update_ui_stopped(self):
        self.start_btn.setText("🚀 启动 ZMQ 服务")
        self.start_btn.setEnabled(True)
        self.server_label.setText("ZMQ 服务: 未启动")
        self.server_label.setStyleSheet("color: #f38ba8; font-weight: bold;")
        self.left_robot_label.setText("192.168.5.1 (左臂): 未连接")
        self.left_robot_label.setStyleSheet("color: #6c7086;")
        self.right_robot_label.setText("192.168.5.2 (右臂): 未连接")
        self.right_robot_label.setStyleSheet("color: #6c7086;")
        self.mw.state["server_running"] = False
        self.mw.update_status("zmq", "offline")
        self.mw.update_status("left_robot", "offline")
        self.mw.update_status("right_robot", "offline")

    def _go_next(self):
        if not self._running:
            QMessageBox.warning(self, "提示", "请先启动 ZMQ 服务！")
            return
        self.mw.enable_next_step()
        QMessageBox.information(self, "完成", "ZMQ 服务已启动，可以进入数据采集。")

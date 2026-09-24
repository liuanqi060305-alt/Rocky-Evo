"""Main window for the Dobot Xtrainer GUI."""
import sys
import os
import numpy as np

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QListWidget, QListWidgetItem, QStackedWidget,
    QApplication, QSplitter, QLabel,
)
from PyQt6.QtCore import Qt, QTimer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from gui.status_widgets import StatusBar
from gui.camera_widget import CameraWidget
from gui.step_panels.step1_port_setup import Step1PortSetup
from gui.step_panels.step2_offset_calib import Step2OffsetCalib
from gui.step_panels.step3_camera_preview import Step3CameraPreview
from gui.step_panels.step4_server import Step4Server
from gui.step_panels.step5_collection import Step5Collection
from gui.step_panels.step6_processing import Step6Processing


class MainWindow(QMainWindow):
    """Step-by-step wizard for Dobot Xtrainer."""

    def __init__(self):
        super().__init__()

        # ── Shared State ──
        self.state = {
            "ports_configured": False,
            "offsets_calibrated": False,
            "camera_ready": False,
            "server_running": False,
            "collection_active": False,
            "hands_connected": False,
            "robot_connected": False,
            "project_name": "plug-and-unplug-task",
            "save_data_path": os.path.join(
                os.path.dirname(BASE_DIR), "datasets"
            ),
            "connection_status": {
                "top_camera": False,
                "left_camera": False,
                "right_camera": False,
                "left_hand": False,
                "right_hand": False,
                "left_robot": False,
                "right_robot": False,
                "zmq": False,
            },
            "camera_instances": None,
            "camera_dict": None,
        }

        self.setWindowTitle("Dobot Xtrainer 控制面板")
        self.resize(1500, 920)

        # ── Central Widget ──
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Title bar
        title = QLabel("  Dobot Xtrainer 控制面板 v1.0")
        title.setStyleSheet(
            "background-color: #1e1e2e; color: #cdd6f4; font-size: 16px; "
            "font-weight: bold; padding: 8px;"
        )
        title.setFixedHeight(40)
        main_layout.addWidget(title)

        # Splitter: left steps + right panels
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: step list
        self.step_list = QListWidget()
        self.step_list.setFixedWidth(140)
        self.step_list.setStyleSheet("""
            QListWidget {
                background-color: #181825;
                border: none;
                font-size: 14px;
            }
            QListWidget::item {
                color: #6c7086;
                padding: 14px 8px;
                border-bottom: 1px solid #313244;
            }
            QListWidget::item:selected {
                color: #cdd6f4;
                background-color: #45475a;
                font-weight: bold;
            }
            QListWidget::item:disabled {
                color: #585b70;
            }
        """)

        steps = [
            "① 端口扫描",
            "② 偏移校准",
            "③ 相机预览",
            "④ 启动服务",
            "⑤ 数据采集",
            "⑥ 数据处理",
        ]
        for i, step in enumerate(steps):
            item = QListWidgetItem(step)
            if i > 0:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
            self.step_list.addItem(item)

        self.step_list.currentRowChanged.connect(self.on_step_changed)
        splitter.addWidget(self.step_list)

        # Right: stacked panels
        self.stack = QStackedWidget()
        self.stack.setStyleSheet("background-color: #1e1e2e;")
        self.panels = [
            Step1PortSetup(self),
            Step2OffsetCalib(self),
            Step3CameraPreview(self),
            Step4Server(self),
            Step5Collection(self),
            Step6Processing(self),
        ]
        for p in self.panels:
            self.stack.addWidget(p)

        splitter.addWidget(self.stack)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        main_layout.addWidget(splitter)

        # Status bar
        self.status_bar = StatusBar()
        main_layout.addWidget(self.status_bar)

        # Initial step
        self.step_list.setCurrentRow(0)

    def on_step_changed(self, index: int):
        if 0 <= index < len(self.panels):
            self.stack.setCurrentIndex(index)
            panel = self.panels[index]
            if hasattr(panel, "on_enter"):
                panel.on_enter()

    def enable_next_step(self):
        """Enable the next step in the list."""
        current = self.step_list.currentRow()
        if current + 1 < self.step_list.count():
            item = self.step_list.item(current + 1)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEnabled)
            self.step_list.setCurrentRow(current + 1)

    def update_status(self, key: str, status: str):
        """Update status bar indicator."""
        self.status_bar.update_status(key, status)

"""Step 3: Camera preview with 3-camera stitched view."""
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QMessageBox, QCheckBox,
)
from PyQt6.QtCore import QTimer

from gui.camera_widget import CameraWidget
from gui.workers import CameraStreamWorker


class Step3CameraPreview(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.mw = main_window
        self._worker = None
        self._previewing = False

        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)

        title = QLabel("步骤 3: 相机预览")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #cdd6f4;")
        layout.addWidget(title)

        desc = QLabel("启动相机预览，确认三路相机 (top/left/right) 画面正常。")
        desc.setStyleSheet("color: #a6adc8; font-size: 12px;")
        layout.addWidget(desc)

        # Controls
        ctrl_row = QHBoxLayout()
        self.toggle_btn = QPushButton("▶ 启动预览")
        self.toggle_btn.setMinimumHeight(36)
        self.toggle_btn.clicked.connect(self._toggle)
        self.toggle_btn.setStyleSheet(self._btn_style())
        ctrl_row.addWidget(self.toggle_btn)

        self.fps_label = QLabel("FPS: --")
        self.fps_label.setStyleSheet("color: #f9e2af; font-size: 13px;")
        ctrl_row.addWidget(self.fps_label)
        ctrl_row.addStretch()

        self.next_btn = QPushButton("✅ 继续")
        self.next_btn.setMinimumHeight(36)
        self.next_btn.clicked.connect(lambda: self.mw.enable_next_step())
        self.next_btn.setStyleSheet(self._btn_style())
        ctrl_row.addWidget(self.next_btn)
        layout.addLayout(ctrl_row)

        # Camera widget
        self.camera_widget = CameraWidget(1920, 480)
        layout.addWidget(self.camera_widget)

        # Checkboxes
        chk_row = QHBoxLayout()
        self.chk_top = QCheckBox("Top 相机")
        self.chk_left = QCheckBox("Left 相机")
        self.chk_right = QCheckBox("Right 相机")
        self.chk_top.setChecked(True)
        self.chk_left.setChecked(True)
        self.chk_right.setChecked(True)
        for chk in [self.chk_top, self.chk_left, self.chk_right]:
            chk.setStyleSheet("color: #cdd6f4;")
        chk_row.addWidget(self.chk_top)
        chk_row.addWidget(self.chk_left)
        chk_row.addWidget(self.chk_right)
        chk_row.addStretch()
        layout.addLayout(chk_row)

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

    def on_enter(self):
        pass

    def _toggle(self):
        if self._previewing:
            self._stop_preview()
        else:
            self._start_preview()

    def _start_preview(self):
        from scripts.manipulate_utils import load_ini_data_camera
        try:
            camera_dict = load_ini_data_camera()
        except Exception:
            camera_dict = {}
        self.mw.state["camera_dict"] = camera_dict

        self._worker = CameraStreamWorker(camera_dict)
        self._worker.frame_ready.connect(self.camera_widget.update_frame)
        self._worker.fps_update.connect(lambda f: self.fps_label.setText(f"FPS: {f:.1f}"))
        self._worker.camera_error.connect(self._on_cam_error)
        self._worker.start()

        self.camera_widget.start(30)
        self._previewing = True
        self.toggle_btn.setText("⏹ 停止预览")
        self.mw.state["camera_ready"] = True
        self.mw.update_status("camera", "online")

    def _stop_preview(self):
        self.camera_widget.stop()
        if self._worker:
            self._worker.stop()
            self._worker.quit()
            self._worker.wait(3000)
            self._worker = None
        self._previewing = False
        self.toggle_btn.setText("▶ 启动预览")
        self.fps_label.setText("FPS: --")
        self.mw.update_status("camera", "offline")

    def _on_cam_error(self, name, msg):
        pass  # Camera errors are non-critical

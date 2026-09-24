"""Camera display widget using QLabel + QTimer for OpenCV frame rendering."""
import cv2
import numpy as np
from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap


class CameraWidget(QWidget):
    """Displays a camera feed (single 640x480 or stitched 1920x480)."""

    def __init__(self, width=1920, height=480, parent=None):
        super().__init__(parent)
        self._width = width
        self._height = height
        self.label = QLabel()
        self.label.setFixedSize(width, height)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("background-color: #111; border: 1px solid #444;")
        self.label.setText("相机未启动")
        self._frame = np.zeros((height, width, 3), dtype=np.uint8)
        self.timer = QTimer()
        self.timer.timeout.connect(self._refresh)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.label)
        layout.addStretch()
        self.setLayout(layout)

    def update_frame(self, frame: np.ndarray):
        """Called from worker thread (via queued signal) — thread-safe."""
        if frame is not None and frame.shape[0] > 0:
            self._frame = frame.copy()

    def _refresh(self):
        """Render current frame via QPixmap (main thread)."""
        try:
            if self._frame is None or self._frame.size == 0:
                return
            h, w = self._frame.shape[:2]
            if h == 0 or w == 0:
                return
            rgb = cv2.cvtColor(self._frame, cv2.COLOR_BGR2RGB)
            bytes_per_line = 3 * w
            qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            scaled = qt_img.scaled(
                self._width, self._height,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.label.setPixmap(QPixmap.fromImage(scaled))
        except Exception:
            pass

    def start(self, fps=30):
        self.timer.start(1000 // fps)

    def stop(self):
        self.timer.stop()
        self.label.setText("相机已停止")
        self.label.setPixmap(QPixmap())
        self._frame = np.zeros((self._height, self._width, 3), dtype=np.uint8)

    def set_preview_text(self, text: str):
        self.label.setText(text)

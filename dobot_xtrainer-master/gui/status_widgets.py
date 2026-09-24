"""Status indicator widgets for the GUI status bar."""
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPainter, QBrush


class LedIndicator(QWidget):
    """A circular LED indicator: green=ok, red=error, gray=offline."""

    def __init__(self, label="", parent=None):
        super().__init__(parent)
        self._color = QColor(128, 128, 128)  # gray = offline
        self._label = label
        self.setFixedSize(16, 16)
        self.setToolTip(label)

    def set_ok(self):
        self._color = QColor(0, 200, 0)
        self.update()

    def set_error(self):
        self._color = QColor(220, 30, 30)
        self.update()

    def set_offline(self):
        self._color = QColor(128, 128, 128)
        self.update()

    def set_active(self, color_name: str):
        colors = {
            "green": QColor(0, 200, 0),
            "red": QColor(220, 30, 30),
            "yellow": QColor(220, 220, 0),
            "gray": QColor(128, 128, 128),
            "offline": QColor(128, 128, 128),
        }
        self._color = colors.get(color_name, QColor(128, 128, 128))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QBrush(self._color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(2, 2, 12, 12)


class StatusBar(QWidget):
    """Horizontal status bar with LED indicators and labels."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout()
        layout.setContentsMargins(8, 2, 8, 2)
        layout.setSpacing(6)

        self.indicators = {}

        items = [
            ("camera", "相机"),
            ("left_hand", "左手"),
            ("right_hand", "右手"),
            ("left_robot", "左臂"),
            ("right_robot", "右臂"),
            ("zmq", "ZMQ"),
            ("recording", "录像"),
        ]

        for key, text in items:
            led = LedIndicator(text)
            label = QLabel(text)
            label.setStyleSheet("font-size: 11px; color: #ccc;")
            layout.addWidget(led)
            layout.addWidget(label)
            layout.addSpacing(10)
            self.indicators[key] = led

        layout.addStretch()
        self.setLayout(layout)
        self.setStyleSheet("background-color: #2b2b2b; border-top: 1px solid #444;")
        self.setFixedHeight(32)
        self.set_all_offline()

    def set_all_offline(self):
        for led in self.indicators.values():
            led.set_offline()

    def update_status(self, key: str, status: str):
        """Update LED: status = 'online'|'error'|'offline'|'green'|'red'|'yellow'"""
        led = self.indicators.get(key)
        if not led:
            return
        if status in ("online", "green"):
            led.set_ok()
        elif status in ("error", "red"):
            led.set_error()
        elif status == "yellow":
            led.set_active("yellow")
        else:
            led.set_offline()

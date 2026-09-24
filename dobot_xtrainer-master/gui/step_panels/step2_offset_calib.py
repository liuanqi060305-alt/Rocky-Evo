"""Step 2: Joint offset calibration."""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QTableWidget, QTableWidgetItem, QTextEdit, QLabel,
    QHeaderView, QMessageBox,
)

from gui.workers import OffsetReadWorker


class Step2OffsetCalib(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.mw = main_window

        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)

        title = QLabel("步骤 2: 关节偏移校准")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #cdd6f4;")
        layout.addWidget(title)

        desc = QLabel(
            "请将双手主手置于标准起始姿态，然后点击「读取偏移」。\n"
            "程序会计算当前关节角度与预设起始角度的差值并保存。\n\n"
            "左手起始: [-90°, 0°, -90°, 0°, 90°, 90°]\n"
            "右手起始: [90°, 0°, 90°, 0°, -90°, -90°]"
        )
        desc.setStyleSheet("color: #a6adc8; font-size: 12px; margin-bottom: 8px;")
        layout.addWidget(desc)

        btn_row = QHBoxLayout()
        self.read_btn = QPushButton("📐 读取偏移")
        self.read_btn.setMinimumHeight(36)
        self.read_btn.clicked.connect(self._read_offsets)
        self.read_btn.setStyleSheet(self._btn_style())

        self.save_btn = QPushButton("✅ 保存并继续")
        self.save_btn.setMinimumHeight(36)
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._save_and_next)
        self.save_btn.setStyleSheet(self._btn_style())

        btn_row.addWidget(self.read_btn)
        btn_row.addWidget(self.save_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # Tables
        self.left_table = self._make_table("左手 (HAND_LEFT)")
        layout.addWidget(QLabel("左手 (HAND_LEFT)"))
        layout.addWidget(self.left_table)

        self.right_table = self._make_table("右手 (HAND_RIGHT)")
        layout.addWidget(QLabel("右手 (HAND_RIGHT)"))
        layout.addWidget(self.right_table)

        # Log
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(120)
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

    def _make_table(self, title):
        t = QTableWidget(6, 4)
        t.setHorizontalHeaderLabels(["关节", "当前角度(°)", "起始角度(°)", "偏移量"])
        t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        t.setStyleSheet("""
            QTableWidget {
                background-color: #181825; color: #cdd6f4;
                gridline-color: #313244; border: 1px solid #313244;
            }
            QHeaderView::section {
                background-color: #313244; color: #cdd6f4; padding: 4px;
                border: none; font-weight: bold;
            }
        """)
        t.setMaximumHeight(180)
        return t

    def append_log(self, msg: str):
        self.log.append(msg)

    def on_enter(self):
        pass

    def _read_offsets(self):
        self.read_btn.setEnabled(False)
        self.read_btn.setText("读取中...")
        self.append_log("正在读取关节偏移...")

        self.worker = OffsetReadWorker()
        self.worker.log_message.connect(self.append_log)
        self.worker.offsets_ready.connect(self._on_offsets)
        self.worker.error.connect(lambda e: self.append_log(f"错误: {e}"))
        self.worker.finished.connect(lambda: (
            self.read_btn.setEnabled(True),
            self.read_btn.setText("📐 读取偏移"),
        ))
        self.worker.start()

    def _on_offsets(self, data):
        for which, label, table in [
            ("HAND_LEFT", "左手", self.left_table),
            ("HAND_RIGHT", "右手", self.right_table),
        ]:
            d = data.get(which, {})
            curr = d.get("curr_joints", [0] * 6)
            offsets = d.get("offsets", [0] * 6)
            start = [-90, 0, -90, 0, 90, 90] if which == "HAND_LEFT" else [90, 0, 90, 0, -90, -90]
            for i in range(6):
                table.setItem(i, 0, QTableWidgetItem(f"J{i + 1}"))
                table.setItem(i, 1, QTableWidgetItem(f"{curr[i]:.1f}"))
                table.setItem(i, 2, QTableWidgetItem(f"{start[i]}"))
                table.setItem(i, 3, QTableWidgetItem(f"{offsets[i]:.2f}"))

        self.save_btn.setEnabled(True)
        self.mw.state["offsets_calibrated"] = True
        self.append_log("偏移量读取完成！")

    def _save_and_next(self):
        self.mw.enable_next_step()
        QMessageBox.information(self, "完成", "偏移校准已完成，可以进入下一步。")

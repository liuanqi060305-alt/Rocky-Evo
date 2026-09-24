"""Step 1: Port scanning and auto-configuration."""
import os
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTableWidget,
    QTableWidgetItem, QTextEdit, QLabel, QHeaderView, QMessageBox,
)
from PyQt6.QtCore import Qt

from gui.workers import PortScanWorker, PortConfigWorker


class Step1PortSetup(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.mw = main_window
        self._ports = []

        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)

        title = QLabel("步骤 1: 端口扫描与自动配置")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #cdd6f4;")
        layout.addWidget(title)

        desc = QLabel("检测 USB 串口设备，并自动分配主手和夹爪的端口映射。")
        desc.setStyleSheet("color: #a6adc8; font-size: 12px; margin-bottom: 8px;")
        layout.addWidget(desc)

        # Buttons
        btn_row = QHBoxLayout()
        self.scan_btn = QPushButton("🔍 扫描端口")
        self.scan_btn.clicked.connect(self._start_scan)
        self.config_btn = QPushButton("⚙ 自动配置")
        self.config_btn.setEnabled(False)
        self.config_btn.clicked.connect(self._start_config)
        self.save_btn = QPushButton("✅ 保存并继续")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._save_and_next)

        for btn in [self.scan_btn, self.config_btn, self.save_btn]:
            btn.setMinimumHeight(36)
            btn.setStyleSheet(self._btn_style())

        btn_row.addWidget(self.scan_btn)
        btn_row.addWidget(self.config_btn)
        btn_row.addWidget(self.save_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # Table
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["设备", "类型", "角色", "状态"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: #181825; color: #cdd6f4;
                gridline-color: #313244; border: 1px solid #313244;
            }
            QHeaderView::section {
                background-color: #313244; color: #cdd6f4; padding: 6px;
                border: none; font-weight: bold;
            }
        """)
        layout.addWidget(self.table)

        # Log
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(200)
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

    def append_log(self, msg: str):
        self.log.append(msg)

    def on_enter(self):
        pass

    def _start_scan(self):
        self.scan_btn.setEnabled(False)
        self.scan_btn.setText("扫描中...")
        self._ports = []
        self.table.setRowCount(0)

        self.worker = PortScanWorker()
        self.worker.ports_found.connect(self._on_ports_found)
        self.worker.log_message.connect(self.append_log)
        self.worker.error.connect(lambda e: self.append_log(f"错误: {e}"))
        self.worker.finished.connect(lambda: (
            self.scan_btn.setEnabled(True),
            self.scan_btn.setText("🔍 扫描端口"),
        ))
        self.worker.start()

    def _on_ports_found(self, ports):
        self._ports = ports
        self.table.setRowCount(len(ports))
        for i, p in enumerate(ports):
            self.table.setItem(i, 0, QTableWidgetItem(p))
            if "USB" in p:
                self.table.setItem(i, 1, QTableWidgetItem("FTDI (夹爪)"))
            else:
                self.table.setItem(i, 1, QTableWidgetItem("ACM (主手)"))
            self.table.setItem(i, 2, QTableWidgetItem("-"))
            self.table.setItem(i, 3, QTableWidgetItem("已发现"))

        self.config_btn.setEnabled(len(ports) >= 4)
        self.mw.update_status("left_hand", "online" if len(ports) >= 2 else "offline")
        self.mw.update_status("right_hand", "online" if len(ports) >= 2 else "offline")

    def _start_config(self):
        self.config_btn.setEnabled(False)
        self.config_btn.setText("配置中...")
        self.append_log("开始自动配置端口...")

        self.cfg_worker = PortConfigWorker()
        self.cfg_worker.log_message.connect(self.append_log)
        self.cfg_worker.warning.connect(lambda m: self.append_log(f"⚠ {m}"))
        self.cfg_worker.config_complete.connect(self._on_config_done)
        self.cfg_worker.finished.connect(lambda: (
            self.config_btn.setText("⚙ 自动配置"),
            self.config_btn.setEnabled(True),
        ))
        self.cfg_worker.start()

    def _on_config_done(self, result):
        self.append_log(f"配置完成: {result}")
        for row in range(self.table.rowCount()):
            device = self.table.item(row, 0).text()
            role = "-"
            for k, v in result.items():
                if v == device:
                    role = k
                    break
            self.table.setItem(row, 2, QTableWidgetItem(role))
            self.table.setItem(row, 3, QTableWidgetItem("✅ 已配置"))

        self.save_btn.setEnabled(True)
        self.mw.state["ports_configured"] = True
        self.mw.update_status("left_hand", "online")
        self.mw.update_status("right_hand", "online")

    def _save_and_next(self):
        self.mw.enable_next_step()
        QMessageBox.information(self, "完成", "端口配置已完成，可以进入下一步。")

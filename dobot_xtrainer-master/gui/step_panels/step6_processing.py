"""Step 6: Data processing — HDF5 conversion and frame counting."""
import os
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QTextEdit, QLabel, QComboBox, QProgressBar, QGroupBox, QGridLayout,
)

from gui.workers import DatasetScanWorker, FrameCountWorker, HDF5ConversionWorker


class Step6Processing(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.mw = main_window

        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        title = QLabel("步骤 6: 数据处理")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #cdd6f4;")
        layout.addWidget(title)

        desc = QLabel("将采集的原始数据转换为 HDF5 训练格式，并统计每个 episode 的帧数。")
        desc.setStyleSheet("color: #a6adc8; font-size: 12px;")
        layout.addWidget(desc)

        # Dataset selector
        scan_row = QHBoxLayout()
        scan_row.addWidget(QLabel("数据集路径:"))
        self.path_label = QLabel(
            os.path.join(self.mw.state["save_data_path"], self.mw.state["project_name"])
        )
        self.path_label.setStyleSheet("color: #a6adc8;")
        scan_row.addWidget(self.path_label)
        scan_row.addStretch()
        layout.addLayout(scan_row)

        self.scan_btn = QPushButton("🔍 扫描数据集")
        self.scan_btn.setMinimumHeight(36)
        self.scan_btn.clicked.connect(self._scan)
        self.scan_btn.setStyleSheet(self._btn_style())
        layout.addWidget(self.scan_btn)

        # Episode list
        self.dataset_combo = QComboBox()
        self.dataset_combo.setStyleSheet(
            "background-color: #181825; color: #cdd6f4; border: 1px solid #45475a; padding: 6px;"
        )
        self.dataset_combo.setMinimumHeight(30)
        layout.addWidget(self.dataset_combo)

        # Stats
        stats = QGroupBox("统计")
        stats.setStyleSheet(self._group_style())
        stats_grid = QGridLayout()
        self.ep_count_label = QLabel("Episode 数: -")
        self.total_frames_label = QLabel("总帧数: -")
        self.max_frames_label = QLabel("最大帧/Episode: -")
        for lbl in [self.ep_count_label, self.total_frames_label, self.max_frames_label]:
            lbl.setStyleSheet("color: #a6adc8;")
        stats_grid.addWidget(self.ep_count_label, 0, 0)
        stats_grid.addWidget(self.total_frames_label, 0, 1)
        stats_grid.addWidget(self.max_frames_label, 0, 2)
        stats.setLayout(stats_grid)
        layout.addWidget(stats)

        # Buttons
        btn_row = QHBoxLayout()
        self.count_btn = QPushButton("📊 统计帧数")
        self.count_btn.setMinimumHeight(36)
        self.count_btn.clicked.connect(self._count_frames)
        self.count_btn.setStyleSheet(self._btn_style())
        btn_row.addWidget(self.count_btn)

        self.convert_btn = QPushButton("🔄 转换为 HDF5")
        self.convert_btn.setMinimumHeight(36)
        self.convert_btn.clicked.connect(self._convert)
        self.convert_btn.setStyleSheet(self._btn_style())
        btn_row.addWidget(self.convert_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # Progress
        self.progress = QProgressBar()
        self.progress.setStyleSheet("""
            QProgressBar {
                background-color: #181825; border: 1px solid #45475a;
                border-radius: 4px; height: 20px; color: #cdd6f4;
            }
            QProgressBar::chunk { background-color: #a6e3a1; border-radius: 3px; }
        """)
        layout.addWidget(self.progress)

        # Log
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setStyleSheet("background-color: #11111b; color: #a6e3a1; font-family: monospace;")
        layout.addWidget(self.log)

        self.setLayout(layout)
        self._datasets = []

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

    def _data_root(self):
        return os.path.join(self.mw.state["save_data_path"], self.mw.state["project_name"])

    def append_log(self, msg: str):
        self.log.append(msg)

    def on_enter(self):
        pass

    def _scan(self):
        self.scan_btn.setEnabled(False)
        self.scan_btn.setText("扫描中...")
        self.append_log("正在扫描数据集...")
        self._datasets = []

        self.worker = DatasetScanWorker(self._data_root())
        self.worker.datasets_found.connect(self._on_datasets)
        self.worker.error.connect(lambda e: self.append_log(f"错误: {e}"))
        self.worker.finished.connect(lambda: (
            self.scan_btn.setEnabled(True),
            self.scan_btn.setText("🔍 扫描数据集"),
        ))
        self.worker.start()

    def _on_datasets(self, datasets):
        self._datasets = datasets
        self.dataset_combo.clear()
        total_frames = 0
        for d in datasets:
            self.dataset_combo.addItem(f"{d['name']} ({d['frames']} 帧)")
            total_frames += d["frames"]
        self.ep_count_label.setText(f"Episode 数: {len(datasets)}")
        self.total_frames_label.setText(f"总帧数: {total_frames}")
        self.append_log(f"发现 {len(datasets)} 个 episode，共 {total_frames} 帧")

    def _count_frames(self):
        self.count_btn.setEnabled(False)
        self.count_btn.setText("统计中...")
        self.append_log("正在统计...")

        self.count_worker = FrameCountWorker(self._data_root())
        self.count_worker.count_complete.connect(self._on_count)
        self.count_worker.error.connect(lambda e: self.append_log(f"错误: {e}"))
        self.count_worker.finished.connect(lambda: (
            self.count_btn.setEnabled(True),
            self.count_btn.setText("📊 统计帧数"),
        ))
        self.count_worker.start()

    def _on_count(self, total, max_frames):
        self.total_frames_label.setText(f"总帧数: {total}")
        self.max_frames_label.setText(f"最大帧/Episode: {max_frames}")
        self.append_log(f"统计完成: {total} 帧, 最大 episode {max_frames} 帧")

    def _convert(self):
        if not self._datasets:
            self.append_log("请先扫描数据集！")
            return
        self.convert_btn.setEnabled(False)
        self.convert_btn.setText("转换中...")
        self.progress.setMaximum(len(self._datasets))
        self.progress.setValue(0)
        self.append_log(f"开始转换 {len(self._datasets)} 个 episode...")

        self.conv_worker = HDF5ConversionWorker(
            self.mw.state["save_data_path"],
            self.mw.state["project_name"],
            self._datasets,
        )
        self.conv_worker.progress.connect(self.progress.setValue)
        self.conv_worker.log_message.connect(self.append_log)
        self.conv_worker.conversion_complete.connect(self._on_conversion_done)
        self.conv_worker.error.connect(lambda e: self.append_log(f"错误: {e}"))
        self.conv_worker.finished.connect(lambda: (
            self.convert_btn.setEnabled(True),
            self.convert_btn.setText("🔄 转换为 HDF5"),
        ))
        self.conv_worker.start()

    def _on_conversion_done(self, path):
        self.append_log(f"✅ 全部转换完成！输出目录: {path}")

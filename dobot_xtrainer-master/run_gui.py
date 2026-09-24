#!/usr/bin/env python3
"""Entry point for the Dobot Xtrainer GUI."""
import sys
import os

# Set BASE_DIR to project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Fix Qt plugin conflict: cv2 bundles its own Qt plugins which conflict with PyQt6.
# Must be set BEFORE importing PyQt6.
#
# 路径从已安装的 PyQt6 现算，不要硬编码：原先写死
# ~/miniconda3/envs/x_trainer/lib/python3.10/... —— 本机是 Miniforge + Python 3.8，
# 该路径不存在，GUI 起不来。
import PyQt6
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(
    os.path.dirname(PyQt6.__file__), "Qt6", "plugins", "platforms")
# Clear cv2's interfering QT_PLUGIN_PATH
os.environ.pop("QT_PLUGIN_PATH", None)

# 串口/相机权限：原先这里把 ini 里的明文口令管道给 sudo -S 去 chmod 777 每个设备。
# 已删除，三个原因：
#   1. 不需要了 —— iml 已在 dialout 组（串口 crw-rw---- root:dialout），
#      /dev/video* 由 logind 打的 ACL 授予 iml rw，无需提权。
#   2. 不安全 —— 明文口令进命令行，且 chmod 777 让全机所有用户可读写机械臂。
#   3. 本来就不可靠 —— snap 版 VS Code 的终端带 NoNewPrivs=1，里面 sudo 必然失败。
# 若换机后串口报 Permission denied：sudo usermod -aG dialout $USER 然后重新登录。

# Launch GUI
from PyQt6.QtWidgets import QApplication
from gui.main_window import MainWindow

app = QApplication(sys.argv)
app.setStyle("Fusion")

# Dark theme palette
from PyQt6.QtGui import QPalette, QColor
palette = QPalette()
palette.setColor(QPalette.ColorRole.Window, QColor(30, 30, 46))
palette.setColor(QPalette.ColorRole.WindowText, QColor(205, 214, 244))
palette.setColor(QPalette.ColorRole.Base, QColor(24, 24, 37))
palette.setColor(QPalette.ColorRole.AlternateBase, QColor(30, 30, 46))
palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(30, 30, 46))
palette.setColor(QPalette.ColorRole.ToolTipText, QColor(205, 214, 244))
palette.setColor(QPalette.ColorRole.Text, QColor(205, 214, 244))
palette.setColor(QPalette.ColorRole.Button, QColor(69, 71, 90))
palette.setColor(QPalette.ColorRole.ButtonText, QColor(205, 214, 244))
palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 255, 255))
palette.setColor(QPalette.ColorRole.Link, QColor(137, 180, 250))
palette.setColor(QPalette.ColorRole.Highlight, QColor(137, 180, 250))
palette.setColor(QPalette.ColorRole.HighlightedText, QColor(30, 30, 46))
app.setPalette(palette)
app.setStyleSheet("""
    QToolTip { color: #cdd6f4; background-color: #313244; border: 1px solid #45475a; }
""")

window = MainWindow()
window.show()
sys.exit(app.exec())

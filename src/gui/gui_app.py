#!/usr/bin/env python3.12
# -*- coding: utf-8 -*-
"""GUI 启动入口 (兼容旧路径)"""
import sys
from pathlib import Path
from PyQt6.QtWidgets import QApplication

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.gui.main_window import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

#!/usr/bin/env python3.12
# -*- coding: utf-8 -*-
"""GUI 启动入口 (兼容旧路径)"""
import sys
from pathlib import Path
from PyQt6.QtWidgets import QApplication

if __package__ is None or __package__ == "":
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "src").is_dir():
            sys.path.insert(0, str(parent))
            break

from src.gui.main_window import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

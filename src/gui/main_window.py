#!/usr/bin/env python3.12
# -*- coding: utf-8 -*-
"""
文件: main_window.py
模块: src.gui.main_window
功能: 使用 PyQt6 加载您在 Qt Designer 设计的 UI 文件，将前端界面与后端处理模块对接，并自动绑定信号槽
作者: 孟诣楠
版本: v1.0.3
创建时间: 2025-06-18
最近更新: 2025-06-18
较上一版本改进:
    a) 适配用户提供的 UI 文件 main_window.ui
    b) 使用 PyQt6.uic 动态加载 .ui，无需先转换为 .py，减少迭代成本
    c) 保持接口不变，提供可单独运行测试的入口
"""
import sys
from pathlib import Path
if __package__ is None or __package__ == "":
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "src").is_dir():
            sys.path.insert(0, str(parent))
            break
import os
from PyQt6 import uic
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QProgressDialog
)
from src.workers.display_worker import DisplayWorker
from src.processing.task_manager import TaskManager
from src.workers.processing_worker import ProcessingWorker

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # 动态加载 UI
        ui_path = os.path.join(os.path.dirname(__file__), 'ui', 'yaogan', 'yaogan.ui')
        uic.loadUi(ui_path, self)

        # 初始化任务管理器（加载默认配置）
        self.task_manager = TaskManager()

        # 进度对话框
        self.progressDialog = QProgressDialog(self)
        self.progressDialog.setAutoClose(False)
        self.progressDialog.setLabelText("准备中…")

        # 统一加载所有菜单动作的槽
        
        
        # TODO: 为其他 actionXXX 依次绑定对应的槽函数


    def _on_task_complete(self, name: str, result_path: str):
        """统一任务完成处理"""
        self.progressDialog.hide()
        self.statusBar().showMessage(f"{name} 完成，结果保存在 {result_path}", 5000)
        # TODO: 在图层面板中加载结果文件

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

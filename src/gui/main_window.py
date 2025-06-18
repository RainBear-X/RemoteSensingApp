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
from PyQt6.QtCore import QThread
from src.workers.display_worker import DisplayWorker
from src.workers.processing_worker import ProcessingWorker
from src.workers.file_worker import FileWorker
from src.workers.file_saver_worker import FileSaverWorker
from src.workers.vector_worker import VectorWorker
from src.processing.task_manager import TaskManager
from src.processing.task_result import TaskResult

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
        self.progressDialog.hide()
        # 取消按钮关闭当前线程
        self.progressDialog.canceled.connect(self.cancel_current_worker)

        # 当前运行的后台线程引用，避免被垃圾回收
        self.current_worker: QThread | None = None

        # 绑定部分菜单动作到后台任务
        if hasattr(self, 'actionOpenImageFile'):
            self.actionOpenImageFile.triggered.connect(self.run_file_operation)
        if hasattr(self, 'actionOpenVectorData'):
            self.actionOpenVectorData.triggered.connect(self.run_vector_processing)
        if hasattr(self, 'actionSaveImageFileAs'):
            self.actionSaveImageFileAs.triggered.connect(self.run_file_save)
        if hasattr(self, 'actionSaveVectorFileAs'):
            self.actionSaveVectorFileAs.triggered.connect(self.run_vector_processing)
        if hasattr(self, 'actionExit'):
            self.actionExit.triggered.connect(self.close)
        if hasattr(self, 'actionBandextraction'):
            self.actionBandextraction.triggered.connect(self.run_image_display)
        if hasattr(self, 'actionImagestretching'):
            self.actionImagestretching.triggered.connect(self.run_image_processing)

        # TODO: 为其他 actionXXX 依次绑定对应的槽函数



    # ===== 后台任务接口 =====
    def _start_worker(self, worker: QThread, title: str):
        """通用启动方法"""
        # 保存当前线程引用，避免被回收
        self.current_worker = worker

        worker.progress.connect(self.progressDialog.setLabelText)
        worker.finished.connect(lambda res: self._handle_result(title, res))
        worker.finished.connect(self._clear_current_worker)

        self.progressDialog.setLabelText(f"{title}…")
        self.progressDialog.show()
        worker.start()

    def _handle_result(self, title: str, result: TaskResult):
        self.progressDialog.hide()
        if result.status == "success":
            msg = f"{title}完成"
        else:
            msg = f"{title}失败: {result.message}"
        self.statusBar().showMessage(msg, 5000)


    def cancel_current_worker(self):
        """取消当前运行的后台线程"""
        if self.current_worker and self.current_worker.isRunning():
            self.current_worker.terminate()

    # 向后兼容旧接口
    def _cancel_current_worker(self):
        self.cancel_current_worker()

    def _clear_current_worker(self):
        self.current_worker = None

    def run_file_operation(self):
        params = getattr(self.task_manager.config, "file_operation_params", {})
        worker = FileWorker(params=params)
        self._start_worker(worker, "文件加载")

    def run_image_display(self):
        params = getattr(self.task_manager.config, "image_display_params", {})
        worker = DisplayWorker(params=params)
        self._start_worker(worker, "波段可视化")

    def run_image_processing(self):
        params = getattr(self.task_manager.config, "image_processing_params", {})
        worker = ProcessingWorker(params=params)
        self._start_worker(worker, "图像处理")

    def run_file_save(self):
        params = getattr(self.task_manager.config, "file_saver_params", {})
        worker = FileSaverWorker(params=params)
        self._start_worker(worker, "文件保存")

    def run_vector_processing(self):
        params = getattr(self.task_manager.config, "vector_processing_params", {})
        worker = VectorWorker(params=params)
        self._start_worker(worker, "矢量处理")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

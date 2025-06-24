# vector_worker.py
# --------------------------
#!/usr/bin/env python3.12
# -*- coding: utf-8 -*-
"""
文件: vector_worker.py
模块: src.processing.workers.vector_worker
功能: 矢量处理后台工作线程
作者: 孟诣楠
版本: v1.0.0
创建时间: 2025-06-19
最近更新: 2025-06-19
较上一版改进:
  - 首次创建
"""
from PyQt6.QtCore import QThread, pyqtSignal
from src.processing.task_manager import TaskManager
from src.processing.task_result import TaskResult

class VectorWorker(QThread):
    """
    后台线程：执行 vector_processing 任务并发出信号。
    """
    progress = pyqtSignal(str)
    finished = pyqtSignal(TaskResult)

    def __init__(self, config_path: str = None, params: dict = None):
        super().__init__()
        self.manager = TaskManager(config_path)
        self.task_name = "vector_processing"
        self.params = params or {}

    def run(self):
        self.progress.emit(f"开始任务: {self.task_name}")
        result = self.manager.run_task(self.task_name, self.params)
        self.finished.emit(result)

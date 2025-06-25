# evaluation_worker.py
# --------------------------
#!/usr/bin/env python3.12
# -*- coding: utf-8 -*-
"""
文件: evaluation_worker.py
模块: src.processing.workers.evaluation_worker
功能: 评估后台工作线程
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

class EvaluationWorker(QThread):
    """
    后台线程：执行 evaluation 任务并发出信号。
    """
    progress = pyqtSignal(str)
    finished = pyqtSignal(TaskResult)

    def __init__(self, config_path: str = None, params: dict = None):
        super().__init__()
        self.manager = TaskManager(config_path)
        self.task_name = "evaluation"
        self.params = params or {}

    def run(self):
        self.progress.emit(f"开始任务: {self.task_name}")
        result = self.manager.run_task(self.task_name, self.params)
        self.finished.emit(result)

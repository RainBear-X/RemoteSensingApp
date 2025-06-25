#!/usr/bin/env python3.12
# -*- coding: utf-8 -*-
"""
文件: feature_worker.py
模块: src.workers.feature_worker
作者：张子涵
功能: 特征提取后台工作线程
"""
from PyQt6.QtCore import QThread, pyqtSignal
from src.processing.task_manager import TaskManager
from src.processing.task_result import TaskResult

class FeatureWorker(QThread):
    """执行 feature_extraction 任务的线程"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(TaskResult)

    def __init__(self, config_path: str = None, params: dict | None = None):
        super().__init__()
        self.manager = TaskManager(config_path)
        self.task_name = "feature_extraction"
        self.params = params or {}

    def run(self):
        self.progress.emit(f"开始任务: {self.task_name}")
        result = self.manager.run_task(self.task_name, self.params)
        self.finished.emit(result)

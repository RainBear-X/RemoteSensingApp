#!/usr/bin/env python3.12
# -*- coding: utf-8 -*-
"""后台线程：执行文件保存任务"""
from PyQt6.QtCore import QThread, pyqtSignal
from src.processing.task_manager import TaskManager
from src.processing.task_result import TaskResult


class FileSaverWorker(QThread):
    """执行 file_saver 任务并发出进度与完成信号"""

    progress = pyqtSignal(str)
    finished = pyqtSignal(TaskResult)

    def __init__(self, config_path: str | None = None, params: dict | None = None):
        super().__init__()
        self.manager = TaskManager(config_path)
        self.task_name = "file_saver"
        self.params = params or {}

    def run(self) -> None:
        self.progress.emit(f"开始任务: {self.task_name}")
        result = self.manager.run_task(self.task_name, self.params)
        self.finished.emit(result)

#!/usr/bin/env python3.12
# -*- coding: utf-8 -*-
"""
文件: task_manager.py
模块: src.processing.task_manager
功能: 提供任务管理接口，封装 RemoteSensingEngine 以便 GUI 调用
作者: 孟诣楠
版本: v1.0.0
创建时间: 2025-06-18
最近更新: 2025-06-18
较上一版改进:
  1. 首次实现 TaskManager，封装 RemoteSensingEngine；
"""
import logging
from typing import Any, Dict

from src.processing.engine import load_config, RemoteSensingEngine
from src.processing.task_result import TaskResult


class TaskManager:
    """
    任务管理器，封装遥感处理引擎，提供给 GUI 调用的统一接口。
    """

    def __init__(self, config_path: str = None):
        """
        初始化 TaskManager。

        参数:
            config_path: 可选，配置文件路径，支持 YAML 或 Python 脚本，不指定则使用根目录 config.py。
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        try:
            self.config = load_config(config_path)
            self.engine = RemoteSensingEngine(self.config)
            self.logger.info(f"加载配置成功: {config_path or '根目录 config.py'}")
        except Exception as e:
            self.logger.error(f"加载配置失败: {e}")
            raise

    def run_task(self, task_name: str, params: Dict[str, Any] = None) -> TaskResult:
        """
        运行单个任务。

        参数:
            task_name: 任务名称，需与引擎注册的任务名一致
            params:    任务参数字典

        返回:
            TaskResult
        """
        if params is None:
            params = {}
        self.logger.info(f"开始运行任务: {task_name}，参数: {params}")
        result = self.engine.run_task(task_name, **params)
        return result

    def run_all(self) -> Dict[str, TaskResult]:
        """
        运行全流程任务。

        返回:
            各任务名称到 TaskResult 的映射
        """
        self.logger.info("开始运行全流程任务")
        results = self.engine.run()
        return results

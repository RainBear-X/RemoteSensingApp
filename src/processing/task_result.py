#!/usr/bin/env python3.12
# -*- coding: utf-8 -*-
"""
文件: task_result.py
模块: src.processing.task_result
功能: 定义任务执行结果统一返回类型 TaskResult
作者: 孟诣楠
版本: v1.0.0
最近更新: 2025-06-18
"""

from dataclasses import dataclass
from typing import List

@dataclass
class TaskResult:
    """
    任务结果封装类

    属性:
        status: 执行状态，"success" 或 "failure"
        message: 详细信息或错误消息
        outputs: 生成的文件路径列表
        logs: 运行日志列表
    """
    status: str
    message: str
    outputs: List[str]
    logs: List[str]

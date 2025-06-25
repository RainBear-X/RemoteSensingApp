#!/usr/bin/env python3.12
# -*- coding: utf-8 -*-
"""
文件: run_file_saver.py
模块: src.processing.file_operations.run_file_saver
功能: 将文件操作输出结果备份/保存到指定目录
作者: 孟诣楠
版本: v1.0.1
最近更新: 2025-06-18
较上一版改进:
  1. 初始实现，支持 save_dir 参数；
  2. 自动读取 config.file_operation_params.output_dir 作为源目录；
"""
import os
import sys
from pathlib import Path
import shutil
from typing import Any, List
if __package__ is None or __package__ == "":
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "src").is_dir():
            sys.path.insert(0, str(parent))
            break

from src.processing.task_result import TaskResult


def run(config: Any, save_dir: str) -> TaskResult:
    """
    备份文件操作模块的输出结果到 save_dir。

    参数:
        config: 配置对象，需包含 file_operation_params.output_dir
        save_dir: 保存目标目录

    返回:
        TaskResult: 包含 status、message、outputs(备份路径列表)、logs(日志列表)
    """
    logs: List[str] = []
    outputs: List[str] = []

    # 源目录来自上一模块输出
    src_dir = None
    try:
        src_dir = config.file_operation_params.get("output_dir")
    except Exception:
        msg = "配置中缺少 file_operation_params.output_dir"
        return TaskResult(status="failure", message=msg, outputs=outputs, logs=[msg])

    if not src_dir or not os.path.isdir(src_dir):
        msg = f"源目录不存在: {src_dir}"
        return TaskResult(status="failure", message=msg, outputs=outputs, logs=[msg])

    # 创建目标目录
    try:
        os.makedirs(save_dir, exist_ok=True)
        logs.append(f"创建保存目录: {save_dir}")
    except Exception as e:
        msg = f"无法创建保存目录 [{save_dir}]: {e}"
        return TaskResult(status="failure", message=msg, outputs=outputs, logs=[msg])

    # 复制文件
    for fname in os.listdir(src_dir):
        src_fp = os.path.join(src_dir, fname)
        dst_fp = os.path.join(save_dir, fname)
        try:
            shutil.copy2(src_fp, dst_fp)
            outputs.append(dst_fp)
            logs.append(f"已保存: {dst_fp}")
        except Exception as e:
            logs.append(f"复制文件失败 [{src_fp}]: {e}")
            # 继续复制其他文件

    return TaskResult(status="success", message="文件保存完成", outputs=outputs, logs=logs)


if __name__ == "__main__":
    # 简单测试
    from src.constants import OUTPUT_DIR, TEST_DATA_DIR
    # 先模拟 file_operation 输出
    tmp_out = os.path.join(OUTPUT_DIR, 'file_operation')
    res = run(config=type('C',(object,),{'file_operation_params':{'output_dir': tmp_out}})(), save_dir=os.path.join(OUTPUT_DIR, 'saved_files'))
    print(res)

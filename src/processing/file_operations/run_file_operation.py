#!/usr/bin/env python3.12
# -*- coding: utf-8 -*-
"""
文件: run_file_operation.py
模块: src.processing.file_operations.run_file_operation
功能: 批量加载遥感影像文件并将像元数据以 NumPy 数组保存为 .npy 格式
作者: 孟诣楠
版本: v1.0.1
创建时间: 2025-06-18
最近更新: 2025-06-20
较上一版改进:
  1. 修改 run 接口，新增 input_dir 参数以支持目录批量处理；
  2. 去除 input_paths，改为遍历 input_dir 下所有 .tif/.tiff 文件；
  3. 增加 options 参数用于图像加载可选项；
  4. 新增 input_paths 可选参数，可直接指定文件列表
"""

import os
import sys
from pathlib import Path
import numpy as np
from typing import Any, Dict, List, Optional
if __package__ is None or __package__ == "":
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "src").is_dir():
            sys.path.insert(0, str(parent))
            break

from src.processing.task_result import TaskResult
from src.processing.file_operations.file_loader import load_image


def run(
        config: Any,
       input_dir: str | None = None,
        output_dir: str = '',
        *,
        input_paths: Optional[List[str]] = None,
        options: Optional[Dict[str, Any]] = None
) -> TaskResult:
    """
    批量加载遥感影像文件（.tif/.tiff），并将像素矩阵以 NumPy 数组形式保存为 .npy。

    参数:
        config: 配置对象，提供 data_dir、temp_dir 等基础路径
        input_dir: 输入数据目录
        output_dir: 保存结果的输出目录
        input_paths: 直接指定的影像文件列表（可选）
        options: 可选加载参数

    返回:
        TaskResult: 包含 status("success"/"failure"）、message、outputs(文件列表)、logs(日志列表）
    """
    logs: List[str] = []
    outputs: List[str] = []
    options = options or {}

    # 创建输出目录
    try:
        os.makedirs(output_dir, exist_ok=True)
        logs.append(f"创建输出目录: {output_dir}")
    except Exception as e:
        msg = f"无法创建输出目录 [{output_dir}]: {e}"
        return TaskResult(status="failure", message=msg, outputs=outputs, logs=[msg])

    # 构建待处理文件列表
    if input_paths:
        file_list = input_paths
    else:
        if input_dir is None:
            msg = "缺少 input_dir 或 input_paths"
            return TaskResult(status="failure", message=msg, outputs=outputs, logs=[msg])
        file_list = [
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if f.lower().endswith((".tif", ".tiff"))
        ]

    # 遍历加载
    for fp in file_list:
        fname = os.path.basename(fp)
        if not fname.lower().endswith((".tif", ".tiff")):
            logs.append(f"跳过非影像文件: {fname}")
            continue
        logs.append(f"开始加载影像: {fp}")
        try:
            arr, meta = load_image(fp, options)
            base = os.path.splitext(fname)[0]
            out_name = base + ".npy"
            out_path = os.path.join(output_dir, out_name)
            np.save(out_path, arr)
            logs.append(f"保存数组到: {out_path}")
            outputs.append(out_path)
        except Exception as e:
            err = f"加载失败 [{fp}]: {e}"
            logs.append(err)
            return TaskResult(status="failure", message=str(e), outputs=outputs, logs=logs)

    return TaskResult(
        status="success",
        message="所有影像加载并保存完成",
        outputs=outputs,
        logs=logs
    )


if __name__ == "__main__":
    from src.constants import TEST_DATA_DIR, OUTPUT_DIR
    # 简单测试
    result = run(
        config=None,
        input_dir=TEST_DATA_DIR,
        output_dir=os.path.join(OUTPUT_DIR, 'file_operation'),
        options=None
    )
    print(result)

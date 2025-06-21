#!/usr/bin/env python3.12
# -*- coding: utf-8 -*-
"""
文件: run_image_processing.py
模块: src.processing.image_processing.run_image_processing
功能: 批量对遥感影像进行增强与滤波处理，并保存结果
作者: 孟诣楠
版本: v1.2.3
最近更新: 2025-06-18
较上一版改进:
  1. 修正 stretch_percent 调用：将 in_range 元组拆分为 low, high 传递到函数；
"""

import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
if __package__ is None or __package__ == "":
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "src").is_dir():
            sys.path.insert(0, str(parent))
            break
from src.processing.task_result import TaskResult

# 引入增强函数
from src.processing.image_processing.enhancement.equalization import hist_equalize
from src.processing.image_processing.enhancement.image_stretching import stretch_percent
# 引入平滑函数
from src.processing.image_processing.filtering.smoothing import (
    smooth_mean, smooth_gaussian, smooth_median
)
# 引入边缘检测函数
from src.processing.image_processing.filtering.edge_detection import (
     edge_sobel, edge_canny, edge_roberts
)
# 引入锐化函数
from src.processing.image_processing.filtering.sharpening import (
    sharpen_unsharp, sharpen_laplacian
)
from src.processing.image_processing.band_math import custom_expression
# 方法映射表
_PROCESS_FUNCS: Dict[str, Any] = {
    'equalization': hist_equalize,
    'stretch': stretch_percent,
    'smooth_mean': smooth_mean,
    'smooth_gaussian': smooth_gaussian,
    'smooth_median': smooth_median,
    'edge_sobel': edge_sobel,
    'edge_canny': edge_canny,
    'edge_roberts': edge_roberts,
    'sharpen_unsharp': sharpen_unsharp,
    'sharpen_laplacian': sharpen_laplacian,
    'band_math': custom_expression,
}


def run(
        config: Any,
        paths: List[str],
        methods: List[str],
        output_dir: str,
        options: Optional[Dict[str, Dict[str, Any]]] = None
) -> TaskResult:
    """
    批量对影像执行一系列处理方法，并保存最终结果。

    参数:
        config: 配置对象（未使用）
        paths: 待处理文件路径列表，支持 .npy 与 .pkl
        methods: 方法列表，可选：
            'equalization', 'stretch',
            'smooth_mean','smooth_gaussian','smooth_median',
            'edge_sobel','edge_canny',
            'sharpen_unsharp','sharpen_laplacian'
            'sharpen_unsharp','sharpen_laplacian',
            'band_math'
        output_dir: 保存结果目录
        options: 每个方法的参数字典

    返回:
        TaskResult: status, message, outputs, logs
    """
    logs: List[str] = []
    outputs: List[str] = []
    opts = options or {}

    # 创建输出目录
    try:
        os.makedirs(output_dir, exist_ok=True)
        logs.append(f"创建输出目录: {output_dir}")
    except Exception as e:
        msg = f"无法创建输出目录 [{output_dir}]: {e}"
        return TaskResult(status="failure", message=msg, outputs=outputs, logs=[msg])

    # 逐文件处理
    for fp in paths:
        logs.append(f"加载文件: {fp}")
        try:
            ext = os.path.splitext(fp)[1].lower()
            if ext == '.npy':
                arr = np.load(fp)
            elif ext in ('.pkl', '.pickle'):
                with open(fp, 'rb') as f:
                    data = pickle.load(f)
                arr = data[0] if isinstance(data, tuple) and data else data
            else:
                raise ValueError(f"不支持的文件格式: {ext}")
        except Exception as e:
            err = f"加载失败 [{fp}]: {e}"
            logs.append(err)
            return TaskResult(status="failure", message=str(e), outputs=outputs, logs=logs)

        # 按顺序应用各方法
        for method in methods:
            if method not in _PROCESS_FUNCS:
                err = f"不支持的处理方法: {method}"
                logs.append(err)
                return TaskResult(status="failure", message=err, outputs=outputs, logs=logs)
            params = opts.get(method, {})
            logs.append(f"应用方法 [{method}]，参数: {params}")
            try:
                if method == 'stretch':
                    # 将 in_range 拆分为 low, high 传入
                    low, high = params.get('in_range', (2, 98))
                    temp = stretch_percent(arr, low, high)
                    # 可选 out_range 线性映射
                    if 'out_range' in params:
                        o_min, o_max = params['out_range']
                        temp = np.clip((temp - temp.min()) / (temp.max() - temp.min() + 1e-8) *
                                       (o_max - o_min) + o_min, o_min, o_max)
                    arr = temp
                elif method == 'band_math':
                    expr = params.get('expr')
                    if not expr:
                        raise ValueError('missing expr')
                    if arr.ndim == 3:
                        arr = custom_expression(expr, *arr)
                    else:
                        arr = custom_expression(expr, arr)
                else:
                    func = _PROCESS_FUNCS[method]
                    arr = func(arr, **params)
            except Exception as e:
                err = f"方法 [{method}] 失败: {e}"
                logs.append(err)
                return TaskResult(status="failure", message=str(e), outputs=outputs, logs=logs)

        # 生成输出文件名
        base = os.path.splitext(os.path.basename(fp))[0]
        methods_str = '_'.join(methods)
        out_name = f"{base}_{methods_str}.npy"
        out_path = os.path.join(output_dir, out_name)
        try:
            np.save(out_path, arr)
            logs.append(f"保存结果: {out_path}")
            outputs.append(out_path)
        except Exception as e:
            err = f"保存失败 [{out_path}]: {e}"
            logs.append(err)
            return TaskResult(status="failure", message=str(e), outputs=outputs, logs=logs)

    return TaskResult(status="success", message="图像处理完成", outputs=outputs, logs=logs)


# 单元测试示例
if __name__ == '__main__':
    from src.constants import TEST_DATA_DIR, OUTPUT_DIR
    import glob
    npys = glob.glob(os.path.join(TEST_DATA_DIR, '*.npy'))
    methods = [
        'equalization', 'stretch',
        'smooth_gaussian', 'edge_sobel', 'sharpen_unsharp'
    ]
    opts = {
        'stretch': {'in_range': (2,98), 'out_range': (0,255)},
        'edge_canny': {'sigma':1.5},
        'smooth_gaussian': {'sigma':1.0}
    }
    res = run(
        config=None,
        paths=npys,
        methods=methods,
        output_dir=os.path.join(OUTPUT_DIR, 'processed'),
        options=opts
    )
    print(res)

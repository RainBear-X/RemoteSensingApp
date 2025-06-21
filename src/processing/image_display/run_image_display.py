#!/usr/bin/env python3.12
# -*- coding: utf-8 -*-
"""
文件: run_image_display.py
模块: src.processing.image_display.run_image_display
功能: 批量提取和可视化影像波段，将指定波段保存为 PNG 格式，支持 `.npy` 与 `.pkl` 文件
作者: 孟诣楠
版本: v1.0.3
最近更新: 2025-06-18
较上一版改进:
  1. 修正 NumPy 2.0 不再支持 ndarray.ptp()，改用 np.ptp()；
"""

import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image
if __package__ is None or __package__ == "":
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "src").is_dir():
            sys.path.insert(0, str(parent))
            break

from src.processing.task_result import TaskResult


def run(
        config: Any,
        paths: List[str],
        bands: List[int],
        output_dir: str,
        options: Optional[Dict[str, Any]] = None
) -> TaskResult:
    """
    批量提取和可视化影像波段，将指定波段保存为 PNG。
    支持读取 `.npy` 和 `.pkl` 文件，其中 `.pkl` 中应存储 (array, meta) 二元组。

    参数:
        config: 配置对象
        paths: .npy 或 .pkl 文件路径列表
        bands: 要提取的波段索引列表（基于1）
        output_dir: PNG 保存目录
        options: 可选参数（如归一化方式）

    返回:
        TaskResult: 包含 status、message、outputs(文件列表)、logs(日志列表)
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

    # 处理每个文件
    for fp in paths:
        logs.append(f"处理文件: {fp}")
        try:
            ext = os.path.splitext(fp)[1].lower()
            if ext == '.npy':
                arr = np.load(fp)
                meta: Dict = {}
            elif ext in ('.pkl', '.pickle'):
                with open(fp, 'rb') as f:
                    data = pickle.load(f)
                if isinstance(data, tuple) and len(data) == 2:
                    arr, meta = data
                else:
                    arr = data
                    meta = {}
            else:
                raise ValueError(f"不支持的文件格式: {ext}")

            # 确保形状为 (C, H, W)
            if arr.ndim == 2:
                arr = arr[np.newaxis, ...]

            # 单波段或三波段合成
            if len(bands) == 3:
                channels = []
                for b in bands:
                    band_data = arr[b-1]
                    band_min = band_data.min()
                    band_ptp = np.ptp(band_data)
                    norm = (band_data - band_min) / (band_ptp + 1e-8)
                    channels.append((norm * 255).astype(np.uint8))
                rgb = np.stack(channels, axis=-1)
                img = Image.fromarray(rgb)
            else:
                b = bands[0]
                band_data = arr[b-1]
                band_min = band_data.min()
                band_ptp = np.ptp(band_data)
                norm = (band_data - band_min) / (band_ptp + 1e-8)
                gray = (norm * 255).astype(np.uint8)
                img = Image.fromarray(gray)

            base = os.path.splitext(os.path.basename(fp))[0]
            band_str = "_".join(str(b) for b in bands)
            out_name = f"{base}_bands_{band_str}.png"
            out_path = os.path.join(output_dir, out_name)
            img.save(out_path)

            logs.append(f"保存图像: {out_path}")
            outputs.append(out_path)
        except Exception as e:
            err = f"处理失败 [{fp}]: {e}"
            logs.append(err)
            return TaskResult(status="failure", message=str(e), outputs=outputs, logs=logs)

    return TaskResult(
        status="success",
        message="波段可视化完成",
        outputs=outputs,
        logs=logs
    )


# ———— 单元测试示例 ————
if __name__ == "__main__":
    from src.constants import TEST_DATA_DIR, OUTPUT_DIR
    import glob
    npys = glob.glob(os.path.join(TEST_DATA_DIR, '*.npy'))
    result = run(
        config=None,
        paths=npys,
        bands=[1, 2, 3],
        output_dir=os.path.join(OUTPUT_DIR, 'display'),
        options=None
    )
    print(result)

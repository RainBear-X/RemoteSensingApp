#!/usr/bin/env python3.12
# -*- coding: utf-8 -*-
"""
文件: run_evaluation.py
模块: src.processing.accuracy_evaluation.run_evaluation
功能: 执行分类精度评估流程，计算混淆矩阵、总体精度（OA）、Kappa系数，并生成图像与文本报告
作者: 孟诣楠
版本: v1.0.0
创建时间: 2025-06-19
最近更新: 2025-06-19
"""

import os
import sys
from pathlib import Path
import numpy as np
from typing import Any, Dict, Optional, List
# 延迟导入 TaskResult 避免循环依赖
if __package__ is None or __package__ == "":
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "src").is_dir():
            sys.path.insert(0, str(parent))
            break
from src.processing.task_result import TaskResult

from src.processing.accuracy_evaluation.sample_verification import extract_valid_samples
from src.processing.accuracy_evaluation.confusion_matrix import (
    compute_confusion_matrix, plot_confusion_matrix
)
from src.processing.accuracy_evaluation.overall_accuracy import compute_overall_accuracy
from src.processing.accuracy_evaluation.kappa_coefficient import compute_kappa
from src.processing.accuracy_evaluation.evaluation_report import generate_text_report


def run(
        config: any,
        class_map_path: str,
        roi_mask_path: str,
        output_dir: str,
        options: Optional[Dict[str, any]] = None
) -> TaskResult:
    """
    执行分类精度评估：
      1. 提取有效样本 y_true 与 y_pred
      2. 计算混淆矩阵并绘图
      3. 计算总体精度（OA）与 Kappa 系数
      4. 生成并保存文本报告

    参数:
        config: 配置对象（可选，包含输出目录基础路径）
        class_map_path: 分类结果路径（.npy 或栅格路径）
        roi_mask_path: ROI 掩膜 .npy 或 .pkl 文件路径
        output_dir: 评估结果保存目录
        options: 可选参数字典（如 plot 配置）

    返回:
        TaskResult: status, message, outputs (图像与报告路径), logs
    """
    logs = []
    outputs = []
    opts = options or {}

    # 创建输出目录
    try:
        os.makedirs(output_dir, exist_ok=True)
        logs.append(f"创建输出目录: {output_dir}")
    except Exception as e:
        msg = f"无法创建输出目录 [{output_dir}]: {e}"
        return TaskResult(status="failure", message=msg, outputs=outputs, logs=logs + [msg])

    # 1. 加载分类图和 ROI 掩膜
    try:
        if class_map_path.endswith('.npy'):
            class_map = np.load(class_map_path)
        else:
            import rasterio
            class_map = rasterio.open(class_map_path).read(1)
        logs.append(f"加载分类图: {class_map_path}")
    except Exception as e:
        msg = f"加载分类图失败: {e}"
        return TaskResult(status="failure", message=msg, outputs=outputs, logs=logs + [msg])

    try:
        if roi_mask_path.endswith('.npy'):
            roi = np.load(roi_mask_path)
        else:
            import pickle
            with open(roi_mask_path, 'rb') as f:
                roi = pickle.load(f)
        logs.append(f"加载 ROI 掩膜: {roi_mask_path}")
    except Exception as e:
        msg = f"加载 ROI 掩膜失败: {e}"
        return TaskResult(status="failure", message=msg, outputs=outputs, logs=logs + [msg])

    # 2. 提取有效样本
    try:
        y_true, y_pred, mask = extract_valid_samples(class_map, roi)
        logs.append(f"提取有效样本: {len(y_true)} 个")
    except Exception as e:
        msg = f"提取有效样本失败: {e}"
        return TaskResult(status="failure", message=msg, outputs=outputs, logs=logs + [msg])

    # 3. 计算混淆矩阵
    try:
        cm = compute_confusion_matrix(y_true, y_pred)
        fig = plot_confusion_matrix(cm, **opts.get('plot', {}))
        cm_path = os.path.join(output_dir, 'confusion_matrix.png')
        fig.savefig(cm_path, dpi=300, bbox_inches='tight')
        logs.append(f"混淆矩阵图保存: {cm_path}")
        outputs.append(cm_path)
    except Exception as e:
        msg = f"混淆矩阵计算或绘图失败: {e}"
        return TaskResult(status="failure", message=msg, outputs=outputs, logs=logs + [msg])

    # 4. 计算总体精度和 Kappa
    try:
        oa = compute_overall_accuracy(cm)
        kappa = compute_kappa(cm)
        logs.append(f"总体精度 (OA): {oa:.4f}")
        logs.append(f"Kappa 系数: {kappa:.4f}")
    except Exception as e:
        msg = f"OA/Kappa 计算失败: {e}"
        return TaskResult(status="failure", message=msg, outputs=outputs, logs=logs + [msg])

    # 5. 生成评估报告
    try:
        report_path = os.path.join(output_dir, 'evaluation_report.txt')
        generate_text_report(
            report_path,
            oa=oa,
            kappa=kappa,
            confusion_matrix=cm,
            sample_count=len(y_true)
        )
        logs.append(f"评估报告生成: {report_path}")
        outputs.append(report_path)
    except Exception as e:
        msg = f"生成评估报告失败: {e}"
        return TaskResult(status="failure", message=msg, outputs=outputs, logs=logs + [msg])

    return TaskResult(
        status="success",
        message="精度评估完成",
        outputs=outputs,
        logs=logs
    )

# ———— 单独运行测试 ————
if __name__ == '__main__':
    class Config: pass
    cfg = Config()
    res = run(
        config=cfg,
        class_map_path='output/class_map.npy',
        roi_mask_path='data/roi.npy',
        output_dir='output/evaluation'
    )
    print(res)

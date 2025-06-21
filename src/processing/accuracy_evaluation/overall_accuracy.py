#!/usr/bin/env python3.12
# -*- coding: utf-8 -*-
"""
文件: overall_accuracy.py
模块: src.processing.accuracy_evaluation
功能: 计算分类总体精度（OA）
作者: 孟诣楠
版本: v1.0.0
最近更新: 2025-06-17
"""
import numpy as np

def compute_overall_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算总体精度 OA
    参数:
        y_true: 真实标签（1D）
        y_pred: 预测标签（1D）
    返回:
        overall_accuracy (float): 总体精度
    """
    from sklearn.metrics import accuracy_score
    return accuracy_score(y_true, y_pred)

if __name__ == "__main__":
    # 单元测试
    import numpy as _np
    y_true = _np.array([1,0,1,1])
    y_pred = _np.array([1,0,0,1])
    oa = compute_overall_accuracy(y_true, y_pred)
    print(f"Overall Accuracy: {oa:.3f}")
#!/usr/bin/env python3.12
# -*- coding: utf-8 -*-
"""
文件: kappa_coefficient.py
模块: src.processing.accuracy_evaluation
功能: 计算Cohen's Kappa系数
作者: 孟诣楠
版本: v1.0.0
最近更新: 2025-06-17
"""
import numpy as np

def compute_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算Kappa系数
    """
    from sklearn.metrics import cohen_kappa_score
    return cohen_kappa_score(y_true, y_pred)

if __name__ == "__main__":
    import numpy as _np
    y_true = _np.array([1,2,1,3])
    y_pred = _np.array([1,2,3,3])
    print("Kappa:", compute_kappa(y_true, y_pred))
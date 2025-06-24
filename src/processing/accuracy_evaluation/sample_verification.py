#!/usr/bin/env python3.12
# -*- coding: utf-8 -*-
"""
文件: sample_verification.py
模块: src.processing.accuracy_evaluation
功能: 提取并验证ROI掩膜中的有效样本点
作者: 孟诣楠
版本: v1.0.0
最近更新: 2025-06-17
"""
import numpy as np

def extract_valid_samples(class_map: np.ndarray, roi_mask: np.ndarray):
    """
    提取ROI>0的位置作为有效样本
    返回:
        y_true, y_pred, mask_indices
    """
    mask = roi_mask > 0
    return roi_mask[mask], class_map[mask], mask

if __name__ == "__main__":
    import numpy as _np
    class_map = _np.array([[1,0],[2,3]])
    roi_mask = _np.array([[0,2],[2,0]])
    y_true, y_pred, _ = extract_valid_samples(class_map, roi_mask)
    print("y_true:", y_true, "y_pred:", y_pred)
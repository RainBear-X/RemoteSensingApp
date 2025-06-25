#!/usr/bin/env python3.12
# -*- coding: utf-8 -*-
"""
文件: sample_verification.py
模块: src.processing.accuracy_evaluation
功能: 支持ROI掩膜、csv/xlsx表格两种方式的采样点提取与验证
作者: 孟诣楠
版本: v1.1.0
最新更改时间: 2025-06-25
"""
import numpy as np
import pandas as pd
import os

def extract_valid_samples(class_map: np.ndarray, roi_mask: np.ndarray):
    """
    兼容旧逻辑：提取ROI>0的位置作为有效样本
    返回:
        y_true, y_pred, mask_indices
    """
    mask = roi_mask > 0
    return roi_mask[mask], class_map[mask], mask

def load_samples_from_file(file_path: str):
    import pandas as pd, os
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == '.csv':
        df = pd.read_csv(file_path)
    elif ext in ('.xls', '.xlsx'):
        df = pd.read_excel(file_path, engine='openpyxl')
    else:
        raise ValueError(f"不支持的文件格式: {ext}，请用csv或xlsx")
    # 自动映射，注意字段名严格区分
    if 'true_label' not in df.columns and 'CLASSIFIED' in df.columns:
        df['true_label'] = df['CLASSIFIED']
    if 'predicted_label' not in df.columns and 'RASTERVALU' in df.columns:
        df['predicted_label'] = df['RASTERVALU']
    if 'true_label' not in df.columns or 'predicted_label' not in df.columns:
        raise ValueError("样本表必须包含 true_label/CLASSIFIED 和 predicted_label/RASTERVALU 字段")
    # 建议过滤未标注行
    df = df[df['true_label'] != -1]
    if len(df) == 0:
        raise ValueError("点表中没有任何已标注的真值，无法评估！")
    y_true = df['true_label'].to_numpy()
    y_pred = df['predicted_label'].to_numpy()
    return y_true, y_pred


if __name__ == "__main__":
    # 测试旧逻辑
    class_map = np.array([[1,0],[2,3]])
    roi_mask = np.array([[0,2],[2,0]])
    y_true, y_pred, _ = extract_valid_samples(class_map, roi_mask)
    print("掩膜 y_true:", y_true, "y_pred:", y_pred)

    # 测试Excel/csv导入（需有示例文件）
    # y_true, y_pred, indices = load_samples_from_file("test_samples.xlsx")
    # print("xlsx y_true:", y_true, "y_pred:", y_pred)

#!/usr/bin/env python3.12
# -*- coding: utf-8 -*-
"""
文件: evaluate.py
模块: src.processing.accuracy_evaluation
功能: 整合各评估功能，输出图像和报告
作者: 孟诣楠
版本: v1.0.0
最近更新: 2025-06-17
"""
import os
import numpy as np
from .sample_verification      import extract_valid_samples
from .confusion_matrix         import compute_confusion_matrix, plot_confusion_matrix
from .overall_accuracy         import compute_overall_accuracy
from .kappa_coefficient        import compute_kappa
from .evaluation_report        import generate_text_report

def run_evaluation(class_map_path: str, roi_mask_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    class_map = np.load(class_map_path) if class_map_path.endswith('.npy') else __import__('rasterio').open(class_map_path).read(1)
    roi_mask  = np.load(roi_mask_path)
    y_true, y_pred, mask = extract_valid_samples(class_map, roi_mask)
    labels = sorted(set(y_true))
    cm = compute_confusion_matrix(y_true, y_pred, labels)
    plot_confusion_matrix(cm, labels, os.path.join(output_dir, "confusion_matrix.png"))
    oa = compute_overall_accuracy(y_true, y_pred)
    kappa = compute_kappa(y_true, y_pred)
    generate_text_report(
        {'overall_accuracy':oa, 'kappa':kappa},
        {i: i for i in labels},
        os.path.join(output_dir, "evaluation_report.txt")
    )
    print(f"评估完成，总体精度={oa*100:.2f}%，Kappa={kappa:.4f}")

if __name__ == "__main__":
    run_evaluation(
        "../output/segmentation_results/kmeans_class_map.npy",
        "../output/ROI/roi_mask.npy",
        "evaluation_results"
    )
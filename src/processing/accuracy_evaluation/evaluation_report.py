#!/usr/bin/env python3.12
# -*- coding: utf-8 -*-
"""
文件: evaluation_report.py
模块: src.processing.accuracy_evaluation
功能: 汇总各项指标并生成文本/表格报告
作者: 孟诣楠
版本: v1.0.0
最近更新: 2025-06-17
"""
import os

def generate_text_report(metrics: dict, cluster_mapping: dict, output_path: str):
    """
    将 metrics（OA, Kappa, 混淆矩阵等）和聚类映射写入文本报告
    """
    lines = []
    lines.append("==== 遥感影像分类精度评估报告 ====")
    lines.append(f"总体精度：{metrics['overall_accuracy']*100:.2f}%")
    lines.append(f"Kappa系数：{metrics['kappa']:.4f}")
    lines.append("聚类映射：")
    for c,v in cluster_mapping.items():
        lines.append(f"  聚类 {c} -> 类别 {v}")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"报告已保存: {output_path}")

if __name__ == "__main__":
    mt = {'overall_accuracy':0.85,'kappa':0.78}
    cm = {0:1,1:2}
    os.makedirs("test_reports", exist_ok=True)
    generate_text_report(mt, cm, "test_reports/report.txt")
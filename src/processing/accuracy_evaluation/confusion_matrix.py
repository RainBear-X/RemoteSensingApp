#!/usr/bin/env python3.12
# -*- coding: utf-8 -*-
"""
文件: confusion_matrix.py
模块: src.processing.accuracy_evaluation
功能: 计算分类混淆矩阵并绘制/保存为图像
作者: 孟诣楠
版本: v2.0.0
最近更新: 2025-06-17
修订历史:
    v1.0.0: 初始版本
    v2.0.0: 重大更新
        - 增加归一化选项（行、列、全局）
        - 支持自定义类别名称映射
        - 增强可视化（热图注释、百分比显示）
        - 添加详细分类报告生成
        - 支持多种输出格式（CSV、JSON）
        - 优化大规模数据处理性能
        - 增强错误处理和输入验证
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple, Union
import json
import csv
import logging
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import warnings
from matplotlib import cm
from matplotlib.colors import ListedColormap
import pandas as pd


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NormalizeMode(Enum):
    """归一化模式枚举"""
    NONE = 'none'
    TRUE = 'true'  # 按行归一化（召回率）
    PRED = 'pred'  # 按列归一化（精确率）
    ALL = 'all'    # 全局归一化

@dataclass
class ClassificationMetrics:
    """分类指标数据类"""
    precision: Dict[Union[str, int], float]
    recall: Dict[Union[str, int], float]
    f1_score: Dict[Union[str, int], float]
    support: Dict[Union[str, int], int]
    accuracy: float
    macro_avg: Dict[str, float]
    weighted_avg: Dict[str, float]
    micro_avg: Dict[str, float]

class ConfusionMatrixAnalyzer:
    """混淆矩阵分析器主类"""

    def __init__(self,
                 y_true: np.ndarray,
                 y_pred: np.ndarray,
                 labels: Optional[List[Union[int, str]]] = None,
                 label_names: Optional[Dict[Union[int, str], str]] = None,
                 sample_weight: Optional[np.ndarray] = None):
        """
        初始化混淆矩阵分析器
        
        参数:
            y_true: 真实标签
            y_pred: 预测标签
            labels: 类别标签列表，如果为None则自动推断
            label_names: 类别名称映射字典，用于显示
            sample_weight: 样本权重
        """

        # 确保中文显示正常
        from src.utils.chinese_config import setup_chinese_all
        setup_chinese_all()
        
        self.y_true = np.asarray(y_true)
        self.y_pred = np.asarray(y_pred)
        self.sample_weight = sample_weight

        # 验证输入
        self._validate_inputs()

        # 自动推断标签
        if labels is None:
            self.labels = sorted(list(set(np.concatenate([self.y_true, self.y_pred]))))
        else:
            self.labels = list(labels)

        # 设置标签名称映射
        if label_names is None:
            self.label_names = {label: str(label) for label in self.labels}
        else:
            self.label_names = label_names
            # 确保所有标签都有名称
            for label in self.labels:
                if label not in self.label_names:
                    self.label_names[label] = str(label)

        # 计算混淆矩阵
        self.cm = None
        self.cm_normalized = {}
        self.metrics = None

    def _validate_inputs(self):
        """验证输入数据"""
        if len(self.y_true) != len(self.y_pred):
            raise ValueError(f"y_true和y_pred长度不匹配: {len(self.y_true)} vs {len(self.y_pred)}")

        if len(self.y_true) == 0:
            raise ValueError("输入数据为空")

        if self.sample_weight is not None:
            if len(self.sample_weight) != len(self.y_true):
                raise ValueError(f"sample_weight长度与数据不匹配: {len(self.sample_weight)} vs {len(self.y_true)}")
            if np.any(self.sample_weight < 0):
                raise ValueError("sample_weight包含负值")

    def compute_confusion_matrix(self, normalize: NormalizeMode = NormalizeMode.NONE) -> np.ndarray:
        """
        计算混淆矩阵
        
        参数:
            normalize: 归一化模式
            
        返回:
            混淆矩阵
        """
        from sklearn.metrics import confusion_matrix

        # 计算基础混淆矩阵
        if self.cm is None:
            self.cm = confusion_matrix(
                self.y_true,
                self.y_pred,
                labels=self.labels,
                sample_weight=self.sample_weight
            )
            logger.info(f"计算混淆矩阵完成，形状: {self.cm.shape}")

        # 归一化处理
        if normalize == NormalizeMode.NONE:
            return self.cm

        if normalize not in self.cm_normalized:
            cm_normalized = self.cm.astype('float')

            if normalize == NormalizeMode.TRUE:
                # 按行归一化（召回率）
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cm_normalized = cm_normalized / cm_normalized.sum(axis=1, keepdims=True)
            elif normalize == NormalizeMode.PRED:
                # 按列归一化（精确率）
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cm_normalized = cm_normalized / cm_normalized.sum(axis=0, keepdims=True)
            elif normalize == NormalizeMode.ALL:
                # 全局归一化
                cm_normalized = cm_normalized / cm_normalized.sum()

            # 处理NaN值（当某个类别没有样本时）
            cm_normalized = np.nan_to_num(cm_normalized)
            self.cm_normalized[normalize] = cm_normalized

        return self.cm_normalized[normalize]

    def compute_metrics(self) -> ClassificationMetrics:
        """计算详细的分类指标"""
        if self.metrics is not None:
            return self.metrics

        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

        # 确保混淆矩阵已计算
        if self.cm is None:
            self.compute_confusion_matrix()

        # 计算每个类别的指标
        precision = {}
        recall = {}
        f1 = {}
        support = {}

        for i, label in enumerate(self.labels):
            # 使用混淆矩阵计算指标，避免重复计算
            tp = self.cm[i, i]
            fp = self.cm[:, i].sum() - tp
            fn = self.cm[i, :].sum() - tp

            # 精确率
            if tp + fp > 0:
                precision[label] = tp / (tp + fp)
            else:
                precision[label] = 0.0

            # 召回率
            if tp + fn > 0:
                recall[label] = tp / (tp + fn)
            else:
                recall[label] = 0.0

            # F1分数
            if precision[label] + recall[label] > 0:
                f1[label] = 2 * precision[label] * recall[label] / (precision[label] + recall[label])
            else:
                f1[label] = 0.0

            # 支持度（该类别的样本数）
            support[label] = self.cm[i, :].sum()

        # 计算总体精度
        accuracy = accuracy_score(self.y_true, self.y_pred, sample_weight=self.sample_weight)

        # 计算宏平均
        macro_avg = {
            'precision': np.mean(list(precision.values())),
            'recall': np.mean(list(recall.values())),
            'f1_score': np.mean(list(f1.values()))
        }

        # 计算加权平均
        total_support = sum(support.values())
        weighted_avg = {
            'precision': sum(precision[l] * support[l] for l in self.labels) / total_support,
            'recall': sum(recall[l] * support[l] for l in self.labels) / total_support,
            'f1_score': sum(f1[l] * support[l] for l in self.labels) / total_support
        }

        # 计算微平均（等同于总体精度）
        micro_avg = {
            'precision': accuracy,
            'recall': accuracy,
            'f1_score': accuracy
        }

        self.metrics = ClassificationMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1,
            support=support,
            accuracy=accuracy,
            macro_avg=macro_avg,
            weighted_avg=weighted_avg,
            micro_avg=micro_avg
        )

        logger.info(f"分类指标计算完成，总体精度: {accuracy:.4f}")
        return self.metrics

    def plot_confusion_matrix(self,
                              save_path: str,
                              normalize: NormalizeMode = NormalizeMode.NONE,
                              figsize: Tuple[int, int] = (10, 8),
                              cmap: str = 'Blues',
                              show_percentages: bool = True,
                              show_counts: bool = True,
                              font_size: int = 10,
                              dpi: int = 300,
                              title: Optional[str] = None):
        """
        绘制并保存混淆矩阵图像
        
        参数:
            save_path: 保存路径
            normalize: 归一化模式
            figsize: 图像大小
            cmap: 颜色映射
            show_percentages: 是否显示百分比
            show_counts: 是否显示计数
            font_size: 字体大小
            dpi: 图像分辨率
            title: 自定义标题
        """
        # 获取混淆矩阵
        cm = self.compute_confusion_matrix(normalize)

        # 创建图形
        plt.figure(figsize=figsize)

        # 使用seaborn绘制热图
        mask = None
        if normalize != NormalizeMode.NONE:
            # 对于归一化的矩阵，屏蔽0值以获得更好的视觉效果
            mask = cm == 0

        # 准备标签
        display_labels = [self.label_names[label] for label in self.labels]

        # 绘制热图
        ax = sns.heatmap(
            cm,
            annot=False,  # 我们将自定义注释
            fmt='.2f' if normalize != NormalizeMode.NONE else 'd',
            cmap=cmap,
            square=True,
            cbar_kws={'label': '归一化值' if normalize != NormalizeMode.NONE else '样本数'},
            mask=mask,
            linewidths=0.5,
            linecolor='gray'
        )

        # 添加自定义注释
        if show_percentages or show_counts:
            for i in range(len(self.labels)):
                for j in range(len(self.labels)):
                    if normalize == NormalizeMode.NONE:
                        # 显示原始计数
                        text = f'{int(cm[i, j])}'
                    else:
                        # 显示归一化值
                        value = cm[i, j]
                        if show_counts and show_percentages:
                            count = int(self.cm[i, j])
                            text = f'{value:.1%}\n({count})'
                        elif show_percentages:
                            text = f'{value:.1%}'
                        else:
                            count = int(self.cm[i, j])
                            text = f'{count}'

                    # 根据背景颜色选择文本颜色
                    text_color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
                    ax.text(j + 0.5, i + 0.5, text,
                            ha='center', va='center',
                            color=text_color, fontsize=font_size)

        # 设置标签
        ax.set_xticklabels(display_labels, rotation=45, ha='right')
        ax.set_yticklabels(display_labels, rotation=0)

        # 设置标题和轴标签
        if title is None:
            if normalize == NormalizeMode.NONE:
                title = "混淆矩阵"
            elif normalize == NormalizeMode.TRUE:
                title = "归一化混淆矩阵（按真实类别）"
            elif normalize == NormalizeMode.PRED:
                title = "归一化混淆矩阵（按预测类别）"
            else:
                title = "归一化混淆矩阵（全局）"

        plt.title(title, fontsize=font_size + 2, pad=20)
        plt.xlabel('预测标签', fontsize=font_size + 1)
        plt.ylabel('真实标签', fontsize=font_size + 1)

        # 调整布局
        plt.tight_layout()

        # 保存图像
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"混淆矩阵图像已保存到: {save_path}")

    def generate_classification_report(self) -> str:
        """生成详细的分类报告"""
        metrics = self.compute_metrics()

        # 构建报告
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("分类报告")
        report_lines.append("=" * 80)
        report_lines.append("")

        # 表头
        header = f"{'类别':>15} {'精确率':>10} {'召回率':>10} {'F1分数':>10} {'支持度':>10}"
        report_lines.append(header)
        report_lines.append("-" * 80)

        # 每个类别的指标
        for label in self.labels:
            name = self.label_names[label]
            line = f"{name:>15} {metrics.precision[label]:>10.4f} {metrics.recall[label]:>10.4f} "
            line += f"{metrics.f1_score[label]:>10.4f} {metrics.support[label]:>10d}"
            report_lines.append(line)

        report_lines.append("-" * 80)

        # 平均值
        report_lines.append(f"{'宏平均':>15} {metrics.macro_avg['precision']:>10.4f} "
                            f"{metrics.macro_avg['recall']:>10.4f} "
                            f"{metrics.macro_avg['f1_score']:>10.4f} {sum(metrics.support.values()):>10d}")

        report_lines.append(f"{'加权平均':>15} {metrics.weighted_avg['precision']:>10.4f} "
                            f"{metrics.weighted_avg['recall']:>10.4f} "
                            f"{metrics.weighted_avg['f1_score']:>10.4f} {sum(metrics.support.values()):>10d}")

        report_lines.append(f"{'微平均':>15} {metrics.micro_avg['precision']:>10.4f} "
                            f"{metrics.micro_avg['recall']:>10.4f} "
                            f"{metrics.micro_avg['f1_score']:>10.4f} {sum(metrics.support.values()):>10d}")

        report_lines.append("")
        report_lines.append(f"总体精度: {metrics.accuracy:.4f}")
        report_lines.append("=" * 80)

        return "\n".join(report_lines)

    def save_results(self,
                     output_dir: str,
                     formats: List[str] = ['png', 'csv', 'json', 'txt'],
                     prefix: str = 'confusion_matrix'):
        """
        保存结果到多种格式
        
        参数:
            output_dir: 输出目录
            formats: 输出格式列表
            prefix: 文件名前缀
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 保存混淆矩阵图像
        if 'png' in formats:
            self.plot_confusion_matrix(
                str(output_path / f"{prefix}.png"),
                normalize=NormalizeMode.NONE
            )
            self.plot_confusion_matrix(
                str(output_path / f"{prefix}_normalized.png"),
                normalize=NormalizeMode.TRUE
            )

        # 保存CSV格式
        if 'csv' in formats:
            df = pd.DataFrame(
                self.cm,
                index=[self.label_names[l] for l in self.labels],
                columns=[self.label_names[l] for l in self.labels]
            )
            df.to_csv(output_path / f"{prefix}.csv")
            logger.info(f"混淆矩阵CSV已保存到: {output_path / f'{prefix}.csv'}")

        # 保存JSON格式
        if 'json' in formats:
            metrics = self.compute_metrics()
            result = {
                'confusion_matrix': self.cm.tolist(),
                'labels': self.labels,
                'label_names': self.label_names,
                'metrics': {
                    'accuracy': metrics.accuracy,
                    'precision': metrics.precision,
                    'recall': metrics.recall,
                    'f1_score': metrics.f1_score,
                    'support': metrics.support,
                    'macro_avg': metrics.macro_avg,
                    'weighted_avg': metrics.weighted_avg,
                    'micro_avg': metrics.micro_avg
                }
            }
            with open(output_path / f"{prefix}.json", 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logger.info(f"结果JSON已保存到: {output_path / f'{prefix}.json'}")

        # 保存文本报告
        if 'txt' in formats:
            report = self.generate_classification_report()
            with open(output_path / f"{prefix}_report.txt", 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"分类报告已保存到: {output_path / f'{prefix}_report.txt'}")

# 保留原有接口以确保兼容性
def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: list) -> np.ndarray:
    """
    计算混淆矩阵（保留原有接口）
    参数:
        y_true (np.ndarray): 真实标签（1D）
        y_pred (np.ndarray): 预测标签（1D）
        labels (list): 类别列表，如 [1,2,3]
    返回:
        cm (np.ndarray): 混淆矩阵
    """
    analyzer = ConfusionMatrixAnalyzer(y_true, y_pred, labels)
    return analyzer.compute_confusion_matrix()

def plot_confusion_matrix(cm: np.ndarray, labels: list, save_path: str):
    """
    绘制并保存混淆矩阵图像（保留原有接口）
    """
    # 创建一个临时分析器来使用新的绘图功能
    # 由于我们已经有了混淆矩阵，需要反向创建数据
    n_samples = 1000
    y_true = []
    y_pred = []

    for i in range(len(labels)):
        for j in range(len(labels)):
            count = int(cm[i, j])
            y_true.extend([labels[i]] * count)
            y_pred.extend([labels[j]] * count)

    analyzer = ConfusionMatrixAnalyzer(
        np.array(y_true),
        np.array(y_pred),
        labels
    )
    analyzer.cm = cm  # 直接使用提供的混淆矩阵
    analyzer.plot_confusion_matrix(save_path)

if __name__ == "__main__":
    # 扩展的单元测试示例
    import os

    # 测试1：基本功能（保持向后兼容）
    print("测试1：基本功能")
    y_true = np.array([1, 2, 1, 3, 2, 3, 1, 2, 3, 3, 2, 1])
    y_pred = np.array([1, 2, 3, 3, 1, 2, 1, 2, 3, 2, 2, 1])
    labels = [1, 2, 3]

    cm = compute_confusion_matrix(y_true, y_pred, labels)
    os.makedirs("test_outputs", exist_ok=True)
    plot_confusion_matrix(cm, labels, "test_outputs/confusion_matrix.png")
    print("混淆矩阵:", cm)

    # 测试2：使用新的分析器类
    print("\n测试2：高级功能")

    # 创建带有类别名称的分析器
    label_names = {1: "类别A", 2: "类别B", 3: "类别C"}
    analyzer = ConfusionMatrixAnalyzer(y_true, y_pred, labels, label_names)

    # 生成各种格式的输出
    analyzer.save_results("test_outputs", formats=['png', 'csv', 'json', 'txt'])

    # 打印分类报告
    print("\n" + analyzer.generate_classification_report())

    # 测试3：带权重的分类
    print("\n测试3：带权重的分类")
    sample_weight = np.random.rand(len(y_true))
    weighted_analyzer = ConfusionMatrixAnalyzer(
        y_true, y_pred, labels, label_names, sample_weight
    )

    # 绘制归一化的混淆矩阵
    weighted_analyzer.plot_confusion_matrix(
        "test_outputs/confusion_matrix_weighted.png",
        normalize=NormalizeMode.TRUE,
        show_percentages=True,
        show_counts=True
    )

    print("测试完成！请查看 test_outputs 目录中的结果。")
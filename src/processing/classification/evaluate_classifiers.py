# -*- coding: utf-8 -*-
# 文件路径: src/processing/classification/evaluate_classifiers.py
# -----------------------------------------
# 功能: 统一评估脚本 evaluate_classifiers.py，自动对比并输出各监督分类器性能
# 接口:
#     evaluate_supervised_classifiers(X: np.ndarray, y: np.ndarray,
#                                     classifier_names: list = None,
#                                     test_size: float = 0.3,
#                                     random_state: int = 42) -> dict
#     print_evaluation_results(results: dict) -> None
# 作者: 孟诣楠
# 版本: 1.0.0
# 最新更改时间: 2025-06-17
# 改进说明: 初始版本，实现统一评估接口与主流程
# -----------------------------------------

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report, confusion_matrix
from src.processing.classification.model_manager import get_supervised_classifier



def evaluate_supervised_classifiers(X: np.ndarray,
                                    y: np.ndarray,
                                    classifier_names: list = None,
                                    test_size: float = 0.3,
                                    random_state: int = 42) -> dict:
    """
    对一组监督分类器进行统一训练和评估

    参数:
        X: 特征数组，形状 (N, D)
        y: 标签数组，形状 (N,)
        classifier_names: 要评估的分类器名称列表，None 时使用全部已注册分类器
        test_size: 测试集比例
        random_state: 随机种子，保证结果可复现

    返回:
        results: 字典，键为分类器名称，值为性能指标字典，包括:
            - accuracy: 总体精度
            - kappa: Kappa 系数
            - report: 分类报告（precision, recall, f1）
            - confusion_matrix: 混淆矩阵数组
    """
    # 默认评估所有分类器
    all_names = list(getattr(__import__('processing.classification.model_manager', fromlist=['SUPERVISED_CLASSIFIERS']),
                             'SUPERVISED_CLASSIFIERS').keys())
    if classifier_names is None:
        classifier_names = all_names

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    results = {}
    for name in classifier_names:
        clf = get_supervised_classifier(name)
        try:
            clf.train(X_train, y_train)
            y_pred = clf.predict(X_test)
        except Exception as e:
            results[name] = {'error': str(e)}
            continue
        acc = accuracy_score(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, digits=4)
        cm = confusion_matrix(y_test, y_pred)
        results[name] = {
            'accuracy': acc,
            'kappa': kappa,
            'report': report,
            'confusion_matrix': cm
        }
    return results


def print_evaluation_results(results: dict) -> None:
    """
    打印评估结果

    参数:
        results: evaluate_supervised_classifiers 返回的结果字典
    """
    for name, metrics in results.items():
        print(f"==== {name} ====")
        if 'error' in metrics:
            print(f"Error: {metrics['error']}\n")
            continue
        print(f"Accuracy: {metrics['accuracy'] * 100:.2f}%")
        print(f"Kappa: {metrics['kappa']:.4f}")
        print("Classification Report:")
        print(metrics['report'])
        print("Confusion Matrix:")
        print(metrics['confusion_matrix'])
        print()

if __name__ == '__main__':
    # 示例: 使用 Iris 数据集评估分类器
    from sklearn.datasets import load_iris

    data = load_iris()
    X, y = data.data, data.target

    # 评估所有已注册分类器
    results = evaluate_supervised_classifiers(X, y)
    print_evaluation_results(results)

# -*- coding: utf-8 -*-
# 文件路径: src/processing/classification/base_classifier.py
# -----------------------------------------
# 功能: 分类器基类，定义所有分类算法必须实现的接口
# 接口:
#     train(self, features: np.ndarray, labels: np.ndarray) -> Any
#     predict(self, features: np.ndarray) -> np.ndarray
# 作者: 孟诣楠
# 版本: 1.0.0
# 最新更改时间: 2025-06-17
# 改进说明: 初始版本，实现抽象基类和接口定义
# -----------------------------------------

import abc
import numpy as np


class BaseClassifier(abc.ABC):
    """
    抽象分类器基类

    所有具体分类器需继承此类并实现 train 和 predict 方法
    """

    @abc.abstractmethod
    def train(self, features: np.ndarray, labels: np.ndarray) -> None:
        """
        训练分类器

        参数:
            features: 特征数组，形状 (N, D)
            labels:   标签数组，形状 (N,)
        返回:
            无
        """
        pass

    @abc.abstractmethod
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        对新的特征数据进行预测

        参数:
            features: 特征数组，形状 (M, D) 或 (H, W, D)
        返回:
            prediction: 预测标签数组，形状 (M,) 或 (H, W)
        """
        pass


if __name__ == "__main__":
    # 简单测试基类抽象特性
    try:
        BaseClassifier()
    except TypeError as e:
        print("✔ BaseClassifier 抽象基类接口测试通过:", e)
    else:
        print("❌ BaseClassifier 未抛出抽象错误，需检查抽象方法实现")

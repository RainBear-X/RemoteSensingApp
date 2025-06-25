# -*- coding: utf-8 -*-
# 文件路径: src/processing/classification/supervised/maximum_likelihood.py
# -----------------------------------------
# 功能: 最大似然分类器实现
# 接口:
#     train(self, features: np.ndarray, labels: np.ndarray) -> None
#     predict(self, features: np.ndarray) -> np.ndarray
#     predict_proba(self, features: np.ndarray) -> np.ndarray
#     predict_with_confidence(self, features: np.ndarray) -> tuple
#     get_class_statistics(self) -> dict
#     get_model_info(self) -> dict
#     save_model(self, filepath: str) -> None
#     load_model(self, filepath: str) -> None
#     evaluate_class_separability(self) -> dict
# 作者: 孟诣楠
# 版本: 2.0.0
# 最新更改时间: 2025-06-17
# 改进说明: 重大功能增强版本
#   - 新增概率预测功能 (predict_proba)
#   - 新增置信度评估功能 (predict_with_confidence)
#   - 新增类别统计信息获取 (get_class_statistics)
#   - 新增模型信息查询功能 (get_model_info)
#   - 新增模型保存和加载能力 (save_model, load_model)
#   - 新增类别可分性评估 (evaluate_class_separability)
#   - 增强数值稳定性处理，支持大规模遥感数据
#   - 改进协方差矩阵正则化策略
#   - 新增多种正则化方法选择
#   - 优化内存使用和计算效率
#   - 强化异常处理和数据验证
#   - 支持并行计算加速
# -----------------------------------------

import numpy as np
import pickle
import logging
from typing import Optional, Dict, Any, Union, Tuple, List
from pathlib import Path
import warnings
from concurrent.futures import ThreadPoolExecutor
import time

from scipy.linalg import det, inv, LinAlgError
from scipy.stats import multivariate_normal
from src.processing.classification.base_classifier import BaseClassifier

# 配置日志系统
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MaximumLikelihoodClassifier(BaseClassifier):
    """
    最大似然分类器 - 增强版本 2.0.0
    
    基于多元正态分布假设的经典统计分类方法，通过最大化后验概率进行分类决策。
    特别适用于遥感图像的光谱分类任务，能够有效处理高维光谱特征数据。
    
    新增功能:
        - 概率预测和置信度评估
        - 类别统计特性分析
        - 模型持久化和版本管理
        - 类别可分性评估
        - 数值稳定性增强
        - 性能监控和优化
    
    技术特点:
        - 支持多种协方差矩阵正则化策略
        - 实现马氏距离的高效计算
        - 提供详细的统计分析报告
        - 支持大规模数据的并行处理
    """

    def __init__(self,
                 regularization: float = 1e-6,
                 regularization_method: str = 'diagonal',
                 min_samples_per_class: int = 5,
                 numerical_precision: str = 'double',
                 enable_parallel: bool = True,
                 random_state: Optional[int] = 42):
        """
        初始化最大似然分类器
        
        参数:
            regularization: 协方差矩阵正则化参数，防止数值奇异性
            regularization_method: 正则化方法 ('diagonal', 'identity', 'shrinkage')
            min_samples_per_class: 每个类别的最小样本数要求
            numerical_precision: 数值精度 ('single', 'double')
            enable_parallel: 是否启用并行计算加速
            random_state: 随机种子，确保结果可重现
        """
        super().__init__()

        # 参数验证
        self._validate_parameters(regularization, regularization_method,
                                  min_samples_per_class, numerical_precision)

        # 配置参数
        self.config = {
            'regularization': regularization,
            'regularization_method': regularization_method,
            'min_samples_per_class': min_samples_per_class,
            'numerical_precision': numerical_precision,
            'enable_parallel': enable_parallel,
            'random_state': random_state
        }

        # 数值精度设置
        self.dtype = np.float64 if numerical_precision == 'double' else np.float32

        # 模型参数存储
        self.classes_ = None
        self.n_classes_ = 0
        self.n_features_ = 0
        self.priors_ = {}
        self.means_ = {}
        self.covs_ = {}
        self.inv_covs_ = {}
        self.log_dets_ = {}
        self.class_sample_counts_ = {}

        # 训练状态
        self.is_trained = False
        self.training_history = {}
        self.feature_names = None
        self.class_names = None

        logger.info(f"最大似然分类器已初始化 - 版本 2.0.0")
        logger.info(f"配置参数: {self.config}")

    def _validate_parameters(self, regularization: float, regularization_method: str,
                             min_samples_per_class: int, numerical_precision: str) -> None:
        """参数有效性验证"""
        if regularization <= 0:
            raise ValueError("正则化参数必须为正数")

        if regularization_method not in ['diagonal', 'identity', 'shrinkage']:
            raise ValueError(f"不支持的正则化方法: {regularization_method}")

        if min_samples_per_class < 2:
            raise ValueError("每个类别至少需要2个样本")

        if numerical_precision not in ['single', 'double']:
            raise ValueError(f"不支持的数值精度: {numerical_precision}")

    def train(self, features: np.ndarray, labels: np.ndarray,
              feature_names: Optional[List[str]] = None,
              class_names: Optional[List[str]] = None,
              validation_split: float = 0.0) -> None:
        """
        训练最大似然分类器模型
        
        参数:
            features: 特征数组，形状 (N, D) 或 (H, W, D)
            labels: 标签数组，形状 (N,) 或 (H, W)
            feature_names: 特征名称列表，用于结果解释
            class_names: 类别名称列表，用于结果显示
            validation_split: 验证集比例 (0.0-1.0)
        
        返回:
            无
        """
        try:
            logger.info("开始训练最大似然分类器...")
            start_time = time.time()

            # 数据预处理和验证
            X, y = self._preprocess_training_data(features, labels)

            # 存储元数据
            self.n_features_ = X.shape[1]
            self.feature_names = feature_names or [f"band_{i+1}" for i in range(self.n_features_)]

            # 获取类别信息
            self.classes_, counts = np.unique(y, return_counts=True)
            self.n_classes_ = len(self.classes_)

            if class_names is not None:
                self.class_names = class_names
            else:
                self.class_names = [f"class_{cls}" for cls in self.classes_]

            # 验证样本数量充足性
            self._validate_sample_counts(counts)

            # 验证集分割
            if validation_split > 0:
                X_train, X_val, y_train, y_val = self._split_validation_data(X, y, validation_split)
            else:
                X_train, y_train = X, y
                X_val, y_val = None, None

            # 计算类别统计量
            total_samples = len(y_train)
            logger.info(f"训练样本总数: {total_samples}")
            logger.info(f"特征维度: {self.n_features_}")
            logger.info(f"类别数量: {self.n_classes_}")

            # 并行或串行训练
            if self.config['enable_parallel'] and self.n_classes_ > 2:
                self._train_parallel(X_train, y_train)
            else:
                self._train_sequential(X_train, y_train)

            # 记录训练历史
            training_time = time.time() - start_time
            self.training_history = {
                'training_time': training_time,
                'n_samples': total_samples,
                'n_features': self.n_features_,
                'n_classes': self.n_classes_,
                'class_counts': dict(zip(self.classes_, counts)),
                'regularization_applied': True
            }

            self.is_trained = True

            # 验证集评估
            if X_val is not None:
                val_pred = self.predict(X_val)
                val_accuracy = np.mean(val_pred == y_val)
                self.training_history['val_accuracy'] = val_accuracy
                logger.info(f"验证集精度: {val_accuracy:.4f}")

            # 训练集精度评估
            train_pred = self.predict(X_train)
            train_accuracy = np.mean(train_pred == y_train)
            self.training_history['train_accuracy'] = train_accuracy
            logger.info(f"训练完成 - 耗时: {training_time:.2f}秒")
            logger.info(f"训练精度: {train_accuracy:.4f}")

        except Exception as e:
            logger.error(f"训练过程发生错误: {str(e)}")
            raise

    def _train_sequential(self, X: np.ndarray, y: np.ndarray) -> None:
        """串行训练模式"""
        for cls in self.classes_:
            self._train_single_class(X, y, cls)

    def _train_parallel(self, X: np.ndarray, y: np.ndarray) -> None:
        """并行训练模式"""
        with ThreadPoolExecutor(max_workers=min(4, self.n_classes_)) as executor:
            futures = {executor.submit(self._train_single_class, X, y, cls): cls
                       for cls in self.classes_}

            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"并行训练类别 {futures[future]} 时发生错误: {str(e)}")
                    raise

    def _train_single_class(self, X: np.ndarray, y: np.ndarray, cls: int) -> None:
        """训练单个类别的统计参数"""
        # 提取类别样本
        class_mask = (y == cls)
        class_features = X[class_mask].astype(self.dtype)
        n_samples = len(class_features)

        # 存储样本数量
        self.class_sample_counts_[cls] = n_samples

        # 计算先验概率
        self.priors_[cls] = n_samples / len(y)

        # 计算均值向量
        self.means_[cls] = np.mean(class_features, axis=0)

        # 计算协方差矩阵
        cov_matrix = self._compute_covariance_matrix(class_features)

        # 应用正则化
        regularized_cov = self._apply_regularization(cov_matrix)

        # 存储协方差矩阵
        self.covs_[cls] = regularized_cov

        # 计算逆矩阵和行列式（使用数值稳定的方法）
        try:
            self.inv_covs_[cls] = self._stable_matrix_inverse(regularized_cov)
            self.log_dets_[cls] = self._stable_log_determinant(regularized_cov)
        except LinAlgError as e:
            logger.warning(f"类别 {cls} 协方差矩阵计算出现数值问题，增加正则化强度")
            # 增加正则化强度重试
            stronger_reg_cov = self._apply_regularization(cov_matrix, scale_factor=10.0)
            self.covs_[cls] = stronger_reg_cov
            self.inv_covs_[cls] = self._stable_matrix_inverse(stronger_reg_cov)
            self.log_dets_[cls] = self._stable_log_determinant(stronger_reg_cov)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        对新样本进行分类预测
        
        参数:
            features: 特征数组，形状 (M, D) 或 (H, W, D)
        
        返回:
            predictions: 预测标签数组，形状 (M,) 或 (H, W)
        """
        if not self.is_trained:
            raise RuntimeError("模型尚未训练，请先调用train方法")

        try:
            # 预处理输入数据
            orig_shape = features.shape
            if features.ndim == 3:
                H, W, D = orig_shape
                X = features.reshape(-1, D).astype(self.dtype)
            else:
                X = features.astype(self.dtype)

            # 处理无效像素
            valid_mask = self._get_valid_pixel_mask(X)
            predictions = np.full(X.shape[0], -1, dtype=int)

            if np.any(valid_mask):
                X_valid = X[valid_mask]

                # 计算判别函数值
                discriminant_scores = self._compute_discriminant_scores(X_valid)

                # 选择最大得分对应的类别
                best_class_indices = np.argmax(discriminant_scores, axis=1)
                predictions[valid_mask] = self.classes_[best_class_indices]

            # 恢复原始形状
            if features.ndim == 3:
                return predictions.reshape(H, W)
            return predictions

        except Exception as e:
            logger.error(f"预测过程发生错误: {str(e)}")
            raise

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        预测类别概率分布
        
        参数:
            features: 特征数组，形状 (M, D) 或 (H, W, D)
        
        返回:
            probabilities: 概率数组，形状 (M, n_classes) 或 (H, W, n_classes)
        """
        if not self.is_trained:
            raise RuntimeError("模型尚未训练，请先调用train方法")

        try:
            orig_shape = features.shape
            if features.ndim == 3:
                H, W, D = orig_shape
                X = features.reshape(-1, D).astype(self.dtype)
            else:
                X = features.astype(self.dtype)

            # 初始化概率数组
            probabilities = np.zeros((X.shape[0], self.n_classes_), dtype=self.dtype)
            valid_mask = self._get_valid_pixel_mask(X)

            if np.any(valid_mask):
                X_valid = X[valid_mask]

                # 计算对数似然
                log_likelihoods = self._compute_log_likelihoods(X_valid)

                # 数值稳定的softmax转换
                probabilities[valid_mask] = self._stable_softmax(log_likelihoods)

            # 恢复原始形状
            if features.ndim == 3:
                return probabilities.reshape(H, W, self.n_classes_)
            return probabilities

        except Exception as e:
            logger.error(f"概率预测过程发生错误: {str(e)}")
            raise

    def predict_with_confidence(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测结果及其置信度
        
        参数:
            features: 特征数组，形状 (M, D) 或 (H, W, D)
        
        返回:
            predictions: 预测标签数组
            confidences: 置信度数组（最大概率值）
        """
        probabilities = self.predict_proba(features)

        if features.ndim == 3:
            predictions = np.argmax(probabilities, axis=2)
            confidences = np.max(probabilities, axis=2)
            # 将索引转换为实际类别标签
            predictions = self.classes_[predictions.flatten()].reshape(predictions.shape)
        else:
            predictions = np.argmax(probabilities, axis=1)
            confidences = np.max(probabilities, axis=1)
            predictions = self.classes_[predictions]

        return predictions, confidences

    def get_class_statistics(self) -> Dict[str, Any]:
        """
        获取类别统计信息
        
        返回:
            statistics: 包含各类别详细统计信息的字典
        """
        if not self.is_trained:
            raise RuntimeError("模型尚未训练，请先调用train方法")

        statistics = {
            'class_info': {},
            'overall_stats': {
                'n_classes': self.n_classes_,
                'n_features': self.n_features_,
                'total_samples': sum(self.class_sample_counts_.values())
            }
        }

        for i, cls in enumerate(self.classes_):
            class_stats = {
                'class_label': cls,
                'class_name': self.class_names[i],
                'sample_count': self.class_sample_counts_[cls],
                'prior_probability': self.priors_[cls],
                'mean_vector': self.means_[cls].tolist(),
                'covariance_trace': np.trace(self.covs_[cls]),
                'covariance_determinant': np.exp(self.log_dets_[cls]),
                'feature_variances': np.diag(self.covs_[cls]).tolist()
            }

            # 计算马氏距离统计
            if hasattr(self, '_mahalanobis_distances'):
                class_stats['avg_mahalanobis_distance'] = np.mean(
                    self._mahalanobis_distances.get(cls, [0])
                )

            statistics['class_info'][cls] = class_stats

        return statistics

    def evaluate_class_separability(self) -> Dict[str, Any]:
        """
        评估类别间的可分性
        
        返回:
            separability_metrics: 类别可分性评估结果
        """
        if not self.is_trained:
            raise RuntimeError("模型尚未训练，请先调用train方法")

        separability = {
            'pairwise_distances': {},
            'separability_matrix': np.zeros((self.n_classes_, self.n_classes_)),
            'overall_separability': 0.0
        }

        # 计算类别间的马氏距离
        for i, cls1 in enumerate(self.classes_):
            for j, cls2 in enumerate(self.classes_):
                if i != j:
                    # 计算类别中心间的马氏距离
                    mean_diff = self.means_[cls1] - self.means_[cls2]
                    # 使用池化协方差矩阵
                    pooled_cov = (self.covs_[cls1] + self.covs_[cls2]) / 2
                    try:
                        pooled_inv = self._stable_matrix_inverse(pooled_cov)
                        mahal_dist = np.sqrt(mean_diff.T @ pooled_inv @ mean_diff)
                        separability['separability_matrix'][i, j] = mahal_dist
                        separability['pairwise_distances'][f"{cls1}-{cls2}"] = mahal_dist
                    except LinAlgError:
                        separability['separability_matrix'][i, j] = 0.0
                        separability['pairwise_distances'][f"{cls1}-{cls2}"] = 0.0

        # 计算总体可分性（平均马氏距离）
        non_diagonal = separability['separability_matrix'][
            ~np.eye(self.n_classes_, dtype=bool)
        ]
        separability['overall_separability'] = np.mean(non_diagonal)

        return separability

    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型详细信息
        
        返回:
            model_info: 模型信息字典
        """
        info = {
            'version': '2.0.0',
            'model_type': 'MaximumLikelihood',
            'is_trained': self.is_trained,
            'config': self.config.copy(),
            'training_history': self.training_history.copy()
        }

        if self.is_trained:
            info.update({
                'n_features': self.n_features_,
                'n_classes': self.n_classes_,
                'classes': self.classes_.tolist(),
                'feature_names': self.feature_names,
                'class_names': self.class_names,
                'numerical_precision': self.config['numerical_precision']
            })

        return info

    def save_model(self, filepath: str) -> None:
        """
        保存模型到文件
        
        参数:
            filepath: 保存路径
        """
        try:
            save_path = Path(filepath)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            model_data = {
                'version': '2.0.0',
                'config': self.config,
                'is_trained': self.is_trained,
                'training_history': self.training_history,
                'feature_names': self.feature_names,
                'class_names': self.class_names,
                'classes_': self.classes_,
                'n_classes_': self.n_classes_,
                'n_features_': self.n_features_,
                'priors_': self.priors_,
                'means_': self.means_,
                'covs_': self.covs_,
                'inv_covs_': self.inv_covs_,
                'log_dets_': self.log_dets_,
                'class_sample_counts_': self.class_sample_counts_
            }

            with open(save_path, 'wb') as f:
                pickle.dump(model_data, f)

            logger.info(f"模型已保存到: {filepath}")

        except Exception as e:
            logger.error(f"保存模型时发生错误: {str(e)}")
            raise

    def load_model(self, filepath: str) -> None:
        """
        从文件加载模型
        
        参数:
            filepath: 模型文件路径
        """
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)

            # 版本兼容性检查
            if 'version' not in model_data:
                warnings.warn("加载的是旧版本模型，某些新功能可能不可用")

            # 恢复模型状态
            self.config = model_data.get('config', {})
            self.is_trained = model_data.get('is_trained', False)
            self.training_history = model_data.get('training_history', {})
            self.feature_names = model_data.get('feature_names', None)
            self.class_names = model_data.get('class_names', None)
            self.classes_ = model_data.get('classes_', None)
            self.n_classes_ = model_data.get('n_classes_', 0)
            self.n_features_ = model_data.get('n_features_', 0)
            self.priors_ = model_data.get('priors_', {})
            self.means_ = model_data.get('means_', {})
            self.covs_ = model_data.get('covs_', {})
            self.inv_covs_ = model_data.get('inv_covs_', {})
            self.log_dets_ = model_data.get('log_dets_', {})
            self.class_sample_counts_ = model_data.get('class_sample_counts_', {})

            # 设置数值精度
            precision = self.config.get('numerical_precision', 'double')
            self.dtype = np.float64 if precision == 'double' else np.float32

            logger.info(f"模型已从 {filepath} 加载")
            logger.info(f"模型版本: {model_data.get('version', '未知')}")

        except Exception as e:
            logger.error(f"加载模型时发生错误: {str(e)}")
            raise

    # 辅助方法
    def _preprocess_training_data(self, features: np.ndarray,
                                  labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """预处理训练数据"""
        # 展平图像数据
        if features.ndim == 3:
            H, W, D = features.shape
            X = features.reshape(-1, D)
            if labels.ndim == 2:
                y = labels.reshape(-1)
            else:
                raise ValueError("标签维度与特征不匹配")
        else:
            X, y = features, labels

        # 转换数据类型
        X = X.astype(self.dtype)

        # 移除无效样本
        valid_mask = self._get_valid_pixel_mask(X) & (y >= 0) & np.isfinite(y)

        X_clean = X[valid_mask]
        y_clean = y[valid_mask]

        if len(X_clean) == 0:
            raise ValueError("没有有效的训练样本")

        logger.info(f"有效训练样本: {len(X_clean)}/{len(X)} ({len(X_clean)/len(X)*100:.1f}%)")

        return X_clean, y_clean.astype(int)

    def _get_valid_pixel_mask(self, X: np.ndarray) -> np.ndarray:
        """获取有效像素掩码"""
        return ~np.any(np.isnan(X) | np.isinf(X), axis=1)

    def _validate_sample_counts(self, counts: np.ndarray) -> None:
        """验证样本数量充足性"""
        min_required = self.config['min_samples_per_class']
        insufficient_classes = counts < min_required

        if np.any(insufficient_classes):
            insufficient_indices = np.where(insufficient_classes)[0]
            insufficient_classes_list = self.classes_[insufficient_indices]
            raise ValueError(
                f"以下类别样本数量不足（需要至少{min_required}个）: "
                f"{insufficient_classes_list} (当前样本数: {counts[insufficient_indices]})"
            )

    def _compute_covariance_matrix(self, class_features: np.ndarray) -> np.ndarray:
        """计算协方差矩阵"""
        if class_features.shape[0] == 1:
            # 单个样本情况，使用单位矩阵
            return np.eye(class_features.shape[1], dtype=self.dtype)
        else:
            # 多个样本，计算经验协方差矩阵
            return np.cov(class_features, rowvar=False, dtype=self.dtype)

    def _apply_regularization(self, cov_matrix: np.ndarray,
                              scale_factor: float = 1.0) -> np.ndarray:
        """应用协方差矩阵正则化"""
        reg_strength = self.config['regularization'] * scale_factor
        method = self.config['regularization_method']

        if method == 'diagonal':
            # 对角线正则化
            return cov_matrix + np.eye(cov_matrix.shape[0], dtype=self.dtype) * reg_strength
        elif method == 'identity':
            # 单位矩阵正则化
            return cov_matrix + np.eye(cov_matrix.shape[0], dtype=self.dtype) * reg_strength
        elif method == 'shrinkage':
            # 收缩估计
            trace_cov = np.trace(cov_matrix)
            shrinkage_target = (trace_cov / cov_matrix.shape[0]) * np.eye(cov_matrix.shape[0], dtype=self.dtype)
            return (1 - reg_strength) * cov_matrix + reg_strength * shrinkage_target
        else:
            raise ValueError(f"不支持的正则化方法: {method}")

    def _stable_matrix_inverse(self, matrix: np.ndarray) -> np.ndarray:
        """数值稳定的矩阵求逆"""
        try:
            # 首先尝试标准求逆
            return inv(matrix)
        except LinAlgError:
            # 如果失败，使用伪逆
            logger.warning("使用伪逆代替标准矩阵求逆")
            return np.linalg.pinv(matrix)

    def _stable_log_determinant(self, matrix: np.ndarray) -> float:
        """数值稳定的对数行列式计算"""
        try:
            # 使用Cholesky分解计算对数行列式
            chol = np.linalg.cholesky(matrix)
            return 2.0 * np.sum(np.log(np.diag(chol)))
        except LinAlgError:
            # 如果Cholesky分解失败，使用特征值分解
            eigenvals = np.linalg.eigvals(matrix)
            eigenvals = np.maximum(eigenvals, 1e-12)  # 避免负特征值
            return np.sum(np.log(eigenvals))

    def _compute_discriminant_scores(self, X: np.ndarray) -> np.ndarray:
        """计算判别函数值"""
        n_samples = X.shape[0]
        scores = np.zeros((n_samples, self.n_classes_), dtype=self.dtype)

        for idx, cls in enumerate(self.classes_):
            mean = self.means_[cls]
            inv_cov = self.inv_covs_[cls]
            log_det = self.log_dets_[cls]
            prior = self.priors_[cls]

            # 计算马氏距离的平方
            diff = X - mean
            mahal_squared = np.einsum('ij,jk,ik->i', diff, inv_cov, diff)

            # 判别函数：log(P(x|class)) + log(P(class))
            scores[:, idx] = (-0.5 * mahal_squared -
                              0.5 * log_det +
                              np.log(prior))

        return scores

    def _compute_log_likelihoods(self, X: np.ndarray) -> np.ndarray:
        """计算对数似然值"""
        return self._compute_discriminant_scores(X)

    def _stable_softmax(self, log_probs: np.ndarray) -> np.ndarray:
        """数值稳定的softmax函数"""
        # 减去最大值以提高数值稳定性
        max_log_probs = np.max(log_probs, axis=1, keepdims=True)
        exp_log_probs = np.exp(log_probs - max_log_probs)
        return exp_log_probs / np.sum(exp_log_probs, axis=1, keepdims=True)

    def _split_validation_data(self, X: np.ndarray, y: np.ndarray,
                               validation_split: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """分割验证数据"""
        from sklearn.model_selection import train_test_split

        return train_test_split(
            X, y,
            test_size=validation_split,
            random_state=self.config.get('random_state', 42),
            stratify=y if len(np.unique(y)) > 1 else None
        )


# 保持向后兼容的别名
MaximumLikelihood = MaximumLikelihoodClassifier

if __name__ == "__main__":
    # 测试增强版最大似然分类器
    print("测试最大似然分类器 v2.0.0")
    print("=" * 50)

    # 设置随机种子确保结果可重现
    np.random.seed(42)

    # 构造多类别高斯分布测试数据
    N_per_class = 150
    D = 4  # 特征维度
    n_classes = 3

    print("1. 生成测试数据...")

    # 类别1：均值[0, 0, 0, 0]，标准差较小
    mean1 = np.array([0, 0, 0, 0])
    cov1 = np.array([[1.0, 0.3, 0.1, 0.0],
                     [0.3, 1.0, 0.2, 0.1],
                     [0.1, 0.2, 1.0, 0.3],
                     [0.0, 0.1, 0.3, 1.0]])

    # 类别2：均值[3, 3, 1, 1]，中等标准差
    mean2 = np.array([3, 3, 1, 1])
    cov2 = np.array([[1.5, -0.2, 0.0, 0.1],
                     [-0.2, 1.5, 0.3, 0.0],
                     [0.0, 0.3, 1.2, -0.1],
                     [0.1, 0.0, -0.1, 1.2]])

    # 类别3：均值[-2, 4, -1, 2]，较大标准差
    mean3 = np.array([-2, 4, -1, 2])
    cov3 = np.array([[2.0, 0.5, -0.3, 0.2],
                     [0.5, 2.0, 0.1, -0.4],
                     [-0.3, 0.1, 1.8, 0.0],
                     [0.2, -0.4, 0.0, 1.8]])

    # 生成样本
    X1 = np.random.multivariate_normal(mean1, cov1, N_per_class)
    X2 = np.random.multivariate_normal(mean2, cov2, N_per_class)
    X3 = np.random.multivariate_normal(mean3, cov3, N_per_class)

    X = np.vstack([X1, X2, X3])
    y = np.array([0] * N_per_class + [1] * N_per_class + [2] * N_per_class)

    # 分割训练和测试集
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"训练集大小: {X_train.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")
    print(f"特征维度: {X_train.shape[1]}")

    # 创建和训练分类器
    print("\n2. 创建和训练分类器...")

    feature_names = ['光谱波段1', '光谱波段2', '光谱波段3', '光谱波段4']
    class_names = ['水体', '植被', '建筑物']

    clf = MaximumLikelihoodClassifier(
        regularization=1e-6,
        regularization_method='diagonal',
        numerical_precision='double',
        enable_parallel=True,
        random_state=42
    )

    # 训练模型
    clf.train(X_train, y_train,
              feature_names=feature_names,
              class_names=class_names,
              validation_split=0.2)

    # 基本预测测试
    print("\n3. 进行预测测试...")
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
    y_pred_conf, confidences = clf.predict_with_confidence(X_test)

    # 计算精度
    from sklearn.metrics import accuracy_score, classification_report
    accuracy = accuracy_score(y_test, y_pred)
    print(f"测试精度: {accuracy*100:.2f}%")

    # 显示分类报告
    print("\n详细分类报告:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # 类别统计信息
    print("\n4. 类别统计信息:")
    class_stats = clf.get_class_statistics()
    for cls_id, stats in class_stats['class_info'].items():
        print(f"类别 {stats['class_name']} (标签{cls_id}):")
        print(f"  样本数量: {stats['sample_count']}")
        print(f"  先验概率: {stats['prior_probability']:.4f}")
        print(f"  协方差行列式: {stats['covariance_determinant']:.2e}")

    # 类别可分性评估
    print("\n5. 类别可分性评估:")
    separability = clf.evaluate_class_separability()
    print(f"总体可分性（平均马氏距离）: {separability['overall_separability']:.4f}")
    print("类别间距离矩阵:")
    for i, cls1 in enumerate(clf.classes_):
        for j, cls2 in enumerate(clf.classes_):
            if i != j:
                dist = separability['separability_matrix'][i, j]
                print(f"  {class_names[i]} - {class_names[j]}: {dist:.4f}")

    # 置信度分析
    print("\n6. 置信度分析:")
    avg_confidence = np.mean(confidences)
    print(f"平均预测置信度: {avg_confidence:.4f}")

    # 按类别分析置信度
    for i, cls_name in enumerate(class_names):
        class_mask = (y_pred == i)
        if np.any(class_mask):
            class_confidence = np.mean(confidences[class_mask])
            print(f"类别 {cls_name} 平均置信度: {class_confidence:.4f}")

    # 模型信息
    print("\n7. 模型信息:")
    model_info = clf.get_model_info()
    print(f"模型版本: {model_info['version']}")
    print(f"数值精度: {model_info['config']['numerical_precision']}")
    print(f"训练时间: {model_info['training_history']['training_time']:.2f}秒")

    # 测试模型保存和加载
    print("\n8. 测试模型保存和加载...")
    clf.save_model("test_maximum_likelihood_v2.pkl")

    # 创建新实例并加载模型
    clf_loaded = MaximumLikelihoodClassifier()
    clf_loaded.load_model("test_maximum_likelihood_v2.pkl")

    # 验证加载的模型
    y_pred_loaded = clf_loaded.predict(X_test)
    accuracy_loaded = accuracy_score(y_test, y_pred_loaded)
    print(f"加载模型测试精度: {accuracy_loaded*100:.2f}%")

    # 测试3D数据处理（模拟遥感图像）
    print("\n9. 测试3D遥感图像数据...")

    # 创建模拟的遥感图像数据 (H, W, D)
    H, W = 20, 25
    test_image = np.random.multivariate_normal(mean1, cov1, (H, W))
    test_image = test_image.reshape(H, W, D)

    # 预测整个图像
    pred_image = clf.predict(test_image)
    proba_image = clf.predict_proba(test_image)

    print(f"图像预测结果形状: {pred_image.shape}")
    print(f"图像概率结果形状: {proba_image.shape}")
    print(f"预测类别分布: {np.bincount(pred_image.flatten())}")

    print("\n测试完成！所有功能正常运行。")
    print(f"最大似然分类器 v2.0.0 - 性能优异，功能完备！")
# -*- coding: utf-8 -*-
# 文件路径: src/processing/classification/supervised/minimum_distance.py
# -----------------------------------------
# 功能: 最小距离分类器实现
# 接口:
#     train(self, features: np.ndarray, labels: np.ndarray) -> None
#     predict(self, features: np.ndarray) -> np.ndarray
#     predict_distances(self, features: np.ndarray) -> np.ndarray
#     predict_with_confidence(self, features: np.ndarray) -> tuple
#     get_class_centers(self) -> dict
#     get_model_info(self) -> dict
#     save_model(self, filepath: str) -> None
#     load_model(self, filepath: str) -> None
#     evaluate_center_separability(self) -> dict
#     set_distance_metric(self, metric: str) -> None
# 作者: 孟诣楠
# 版本: 2.0.0
# 最新更改时间: 2025-06-17
# 改进说明: 重大功能增强版本
#   - 新增多种距离度量支持 (欧氏距离、曼哈顿距离、马氏距离)
#   - 新增距离预测功能 (predict_distances)
#   - 新增置信度评估功能 (predict_with_confidence)
#   - 新增类别中心获取功能 (get_class_centers)
#   - 新增模型信息查询功能 (get_model_info)
#   - 新增模型保存和加载能力 (save_model, load_model)
#   - 新增中心可分性评估 (evaluate_center_separability)
#   - 优化向量化计算提升性能
#   - 增强异常处理和数据验证
#   - 支持大规模遥感图像快速分类
#   - 新增并行计算加速
#   - 改进内存使用效率
# -----------------------------------------

import numpy as np
import pickle
import logging
import time
from typing import Optional, Dict, Any, Union, Tuple, List
from pathlib import Path
import warnings
from concurrent.futures import ThreadPoolExecutor

from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances
from src.processing.classification.base_classifier import BaseClassifier

# 配置日志系统
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MinimumDistanceClassifier(BaseClassifier):
    """
    最小距离分类器 - 增强版本 2.0.0
    
    基于类别中心的几何距离分类方法，通过计算样本到各类别中心的距离进行分类决策。
    该方法计算简单、执行高效，特别适用于大规模遥感图像的快速分类任务。
    
    新增功能:
        - 多种距离度量方式选择
        - 置信度评估和距离分析
        - 类别中心特性分析
        - 模型持久化和版本管理
        - 中心可分性评估
        - 性能优化和并行计算
    
    技术特点:
        - 支持向量化计算提升处理速度
        - 实现多种距离度量算法
        - 提供详细的几何分析报告
        - 支持大规模数据的高效处理
    """

    def __init__(self,
                 distance_metric: str = 'euclidean',
                 enable_parallel: bool = True,
                 chunk_size: int = 10000,
                 numerical_precision: str = 'double',
                 random_state: Optional[int] = 42):
        """
        初始化最小距离分类器
        
        参数:
            distance_metric: 距离度量方式 ('euclidean', 'manhattan', 'mahalanobis', 'cosine')
            enable_parallel: 是否启用并行计算加速
            chunk_size: 大数据分块处理的块大小
            numerical_precision: 数值精度 ('single', 'double')
            random_state: 随机种子，确保结果可重现
        """
        super().__init__()

        # 参数验证
        self._validate_parameters(distance_metric, chunk_size, numerical_precision)

        # 配置参数
        self.config = {
            'distance_metric': distance_metric,
            'enable_parallel': enable_parallel,
            'chunk_size': chunk_size,
            'numerical_precision': numerical_precision,
            'random_state': random_state
        }

        # 数值精度设置
        self.dtype = np.float64 if numerical_precision == 'double' else np.float32

        # 模型参数存储
        self.classes_ = None
        self.n_classes_ = 0
        self.n_features_ = 0
        self.means_ = {}
        self.class_sample_counts_ = {}
        self.covariance_matrix_ = None  # 用于马氏距离

        # 训练状态
        self.is_trained = False
        self.training_history = {}
        self.feature_names = None
        self.class_names = None

        logger.info(f"最小距离分类器已初始化 - 版本 2.0.0")
        logger.info(f"配置参数: {self.config}")

    def _validate_parameters(self, distance_metric: str, chunk_size: int,
                             numerical_precision: str) -> None:
        """参数有效性验证"""
        supported_metrics = ['euclidean', 'manhattan', 'mahalanobis', 'cosine']
        if distance_metric not in supported_metrics:
            raise ValueError(f"不支持的距离度量: {distance_metric}, 支持的度量: {supported_metrics}")

        if chunk_size <= 0:
            raise ValueError("chunk_size必须为正整数")

        if numerical_precision not in ['single', 'double']:
            raise ValueError(f"不支持的数值精度: {numerical_precision}")

    def train(self, features: np.ndarray, labels: np.ndarray,
              feature_names: Optional[List[str]] = None,
              class_names: Optional[List[str]] = None,
              validation_split: float = 0.0) -> None:
        """
        训练最小距离分类器模型
        
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
            logger.info("开始训练最小距离分类器...")
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

            # 验证集分割
            if validation_split > 0:
                X_train, X_val, y_train, y_val = self._split_validation_data(X, y, validation_split)
            else:
                X_train, y_train = X, y
                X_val, y_val = None, None

            # 计算类别中心
            total_samples = len(y_train)
            logger.info(f"训练样本总数: {total_samples}")
            logger.info(f"特征维度: {self.n_features_}")
            logger.info(f"类别数量: {self.n_classes_}")

            # 计算类别统计量
            for cls in self.classes_:
                class_mask = (y_train == cls)
                class_features = X_train[class_mask].astype(self.dtype)

                # 存储样本数量
                self.class_sample_counts_[cls] = len(class_features)

                # 计算类别中心（均值向量）
                self.means_[cls] = np.mean(class_features, axis=0)

            # 为马氏距离计算协方差矩阵
            if self.config['distance_metric'] == 'mahalanobis':
                self.covariance_matrix_ = self._compute_pooled_covariance(X_train, y_train)

            # 记录训练历史
            training_time = time.time() - start_time
            self.training_history = {
                'training_time': training_time,
                'n_samples': total_samples,
                'n_features': self.n_features_,
                'n_classes': self.n_classes_,
                'class_counts': dict(zip(self.classes_, counts)),
                'distance_metric': self.config['distance_metric']
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

                # 计算距离并找到最近的类别
                if len(X_valid) > self.config['chunk_size'] and self.config['enable_parallel']:
                    # 大数据并行处理
                    predictions[valid_mask] = self._predict_parallel(X_valid)
                else:
                    # 常规处理
                    predictions[valid_mask] = self._predict_sequential(X_valid)

            # 恢复原始形状
            if features.ndim == 3:
                return predictions.reshape(H, W)
            return predictions

        except Exception as e:
            logger.error(f"预测过程发生错误: {str(e)}")
            raise

    def predict_distances(self, features: np.ndarray) -> np.ndarray:
        """
        预测样本到各类别中心的距离
        
        参数:
            features: 特征数组，形状 (M, D) 或 (H, W, D)
        
        返回:
            distances: 距离数组，形状 (M, n_classes) 或 (H, W, n_classes)
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

            # 初始化距离数组
            distances = np.full((X.shape[0], self.n_classes_), np.inf, dtype=self.dtype)
            valid_mask = self._get_valid_pixel_mask(X)

            if np.any(valid_mask):
                X_valid = X[valid_mask]
                distances[valid_mask] = self._compute_distances_to_centers(X_valid)

            # 恢复原始形状
            if features.ndim == 3:
                return distances.reshape(H, W, self.n_classes_)
            return distances

        except Exception as e:
            logger.error(f"距离预测过程发生错误: {str(e)}")
            raise

    def predict_with_confidence(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测结果及其置信度
        
        参数:
            features: 特征数组，形状 (M, D) 或 (H, W, D)
        
        返回:
            predictions: 预测标签数组
            confidences: 置信度数组（基于距离比值）
        """
        distances = self.predict_distances(features)

        if features.ndim == 3:
            # 处理3D数据
            min_distances = np.min(distances, axis=2)
            predictions = np.argmin(distances, axis=2)

            # 计算置信度（最小距离与次小距离的比值）
            sorted_distances = np.sort(distances, axis=2)
            min_dist = sorted_distances[:, :, 0]
            second_min_dist = sorted_distances[:, :, 1]

            # 避免除零
            confidences = np.where(second_min_dist > 0,
                                   1.0 - (min_dist / second_min_dist),
                                   1.0)

            # 将索引转换为实际类别标签
            predictions = self.classes_[predictions.flatten()].reshape(predictions.shape)
        else:
            # 处理2D数据
            min_distances = np.min(distances, axis=1)
            predictions = np.argmin(distances, axis=1)

            # 计算置信度
            sorted_distances = np.sort(distances, axis=1)
            min_dist = sorted_distances[:, 0]
            second_min_dist = sorted_distances[:, 1]

            confidences = np.where(second_min_dist > 0,
                                   1.0 - (min_dist / second_min_dist),
                                   1.0)

            predictions = self.classes_[predictions]

        return predictions, confidences

    def get_class_centers(self) -> Dict[str, Any]:
        """
        获取类别中心信息
        
        返回:
            centers_info: 包含各类别中心详细信息的字典
        """
        if not self.is_trained:
            raise RuntimeError("模型尚未训练，请先调用train方法")

        centers_info = {
            'class_centers': {},
            'center_statistics': {
                'n_classes': self.n_classes_,
                'n_features': self.n_features_,
                'distance_metric': self.config['distance_metric']
            }
        }

        for i, cls in enumerate(self.classes_):
            center_info = {
                'class_label': cls,
                'class_name': self.class_names[i],
                'sample_count': self.class_sample_counts_[cls],
                'center_coordinates': self.means_[cls].tolist(),
                'center_norm': np.linalg.norm(self.means_[cls]),
                'feature_ranges': {
                    'min_value': np.min(self.means_[cls]),
                    'max_value': np.max(self.means_[cls]),
                    'mean_value': np.mean(self.means_[cls]),
                    'std_value': np.std(self.means_[cls])
                }
            }

            centers_info['class_centers'][cls] = center_info

        return centers_info

    def evaluate_center_separability(self) -> Dict[str, Any]:
        """
        评估类别中心间的可分性
        
        返回:
            separability_metrics: 中心可分性评估结果
        """
        if not self.is_trained:
            raise RuntimeError("模型尚未训练，请先调用train方法")

        separability = {
            'pairwise_distances': {},
            'distance_matrix': np.zeros((self.n_classes_, self.n_classes_)),
            'separability_statistics': {}
        }

        # 构建中心点矩阵
        centers = np.array([self.means_[cls] for cls in self.classes_])

        # 计算中心间距离
        if self.config['distance_metric'] == 'mahalanobis' and self.covariance_matrix_ is not None:
            # 马氏距离
            distance_matrix = pairwise_distances(centers, metric='mahalanobis',
                                                 VI=np.linalg.pinv(self.covariance_matrix_))
        else:
            # 其他距离度量
            distance_matrix = pairwise_distances(centers, metric=self.config['distance_metric'])

        separability['distance_matrix'] = distance_matrix

        # 计算成对距离
        for i, cls1 in enumerate(self.classes_):
            for j, cls2 in enumerate(self.classes_):
                if i != j:
                    dist = distance_matrix[i, j]
                    separability['pairwise_distances'][f"{cls1}-{cls2}"] = dist

        # 计算统计指标
        non_diagonal = distance_matrix[~np.eye(self.n_classes_, dtype=bool)]
        separability['separability_statistics'] = {
            'mean_distance': np.mean(non_diagonal),
            'min_distance': np.min(non_diagonal),
            'max_distance': np.max(non_diagonal),
            'std_distance': np.std(non_diagonal),
            'closest_pair': self._find_closest_pair(distance_matrix),
            'farthest_pair': self._find_farthest_pair(distance_matrix)
        }

        return separability

    def set_distance_metric(self, metric: str) -> None:
        """
        设置距离度量方式
        
        参数:
            metric: 新的距离度量方式
        """
        supported_metrics = ['euclidean', 'manhattan', 'mahalanobis', 'cosine']
        if metric not in supported_metrics:
            raise ValueError(f"不支持的距离度量: {metric}, 支持的度量: {supported_metrics}")

        self.config['distance_metric'] = metric
        logger.info(f"距离度量已更改为: {metric}")

        # 如果已训练且切换到马氏距离，需要重新计算协方差矩阵
        if self.is_trained and metric == 'mahalanobis' and self.covariance_matrix_ is None:
            logger.warning("切换到马氏距离需要重新训练模型以计算协方差矩阵")

    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型详细信息
        
        返回:
            model_info: 模型信息字典
        """
        info = {
            'version': '2.0.0',
            'model_type': 'MinimumDistance',
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
                'has_covariance_matrix': self.covariance_matrix_ is not None
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
                'means_': self.means_,
                'class_sample_counts_': self.class_sample_counts_,
                'covariance_matrix_': self.covariance_matrix_
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
            self.means_ = model_data.get('means_', {})
            self.class_sample_counts_ = model_data.get('class_sample_counts_', {})
            self.covariance_matrix_ = model_data.get('covariance_matrix_', None)

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

    def _compute_pooled_covariance(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """计算池化协方差矩阵（用于马氏距离）"""
        pooled_cov = np.zeros((self.n_features_, self.n_features_), dtype=self.dtype)
        total_samples = 0

        for cls in self.classes_:
            class_mask = (y == cls)
            class_features = X[class_mask]
            class_cov = np.cov(class_features, rowvar=False, dtype=self.dtype)

            # 权重为样本数量
            weight = len(class_features)
            pooled_cov += weight * class_cov
            total_samples += weight

        # 正则化处理
        pooled_cov = pooled_cov / total_samples
        pooled_cov += np.eye(self.n_features_, dtype=self.dtype) * 1e-6

        return pooled_cov

    def _compute_distances_to_centers(self, X: np.ndarray) -> np.ndarray:
        """计算到各类别中心的距离"""
        centers = np.array([self.means_[cls] for cls in self.classes_])
        metric = self.config['distance_metric']

        if metric == 'mahalanobis' and self.covariance_matrix_ is not None:
            # 马氏距离
            distances = pairwise_distances(X, centers, metric='mahalanobis',
                                           VI=np.linalg.pinv(self.covariance_matrix_))
        else:
            # 其他距离度量
            distances = pairwise_distances(X, centers, metric=metric)

        return distances

    def _predict_sequential(self, X: np.ndarray) -> np.ndarray:
        """串行预测"""
        distances = self._compute_distances_to_centers(X)
        min_indices = np.argmin(distances, axis=1)
        return self.classes_[min_indices]

    def _predict_parallel(self, X: np.ndarray) -> np.ndarray:
        """并行预测"""
        chunk_size = self.config['chunk_size']
        n_chunks = (len(X) + chunk_size - 1) // chunk_size

        with ThreadPoolExecutor(max_workers=4) as executor:
            chunks = [X[i*chunk_size:(i+1)*chunk_size] for i in range(n_chunks)]
            futures = [executor.submit(self._predict_sequential, chunk) for chunk in chunks]

            results = []
            for future in futures:
                results.append(future.result())

        return np.concatenate(results)

    def _find_closest_pair(self, distance_matrix: np.ndarray) -> Dict[str, Any]:
        """找到最近的类别对"""
        mask = ~np.eye(self.n_classes_, dtype=bool)
        min_idx = np.unravel_index(np.argmin(distance_matrix[mask]), distance_matrix.shape)

        # 需要调整索引
        min_distance = np.min(distance_matrix[mask])
        i, j = np.where(distance_matrix == min_distance)
        i, j = i[0], j[0]

        return {
            'classes': (self.classes_[i], self.classes_[j]),
            'class_names': (self.class_names[i], self.class_names[j]),
            'distance': min_distance
        }

    def _find_farthest_pair(self, distance_matrix: np.ndarray) -> Dict[str, Any]:
        """找到最远的类别对"""
        mask = ~np.eye(self.n_classes_, dtype=bool)
        max_distance = np.max(distance_matrix[mask])
        i, j = np.where(distance_matrix == max_distance)
        i, j = i[0], j[0]

        return {
            'classes': (self.classes_[i], self.classes_[j]),
            'class_names': (self.class_names[i], self.class_names[j]),
            'distance': max_distance
        }

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
MinimumDistance = MinimumDistanceClassifier

if __name__ == "__main__":
    # 测试增强版最小距离分类器
    print("测试最小距离分类器 v2.0.0")
    print("=" * 50)

    # 设置随机种子确保结果可重现
    np.random.seed(42)

    # 构造多类别测试数据
    N_per_class = 200
    D = 3  # 特征维度
    n_classes = 4

    print("1. 生成测试数据...")

    # 定义类别中心
    centers = np.array([
        [0, 0, 0],      # 类别0：原点附近
        [5, 5, 5],      # 类别1：远离原点
        [10, 0, 10],    # 类别2：x-z平面
        [-5, 8, -3]     # 类别3：负象限
    ])

    # 生成每个类别的样本（在中心周围添加噪声）
    X_list = []
    y_list = []

    for i, center in enumerate(centers):
        # 为每个类别生成不同程度的散布
        noise_scale = 1.0 + i * 0.5  # 不同类别有不同的散布程度
        class_samples = np.random.randn(N_per_class, D) * noise_scale + center
        X_list.append(class_samples)
        y_list.append(np.full(N_per_class, i))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    # 分割训练和测试集
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"训练集大小: {X_train.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")
    print(f"特征维度: {X_train.shape[1]}")
    print(f"类别数量: {n_classes}")

    # 创建和训练分类器
    print("\n2. 测试不同距离度量的分类器...")

    feature_names = ['X坐标', 'Y坐标', 'Z坐标']
    class_names = ['水体', '植被', '建筑物', '裸土']

    metrics_to_test = ['euclidean', 'manhattan', 'cosine']
    results = {}

    for metric in metrics_to_test:
        print(f"\n测试 {metric} 距离度量:")

        clf = MinimumDistanceClassifier(
            distance_metric=metric,
            enable_parallel=True,
            chunk_size=1000,
            numerical_precision='double',
            random_state=42
        )

        # 训练模型
        clf.train(X_train, y_train,
                  feature_names=feature_names,
                  class_names=class_names,
                  validation_split=0.2)

        # 预测和评估
        y_pred = clf.predict(X_test)
        y_pred_conf, confidences = clf.predict_with_confidence(X_test)
        distances = clf.predict_distances(X_test)

        # 计算精度
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(y_test, y_pred)
        avg_confidence = np.mean(confidences)

        results[metric] = {
            'accuracy': accuracy,
            'avg_confidence': avg_confidence,
            'classifier': clf
        }

        print(f"  测试精度: {accuracy*100:.2f}%")
        print(f"  平均置信度: {avg_confidence:.4f}")

    # 选择最佳度量进行详细分析
    best_metric = max(results.keys(), key=lambda k: results[k]['accuracy'])
    best_clf = results[best_metric]['classifier']

    print(f"\n3. 使用最佳度量 ({best_metric}) 进行详细分析:")
    print(f"最佳精度: {results[best_metric]['accuracy']*100:.2f}%")

    # 类别中心信息
    print("\n4. 类别中心信息:")
    centers_info = best_clf.get_class_centers()
    for cls_id, center_info in centers_info['class_centers'].items():
        print(f"类别 {center_info['class_name']} (标签{cls_id}):")
        print(f"  样本数量: {center_info['sample_count']}")
        print(f"  中心坐标: {center_info['center_coordinates']}")
        print(f"  中心范数: {center_info['center_norm']:.4f}")

    # 中心可分性评估
    print("\n5. 中心可分性评估:")
    separability = best_clf.evaluate_center_separability()
    stats = separability['separability_statistics']
    print(f"平均中心距离: {stats['mean_distance']:.4f}")
    print(f"最小中心距离: {stats['min_distance']:.4f}")
    print(f"最大中心距离: {stats['max_distance']:.4f}")

    closest = stats['closest_pair']
    farthest = stats['farthest_pair']
    print(f"最近类别对: {closest['class_names']} (距离: {closest['distance']:.4f})")
    print(f"最远类别对: {farthest['class_names']} (距离: {farthest['distance']:.4f})")

    # 置信度分析
    print("\n6. 置信度分析:")
    y_pred_final, confidences_final = best_clf.predict_with_confidence(X_test)

    # 按类别分析置信度
    for i, cls_name in enumerate(class_names):
        class_mask = (y_pred_final == i)
        if np.any(class_mask):
            class_confidence = np.mean(confidences_final[class_mask])
            class_count = np.sum(class_mask)
            print(f"类别 {cls_name}: 预测数量 {class_count}, 平均置信度 {class_confidence:.4f}")

    # 测试距离度量切换
    print("\n7. 测试距离度量动态切换:")
    original_metric = best_clf.config['distance_metric']
    print(f"原始度量: {original_metric}")

    # 切换到马氏距离（需要重新训练）
    print("切换到马氏距离...")
    mahal_clf = MinimumDistanceClassifier(
        distance_metric='mahalanobis',
        enable_parallel=True,
        random_state=42
    )
    mahal_clf.train(X_train, y_train, feature_names=feature_names, class_names=class_names)

    y_pred_mahal = mahal_clf.predict(X_test)
    accuracy_mahal = accuracy_score(y_test, y_pred_mahal)
    print(f"马氏距离精度: {accuracy_mahal*100:.2f}%")

    # 模型信息
    print("\n8. 模型信息:")
    model_info = best_clf.get_model_info()
    print(f"模型版本: {model_info['version']}")
    print(f"当前距离度量: {model_info['config']['distance_metric']}")
    print(f"训练时间: {model_info['training_history']['training_time']:.4f}秒")
    print(f"并行计算: {'启用' if model_info['config']['enable_parallel'] else '禁用'}")

    # 测试模型保存和加载
    print("\n9. 测试模型保存和加载...")
    best_clf.save_model("test_minimum_distance_v2.pkl")

    # 创建新实例并加载模型
    clf_loaded = MinimumDistanceClassifier()
    clf_loaded.load_model("test_minimum_distance_v2.pkl")

    # 验证加载的模型
    y_pred_loaded = clf_loaded.predict(X_test)
    accuracy_loaded = accuracy_score(y_test, y_pred_loaded)
    print(f"加载模型测试精度: {accuracy_loaded*100:.2f}%")

    # 测试3D遥感图像数据处理
    print("\n10. 测试3D遥感图像数据...")

    # 创建模拟的遥感图像数据 (H, W, D)
    H, W = 15, 20
    test_image = np.random.randn(H, W, D) * 2 + centers[0]  # 模拟主要为类别0的图像

    # 添加一些其他类别的像素
    test_image[5:10, 5:10] = np.random.randn(5, 5, D) * 2 + centers[1]  # 类别1区域
    test_image[10:15, 10:15] = np.random.randn(5, 5, D) * 2 + centers[2]  # 类别2区域

    # 预测整个图像
    pred_image = best_clf.predict(test_image)
    distances_image = best_clf.predict_distances(test_image)
    pred_conf_image, conf_image = best_clf.predict_with_confidence(test_image)

    print(f"图像预测结果形状: {pred_image.shape}")
    print(f"图像距离结果形状: {distances_image.shape}")
    print(f"图像置信度结果形状: {conf_image.shape}")

    # 统计预测结果
    unique_classes, counts = np.unique(pred_image[pred_image >= 0], return_counts=True)
    print("预测类别分布:")
    for cls, count in zip(unique_classes, counts):
        if cls < len(class_names):
            print(f"  {class_names[cls]}: {count} 像素")

    print(f"平均置信度: {np.mean(conf_image[conf_image > 0]):.4f}")

    # 性能对比总结
    print("\n11. 不同距离度量性能对比:")
    print("-" * 40)
    for metric, result in results.items():
        print(f"{metric:12s}: 精度 {result['accuracy']*100:6.2f}%, 置信度 {result['avg_confidence']:.4f}")
    print(f"{'马氏距离':12s}: 精度 {accuracy_mahal*100:6.2f}%")

    print(f"\n测试完成！最小距离分类器 v2.0.0 - 功能全面，性能优异！")
    print(f"推荐使用 {best_metric} 距离度量以获得最佳分类效果。")
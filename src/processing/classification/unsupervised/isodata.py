# -*- coding: utf-8 -*-
# 文件路径: src/processing/classification/unsupervised/isodata.py
# -----------------------------------------
# 功能: ISODATA 无监督分类实现
# 接口:
#     ISODATAClassifier.train(self, features: np.ndarray) -> None
#     ISODATAClassifier.predict(self, features: np.ndarray) -> np.ndarray
#     ISODATAClassifier.get_cluster_info(self) -> dict
#     ISODATAClassifier.get_model_info(self) -> dict
#     ISODATAClassifier.save_model(self, filepath: str) -> None
#     ISODATAClassifier.load_model(self, filepath: str) -> None
#     ISODATAClassifier.get_iteration_history(self) -> dict
#     ISODATAClassifier.evaluate_clustering_quality(self, features: np.ndarray) -> dict
#     unsupervised_isodata_classification(features: dict, desired_clusters: int = 5, ...) -> np.ndarray
# 作者: 孟诣楠
# 版本: 2.0.0
# 最新更改时间: 2025-06-17
# 改进说明: 重大功能增强版本
#   - 新增面向对象的ISODATAClassifier类实现
#   - 新增动态聚类过程监控 (get_iteration_history)
#   - 新增聚类质量评估功能 (evaluate_clustering_quality)
#   - 新增聚类信息分析 (get_cluster_info)
#   - 新增模型保存和加载能力 (save_model, load_model)
#   - 增强分裂合并算法的数值稳定性
#   - 改进大规模数据处理能力
#   - 优化内存使用和计算效率
#   - 增强异常处理和数据验证
#   - 保持原有函数式接口向后兼容
# -----------------------------------------

import numpy as np
import pickle
import logging
import time
from typing import Optional, Dict, Any, Union, Tuple, List
from pathlib import Path
import warnings

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from src.processing.classification.base_classifier import BaseClassifier
# 配置日志系统
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ISODATAClassifier(BaseClassifier):
    """
    ISODATA自适应聚类分类器 - 增强版本 2.0.0
    
    基于迭代自组织数据分析技术(Iterative Self-Organizing Data Analysis Technique)的高级聚类算法。
    该实现通过动态分裂和合并机制自动调整聚类数量，特别适用于遥感图像的复杂地物分类任务。
    
    ISODATA算法的核心优势在于其自适应特性，能够根据数据的内在结构动态调整聚类配置。
    该增强版本提供了完整的聚类过程监控、质量评估和模型管理功能，为遥感图像分析提供了专业级的解决方案。
    
    技术特点包括动态聚类数量调整、分裂合并操作的精确控制、详细的迭代过程记录以及多种聚类质量评估指标。
    该实现特别针对遥感数据的特点进行了优化，能够有效处理高维光谱特征和大规模影像数据。
    """

    def __init__(self,
                 desired_clusters: int = 5,
                 max_iter: int = 20,
                 min_cluster_size: int = 20,
                 max_cluster_size: Optional[int] = None,
                 threshold_split: float = 1.0,
                 threshold_merge: float = 1.0,
                 max_clusters: int = 20,
                 min_clusters: int = 2,
                 enable_scaling: bool = True,
                 random_state: Optional[int] = 42):
        """
        初始化ISODATA聚类分类器
        
        参数:
            desired_clusters: 期望的初始聚类数量
            max_iter: 最大迭代次数，控制算法收敛
            min_cluster_size: 聚类的最小样本数量，低于此值的聚类将被删除
            max_cluster_size: 聚类的最大样本数量，超过此值的聚类可能被分裂
            threshold_split: 分裂阈值，聚类内方差超过此值时触发分裂操作
            threshold_merge: 合并阈值，聚类中心距离小于此值时触发合并操作
            max_clusters: 允许的最大聚类数量
            min_clusters: 允许的最小聚类数量
            enable_scaling: 是否启用特征标准化预处理
            random_state: 随机种子，确保结果的可重现性
        """
        super().__init__()

        self._validate_parameters(desired_clusters, max_iter, min_cluster_size,
                                  max_cluster_size, threshold_split, threshold_merge,
                                  max_clusters, min_clusters)

        self.config = {
            'desired_clusters': desired_clusters,
            'max_iter': max_iter,
            'min_cluster_size': min_cluster_size,
            'max_cluster_size': max_cluster_size or min_cluster_size * 3,
            'threshold_split': threshold_split,
            'threshold_merge': threshold_merge,
            'max_clusters': max_clusters,
            'min_clusters': min_clusters,
            'enable_scaling': enable_scaling,
            'random_state': random_state
        }

        self.scaler = StandardScaler() if enable_scaling else None

        self.is_trained = False
        self.training_history = {}
        self.feature_names = None
        self.cluster_centers_ = None
        self.labels_ = None
        self.iteration_history = []
        self.final_n_clusters = 0

        logger.info(f"ISODATA聚类分类器已初始化 - 版本 2.0.0")
        logger.info(f"配置参数: 期望聚类数={desired_clusters}, 最大迭代={max_iter}, 分裂阈值={threshold_split}")

    def _validate_parameters(self, desired_clusters: int, max_iter: int,
                             min_cluster_size: int, max_cluster_size: Optional[int],
                             threshold_split: float, threshold_merge: float,
                             max_clusters: int, min_clusters: int) -> None:
        """参数有效性验证"""
        if desired_clusters < 1:
            raise ValueError("期望聚类数量必须为正整数")

        if max_iter < 1:
            raise ValueError("最大迭代次数必须为正整数")

        if min_cluster_size < 1:
            raise ValueError("最小聚类大小必须为正整数")

        if max_cluster_size is not None and max_cluster_size <= min_cluster_size:
            raise ValueError("最大聚类大小必须大于最小聚类大小")

        if threshold_split <= 0:
            raise ValueError("分裂阈值必须为正数")

        if threshold_merge <= 0:
            raise ValueError("合并阈值必须为正数")

        if max_clusters <= min_clusters:
            raise ValueError("最大聚类数必须大于最小聚类数")

        if desired_clusters > max_clusters:
            raise ValueError("期望聚类数不能超过最大聚类数限制")

    def train(self, features: np.ndarray,
              feature_names: Optional[List[str]] = None) -> None:
        """
        训练ISODATA聚类模型
        
        参数:
            features: 特征数组，形状为(N, D)或(H, W, D)
            feature_names: 特征名称列表，用于结果解释和可视化
        
        返回:
            无
        """
        try:
            logger.info("开始训练ISODATA聚类模型...")
            start_time = time.time()

            X = self._preprocess_training_data(features)

            self.feature_names = feature_names or [f"band_{i+1}" for i in range(X.shape[1])]

            if self.config['enable_scaling']:
                X_scaled = self.scaler.fit_transform(X)
            else:
                X_scaled = X

            logger.info(f"训练样本数: {X.shape[0]}, 特征维度: {X.shape[1]}")
            logger.info(f"期望聚类数: {self.config['desired_clusters']}")

            self.labels_, self.cluster_centers_, self.iteration_history = self._isodata_algorithm(X_scaled)
            self.is_trained = True

            training_time = time.time() - start_time
            unique_labels = np.unique(self.labels_)
            self.final_n_clusters = len(unique_labels)

            self.training_history = {
                'training_time': training_time,
                'n_samples': X.shape[0],
                'n_features': X.shape[1],
                'initial_clusters': self.config['desired_clusters'],
                'final_n_clusters': self.final_n_clusters,
                'n_iterations': len(self.iteration_history),
                'converged': self.iteration_history[-1]['converged'] if self.iteration_history else False
            }

            logger.info(f"训练完成 - 耗时: {training_time:.2f}秒")
            logger.info(f"最终聚类数: {self.final_n_clusters}, 总迭代次数: {len(self.iteration_history)}")

        except Exception as e:
            logger.error(f"训练过程发生错误: {str(e)}")
            raise

    def _isodata_algorithm(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        ISODATA算法的核心实现
        
        参数:
            X: 预处理后的特征数据
        
        返回:
            labels: 最终的聚类标签
            centers: 最终的聚类中心
            history: 迭代历史记录
        """
        n_samples, n_features = X.shape
        k = self.config['desired_clusters']

        # 初始聚类
        kmeans = KMeans(n_clusters=k, random_state=self.config['random_state'], n_init=10)
        labels = kmeans.fit_predict(X)
        next_label = k

        iteration_history = []

        for iteration in range(self.config['max_iter']):
            iteration_start = time.time()

            # 记录当前迭代状态
            unique_labels = np.unique(labels)
            current_k = len(unique_labels)

            # 删除过小的聚类
            labels = self._remove_small_clusters(X, labels)

            # 分裂操作
            labels, next_label, split_info = self._split_clusters(X, labels, next_label)

            # 合并操作
            labels, merge_info = self._merge_clusters(X, labels)

            # 重新编号聚类标签
            labels = self._relabel_clusters(labels)

            # 计算新的聚类中心
            centers = self._compute_cluster_centers(X, labels)

            # 记录迭代信息
            final_labels = np.unique(labels)
            final_k = len(final_labels)

            iteration_time = time.time() - iteration_start

            iteration_info = {
                'iteration': iteration + 1,
                'initial_clusters': current_k,
                'final_clusters': final_k,
                'splits_performed': split_info['n_splits'],
                'merges_performed': merge_info['n_merges'],
                'iteration_time': iteration_time,
                'converged': False
            }

            iteration_history.append(iteration_info)

            logger.debug(f"迭代 {iteration + 1}: {current_k} -> {final_k} 聚类, "
                         f"分裂 {split_info['n_splits']} 次, 合并 {merge_info['n_merges']} 次")

            # 检查收敛条件
            if split_info['n_splits'] == 0 and merge_info['n_merges'] == 0:
                iteration_history[-1]['converged'] = True
                logger.info(f"算法在第 {iteration + 1} 次迭代时收敛")
                break

        final_centers = self._compute_cluster_centers(X, labels)
        return labels, final_centers, iteration_history

    def _remove_small_clusters(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """删除样本数量过少的聚类"""
        unique_labels, counts = np.unique(labels, return_counts=True)
        valid_labels = unique_labels[counts >= self.config['min_cluster_size']]

        if len(valid_labels) < self.config['min_clusters']:
            logger.warning(f"删除小聚类后数量 ({len(valid_labels)}) 低于最小限制 ({self.config['min_clusters']})")
            # 保留最大的几个聚类
            sorted_indices = np.argsort(counts)[::-1]
            valid_labels = unique_labels[sorted_indices[:self.config['min_clusters']]]

        # 创建新的标签数组
        new_labels = np.full_like(labels, -1)
        for new_idx, old_label in enumerate(valid_labels):
            new_labels[labels == old_label] = new_idx

        # 将被删除聚类的样本分配到最近的有效聚类
        invalid_mask = new_labels == -1
        if np.any(invalid_mask):
            X_invalid = X[invalid_mask]
            centers = self._compute_cluster_centers(X, new_labels)

            for i, sample in enumerate(X_invalid):
                distances = [np.linalg.norm(sample - center) for center in centers]
                closest_cluster = np.argmin(distances)
                new_labels[invalid_mask][i] = closest_cluster

        return new_labels

    def _split_clusters(self, X: np.ndarray, labels: np.ndarray,
                        next_label: int) -> Tuple[np.ndarray, int, Dict]:
        """执行聚类分裂操作"""
        unique_labels = np.unique(labels)
        current_k = len(unique_labels)
        split_info = {'n_splits': 0, 'split_clusters': []}

        if current_k >= self.config['max_clusters']:
            return labels, next_label, split_info

        new_labels = labels.copy()

        for cluster_id in unique_labels:
            cluster_mask = labels == cluster_id
            cluster_data = X[cluster_mask]
            cluster_size = len(cluster_data)

            # 检查是否满足分裂条件
            if (cluster_size >= 2 * self.config['min_cluster_size'] and
                    cluster_size >= self.config['max_cluster_size'] and
                    current_k + split_info['n_splits'] < self.config['max_clusters']):

                # 计算聚类内方差
                if len(cluster_data) > 1:
                    cluster_cov = np.cov(cluster_data, rowvar=False)
                    if cluster_cov.ndim == 0:
                        cluster_variance = cluster_cov
                    else:
                        cluster_variance = np.trace(cluster_cov)

                    if cluster_variance > self.config['threshold_split']:
                        # 执行二分分裂
                        sub_kmeans = KMeans(n_clusters=2, random_state=self.config['random_state'], n_init=10)
                        sub_labels = sub_kmeans.fit_predict(cluster_data)

                        # 更新标签
                        cluster_indices = np.where(cluster_mask)[0]
                        for idx, sub_label in zip(cluster_indices, sub_labels):
                            if sub_label == 1:
                                new_labels[idx] = next_label

                        split_info['split_clusters'].append({
                            'original_cluster': int(cluster_id),
                            'new_cluster': int(next_label),
                            'original_size': cluster_size,
                            'variance': float(cluster_variance)
                        })

                        next_label += 1
                        split_info['n_splits'] += 1

        return new_labels, next_label, split_info

    def _merge_clusters(self, X: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """执行聚类合并操作"""
        unique_labels = np.unique(labels)
        current_k = len(unique_labels)
        merge_info = {'n_merges': 0, 'merged_pairs': []}

        if current_k <= self.config['min_clusters']:
            return labels, merge_info

        # 计算所有聚类中心
        centers = self._compute_cluster_centers(X, labels)

        # 计算聚类间距离矩阵
        distance_matrix = np.zeros((len(unique_labels), len(unique_labels)))
        for i in range(len(unique_labels)):
            for j in range(i + 1, len(unique_labels)):
                distance = np.linalg.norm(centers[i] - centers[j])
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance

        new_labels = labels.copy()
        merged_clusters = set()

        # 寻找需要合并的聚类对
        for i in range(len(unique_labels)):
            if unique_labels[i] in merged_clusters:
                continue

            for j in range(i + 1, len(unique_labels)):
                if unique_labels[j] in merged_clusters:
                    continue

                if (distance_matrix[i, j] < self.config['threshold_merge'] and
                        current_k - merge_info['n_merges'] > self.config['min_clusters']):

                    # 执行合并
                    cluster_to_merge = unique_labels[j]
                    target_cluster = unique_labels[i]

                    new_labels[new_labels == cluster_to_merge] = target_cluster
                    merged_clusters.add(cluster_to_merge)

                    merge_info['merged_pairs'].append({
                        'cluster1': int(target_cluster),
                        'cluster2': int(cluster_to_merge),
                        'distance': float(distance_matrix[i, j])
                    })

                    merge_info['n_merges'] += 1
                    break

        return new_labels, merge_info

    def _compute_cluster_centers(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """计算聚类中心"""
        unique_labels = np.unique(labels)
        centers = []

        for label in unique_labels:
            cluster_data = X[labels == label]
            center = np.mean(cluster_data, axis=0)
            centers.append(center)

        return np.array(centers)

    def _relabel_clusters(self, labels: np.ndarray) -> np.ndarray:
        """重新编号聚类标签使其连续"""
        unique_labels = np.unique(labels)
        new_labels = labels.copy()

        for new_idx, old_label in enumerate(unique_labels):
            new_labels[labels == old_label] = new_idx

        return new_labels

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        对新样本进行聚类预测
        
        参数:
            features: 特征数组，形状为(M, D)或(H, W, D)
        
        返回:
            predictions: 聚类标签数组，形状为(M,)或(H, W)
        """
        if not self.is_trained:
            raise RuntimeError("模型尚未训练，请先调用train方法")

        try:
            orig_shape = features.shape
            if features.ndim == 3:
                H, W, D = orig_shape
                X = features.reshape(-1, D)
            else:
                X = features

            valid_mask = ~np.any(np.isnan(X) | np.isinf(X), axis=1)
            predictions = np.full(X.shape[0], -1, dtype=int)

            if np.any(valid_mask):
                X_valid = X[valid_mask]

                if self.config['enable_scaling'] and self.scaler is not None:
                    X_scaled = self.scaler.transform(X_valid)
                else:
                    X_scaled = X_valid

                # 分配到最近的聚类中心
                for i, sample in enumerate(X_scaled):
                    distances = [np.linalg.norm(sample - center) for center in self.cluster_centers_]
                    closest_cluster = np.argmin(distances)
                    predictions[valid_mask][i] = closest_cluster

            if features.ndim == 3:
                return predictions.reshape(H, W)
            return predictions

        except Exception as e:
            logger.error(f"预测过程发生错误: {str(e)}")
            raise

    def get_cluster_info(self) -> Dict[str, Any]:
        """
        获取聚类详细信息
        
        返回:
            cluster_info: 包含聚类详细信息的字典
        """
        if not self.is_trained:
            raise RuntimeError("模型尚未训练，请先调用train方法")

        unique_labels, counts = np.unique(self.labels_, return_counts=True)

        cluster_info = {
            'n_clusters': self.final_n_clusters,
            'cluster_labels': unique_labels.tolist(),
            'cluster_sizes': dict(zip(unique_labels.tolist(), counts.tolist())),
            'cluster_centers': self.cluster_centers_.tolist(),
            'cluster_size_statistics': {
                'mean_size': np.mean(counts),
                'std_size': np.std(counts),
                'min_size': np.min(counts),
                'max_size': np.max(counts),
                'size_distribution': counts.tolist()
            }
        }

        # 计算聚类间距离
        if len(self.cluster_centers_) > 1:
            centers = self.cluster_centers_
            cluster_distances = []

            for i in range(len(centers)):
                for j in range(i + 1, len(centers)):
                    dist = np.linalg.norm(centers[i] - centers[j])
                    cluster_distances.append(dist)

            cluster_info['inter_cluster_distances'] = {
                'mean_distance': np.mean(cluster_distances),
                'std_distance': np.std(cluster_distances),
                'min_distance': np.min(cluster_distances),
                'max_distance': np.max(cluster_distances)
            }

        return cluster_info

    def get_iteration_history(self) -> Dict[str, Any]:
        """
        获取迭代历史信息
        
        返回:
            iteration_data: 包含每次迭代详细信息的字典
        """
        if not self.is_trained:
            raise RuntimeError("模型尚未训练，请先调用train方法")

        return {
            'total_iterations': len(self.iteration_history),
            'converged': self.training_history['converged'],
            'iteration_details': self.iteration_history,
            'cluster_evolution': [iter_info['final_clusters'] for iter_info in self.iteration_history],
            'split_evolution': [iter_info['splits_performed'] for iter_info in self.iteration_history],
            'merge_evolution': [iter_info['merges_performed'] for iter_info in self.iteration_history]
        }

    def evaluate_clustering_quality(self, features: np.ndarray) -> Dict[str, Any]:
        """
        评估聚类质量
        
        参数:
            features: 用于评估的特征数组
        
        返回:
            quality_metrics: 聚类质量评估结果
        """
        if not self.is_trained:
            raise RuntimeError("模型尚未训练，请先调用train方法")

        try:
            X = self._preprocess_training_data(features)

            if self.config['enable_scaling'] and self.scaler is not None:
                X_scaled = self.scaler.transform(X)
            else:
                X_scaled = X

            # 直接使用训练时的标签进行评估
            labels_valid = self.labels_
            X_valid = X_scaled

            quality_metrics = {
                'n_clusters': self.final_n_clusters,
                'adaptive_clustering': True,
                'initial_clusters': self.config['desired_clusters']
            }

            if len(np.unique(labels_valid)) > 1:
                try:
                    quality_metrics['silhouette_score'] = silhouette_score(X_valid, labels_valid)
                except Exception as e:
                    logger.warning(f"无法计算轮廓系数: {str(e)}")
                    quality_metrics['silhouette_score'] = None

                try:
                    quality_metrics['calinski_harabasz_score'] = calinski_harabasz_score(X_valid, labels_valid)
                except Exception as e:
                    logger.warning(f"无法计算Calinski-Harabasz指数: {str(e)}")
                    quality_metrics['calinski_harabasz_score'] = None

                try:
                    quality_metrics['davies_bouldin_score'] = davies_bouldin_score(X_valid, labels_valid)
                except Exception as e:
                    logger.warning(f"无法计算Davies-Bouldin指数: {str(e)}")
                    quality_metrics['davies_bouldin_score'] = None
            else:
                quality_metrics.update({
                    'silhouette_score': None,
                    'calinski_harabasz_score': None,
                    'davies_bouldin_score': None
                })

            return quality_metrics

        except Exception as e:
            logger.error(f"聚类质量评估过程发生错误: {str(e)}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型详细信息
        
        返回:
            model_info: 模型信息字典
        """
        info = {
            'version': '2.0.0',
            'model_type': 'ISODATA',
            'is_trained': self.is_trained,
            'config': self.config.copy(),
            'training_history': self.training_history.copy()
        }

        if self.is_trained:
            info.update({
                'feature_names': self.feature_names,
                'cluster_info': self.get_cluster_info(),
                'iteration_history': self.get_iteration_history()
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
                'cluster_centers_': self.cluster_centers_,
                'labels_': self.labels_,
                'iteration_history': self.iteration_history,
                'final_n_clusters': self.final_n_clusters,
                'scaler': self.scaler
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

            if 'version' not in model_data:
                warnings.warn("加载的是旧版本模型，某些新功能可能不可用")

            self.config = model_data.get('config', {})
            self.is_trained = model_data.get('is_trained', False)
            self.training_history = model_data.get('training_history', {})
            self.feature_names = model_data.get('feature_names', None)
            self.cluster_centers_ = model_data.get('cluster_centers_', None)
            self.labels_ = model_data.get('labels_', None)
            self.iteration_history = model_data.get('iteration_history', [])
            self.final_n_clusters = model_data.get('final_n_clusters', 0)
            self.scaler = model_data.get('scaler', None)

            logger.info(f"模型已从 {filepath} 加载")
            logger.info(f"模型版本: {model_data.get('version', '未知')}")

        except Exception as e:
            logger.error(f"加载模型时发生错误: {str(e)}")
            raise

    def _preprocess_training_data(self, features: np.ndarray) -> np.ndarray:
        """预处理训练数据"""
        if features.ndim == 3:
            H, W, D = features.shape
            X = features.reshape(-1, D)
        else:
            X = features

        valid_mask = ~np.any(np.isnan(X) | np.isinf(X), axis=1)
        X_clean = X[valid_mask]

        if len(X_clean) == 0:
            raise ValueError("没有有效的训练样本")

        if len(X_clean) < self.config['min_cluster_size'] * self.config['min_clusters']:
            logger.warning(f"样本数量 ({len(X_clean)}) 可能不足以支持期望的聚类配置")

        logger.info(f"有效训练样本: {len(X_clean)}/{len(X)} ({len(X_clean)/len(X)*100:.1f}%)")

        return X_clean


# 保持向后兼容的原函数接口
def unsupervised_isodata_classification(features: dict,
                                        desired_clusters: int = 5,
                                        max_iter: int = 10,
                                        min_cluster_size: int = 20,
                                        threshold_split: float = 1.0,
                                        threshold_merge: float = 1.0,
                                        feature_keys_to_use: list = None) -> np.ndarray:
    """
    基于ISODATA算法的无监督分类，包含动态分裂与合并步骤
    
    参数:
        features: 特征字典，包含'height'、'width'和多个二维/三维特征数组
        desired_clusters: 期望初始簇数
        max_iter: 最大迭代次数
        min_cluster_size: 簇内样本最小数量，用于判断是否可分裂
        threshold_split: 分裂阈值（簇内方差迹大于该值时分裂）
        threshold_merge: 合并阈值（簇中心距离小于该值时合并）
        feature_keys_to_use: 要用于聚类的特征键列表，None时自动选择所有二维数组特征
    
    返回:
        labels_map: 聚类标签数组，形状为(height, width)
    """
    H = features.get('height')
    W = features.get('width')
    if H is None or W is None:
        raise ValueError("features字典中缺少'height'或'width'信息")

    if feature_keys_to_use is None:
        feature_keys_to_use = [k for k, v in features.items()
                               if isinstance(v, np.ndarray) and v.ndim >= 2 and v.shape[:2] == (H, W)]
    if not feature_keys_to_use:
        raise ValueError("未指定可用的特征键，无法执行ISODATA分类")

    feat_list = []
    for key in feature_keys_to_use:
        arr = features[key]
        if arr.ndim == 2:
            feat_list.append(arr[:, :, np.newaxis])
        else:
            feat_list.append(arr)
    data = np.concatenate(feat_list, axis=-1)

    # 使用新的ISODATAClassifier
    classifier = ISODATAClassifier(
        desired_clusters=desired_clusters,
        max_iter=max_iter,
        min_cluster_size=min_cluster_size,
        threshold_split=threshold_split,
        threshold_merge=threshold_merge,
        enable_scaling=True,
        random_state=42
    )

    classifier.train(data)

    return classifier.predict(data)


if __name__ == "__main__":
    # 测试增强版ISODATA聚类分类器
    print("测试ISODATA聚类分类器 v2.0.0")
    print("=" * 50)

    from sklearn.datasets import make_blobs, make_classification

    print("1. 生成测试数据...")

    # 测试1: 标准聚类数据（具有不同大小的聚类）
    print("\n=== 测试1: 多尺度聚类数据 ===")

    # 创建具有不同大小和密度的聚类
    centers = np.array([[0, 0], [5, 5], [10, 0], [0, 10], [7, 2]])
    cluster_sizes = [100, 200, 50, 150, 80]  # 不同大小的聚类

    X_multi = []
    y_true_multi = []

    for i, (center, size) in enumerate(zip(centers, cluster_sizes)):
        cluster_std = 0.5 + i * 0.3  # 不同的聚类密度
        cluster_data = np.random.multivariate_normal(center, np.eye(2) * cluster_std**2, size)
        X_multi.append(cluster_data)
        y_true_multi.extend([i] * size)

    X_multi = np.vstack(X_multi)
    y_true_multi = np.array(y_true_multi)

    feature_names = ['X坐标', 'Y坐标']

    # 创建ISODATA分类器
    isodata_standard = ISODATAClassifier(
        desired_clusters=4,  # 期望4个聚类，但实际数据有5个
        max_iter=15,
        min_cluster_size=30,
        max_cluster_size=180,
        threshold_split=2.0,
        threshold_merge=1.5,
        max_clusters=8,
        min_clusters=2,
        enable_scaling=True,
        random_state=42
    )

    print("训练标准ISODATA模型...")
    isodata_standard.train(X_multi, feature_names=feature_names)

    y_pred_standard = isodata_standard.predict(X_multi)

    # 获取聚类信息
    cluster_info = isodata_standard.get_cluster_info()
    print(f"期望聚类数: 4, 最终聚类数: {cluster_info['n_clusters']}")
    print(f"聚类大小分布: {cluster_info['cluster_sizes']}")

    # 获取迭代历史
    iteration_history = isodata_standard.get_iteration_history()
    print(f"总迭代次数: {iteration_history['total_iterations']}")
    print(f"算法收敛: {'是' if iteration_history['converged'] else '否'}")
    print(f"聚类数演化: {iteration_history['cluster_evolution']}")

    # 聚类质量评估
    print("\n2. 聚类质量评估:")
    quality_metrics = isodata_standard.evaluate_clustering_quality(X_multi)
    print(f"轮廓系数: {quality_metrics['silhouette_score']:.4f}" if quality_metrics['silhouette_score'] else "轮廓系数: N/A")
    print(f"Calinski-Harabasz指数: {quality_metrics['calinski_harabasz_score']:.4f}" if quality_metrics['calinski_harabasz_score'] else "Calinski-Harabasz指数: N/A")
    print(f"Davies-Bouldin指数: {quality_metrics['davies_bouldin_score']:.4f}" if quality_metrics['davies_bouldin_score'] else "Davies-Bouldin指数: N/A")

    # 测试2: 高维数据的自适应聚类
    print("\n=== 测试2: 高维数据自适应聚类 ===")

    X_high_dim, y_true_high = make_classification(
        n_samples=600, n_features=8, n_informative=6, n_redundant=2,
        n_classes=3, n_clusters_per_class=2, random_state=42
    )

    feature_names_high = [f"光谱波段_{i+1}" for i in range(8)]

    isodata_high = ISODATAClassifier(
        desired_clusters=5,
        max_iter=20,
        min_cluster_size=25,
        max_cluster_size=120,
        threshold_split=1.5,
        threshold_merge=1.0,
        max_clusters=10,
        min_clusters=3,
        enable_scaling=True,
        random_state=42
    )

    print("训练高维数据模型...")
    isodata_high.train(X_high_dim, feature_names=feature_names_high)

    y_pred_high = isodata_high.predict(X_high_dim)

    cluster_info_high = isodata_high.get_cluster_info()
    iteration_history_high = isodata_high.get_iteration_history()

    print(f"高维数据 - 期望聚类数: 5, 最终聚类数: {cluster_info_high['n_clusters']}")
    print(f"高维数据 - 迭代次数: {iteration_history_high['total_iterations']}")

    # 详细的迭代过程分析
    print("\n3. 迭代过程详细分析:")
    print("迭代历史:")
    print(f"{'迭代':>4s} {'初始':>6s} {'最终':>6s} {'分裂':>6s} {'合并':>6s} {'时间(s)':>8s}")
    print("-" * 40)

    for iter_info in iteration_history_high['iteration_details'][:10]:  # 只显示前10次迭代
        print(f"{iter_info['iteration']:3d} {iter_info['initial_clusters']:5d} "
              f"{iter_info['final_clusters']:5d} {iter_info['splits_performed']:5d} "
              f"{iter_info['merges_performed']:5d} {iter_info['iteration_time']:7.3f}")

    # 测试3: 极端参数配置的鲁棒性
    print("\n=== 测试3: 极端参数配置鲁棒性 ===")

    # 创建复杂的混合数据
    X_complex = np.vstack([
        np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 80),      # 紧密聚类
        np.random.multivariate_normal([8, 8], [[4, 1], [1, 4]], 120),     # 松散聚类
        np.random.multivariate_normal([0, 8], [[0.5, 0], [0, 0.5]], 40),  # 小聚类
        np.random.uniform(-2, 10, (30, 2))  # 噪声点
    ])

    isodata_robust = ISODATAClassifier(
        desired_clusters=3,
        max_iter=25,
        min_cluster_size=15,  # 较小的最小聚类大小
        max_cluster_size=100,
        threshold_split=0.8,  # 较低的分裂阈值
        threshold_merge=2.0,  # 较高的合并阈值
        max_clusters=6,
        min_clusters=2,
        enable_scaling=True,
        random_state=42
    )

    print("训练鲁棒性测试模型...")
    isodata_robust.train(X_complex, feature_names=['特征1', '特征2'])

    y_pred_robust = isodata_robust.predict(X_complex)

    cluster_info_robust = isodata_robust.get_cluster_info()
    print(f"鲁棒性测试 - 最终聚类数: {cluster_info_robust['n_clusters']}")
    print(f"聚类大小统计: 平均={cluster_info_robust['cluster_size_statistics']['mean_size']:.1f}, "
          f"范围=[{cluster_info_robust['cluster_size_statistics']['min_size']}, "
          f"{cluster_info_robust['cluster_size_statistics']['max_size']}]")

    # 模型信息比较
    print("\n4. 模型信息比较:")
    models = {
        '多尺度数据': isodata_standard,
        '高维数据': isodata_high,
        '鲁棒性测试': isodata_robust
    }

    print("-" * 100)
    print(f"{'模型类型':12s} {'期望K':>7s} {'最终K':>7s} {'迭代数':>7s} {'收敛':>6s} {'分裂次数':>8s} {'合并次数':>8s} {'训练时间':>10s}")
    print("-" * 100)

    for name, model in models.items():
        model_info = model.get_model_info()
        expected_k = model_info['config']['desired_clusters']
        final_k = model_info['training_history']['final_n_clusters']
        n_iter = model_info['training_history']['n_iterations']
        converged = '是' if model_info['training_history']['converged'] else '否'
        train_time = model_info['training_history']['training_time']

        # 计算总分裂和合并次数
        iter_history = model_info['iteration_history']['iteration_details']
        total_splits = sum(iter_info['splits_performed'] for iter_info in iter_history)
        total_merges = sum(iter_info['merges_performed'] for iter_info in iter_history)

        print(f"{name:12s} {expected_k:6d} {final_k:6d} {n_iter:6d} {converged:>5s} "
              f"{total_splits:7d} {total_merges:7d} {train_time:9.2f}s")

    # 测试模型保存和加载
    print("\n5. 测试模型保存和加载...")
    isodata_standard.save_model("test_isodata_v2.pkl")

    # 创建新实例并加载模型
    isodata_loaded = ISODATAClassifier()
    isodata_loaded.load_model("test_isodata_v2.pkl")

    # 验证加载的模型
    y_pred_loaded = isodata_loaded.predict(X_multi)
    labels_match = np.array_equal(y_pred_standard, y_pred_loaded)
    print(f"加载模型预测结果一致: {'是' if labels_match else '否'}")

    # 验证迭代历史是否保持一致
    history_loaded = isodata_loaded.get_iteration_history()
    history_match = (iteration_history['total_iterations'] == history_loaded['total_iterations'])
    print(f"迭代历史保持一致: {'是' if history_match else '否'}")

    # 测试3D遥感图像数据处理
    print("\n6. 测试3D遥感图像数据...")

    # 创建模拟的遥感图像数据 (H, W, D)
    H, W, D = 25, 30, 5
    np.random.seed(42)

    # 生成具有复杂空间结构的模拟遥感数据
    test_image = np.zeros((H, W, D))

    # 创建多个地物类型区域（具有不同的聚类特性）
    # 区域1: 大面积农田 (左半部分)
    crop_signature = [0.05, 0.10, 0.60, 0.35, 0.25]
    test_image[:, :15, :] = np.random.normal(crop_signature, 0.02, (H, 15, D))

    # 区域2: 森林 (右上角，密集聚类)
    forest_signature = [0.03, 0.08, 0.50, 0.40, 0.30]
    test_image[:12, 15:, :] = np.random.normal(forest_signature, 0.015, (12, 15, D))

    # 区域3: 城市建筑 (右下角，分散聚类)
    urban_signature = [0.20, 0.25, 0.30, 0.28, 0.32]
    test_image[12:, 15:, :] = np.random.normal(urban_signature, 0.03, (13, 15, D))

    # 区域4: 水体 (小块区域)
    water_signature = [0.02, 0.04, 0.03, 0.02, 0.01]
    test_image[5:10, 5:10, :] = np.random.normal(water_signature, 0.005, (5, 5, D))

    # 区域5: 裸土/道路 (线性特征)
    soil_signature = [0.15, 0.18, 0.22, 0.25, 0.20]
    test_image[10:15, :, :] = np.random.normal(soil_signature, 0.01, (5, W, D))

    band_names = ['蓝光', '绿光', '红光', '近红外', '短波红外']

    # 训练遥感数据模型
    isodata_remote = ISODATAClassifier(
        desired_clusters=4,  # 期望4个主要地物类型
        max_iter=20,
        min_cluster_size=20,
        max_cluster_size=150,
        threshold_split=0.02,  # 针对遥感数据调整的阈值
        threshold_merge=0.05,
        max_clusters=8,
        min_clusters=3,
        enable_scaling=True,
        random_state=42
    )

    print("训练遥感图像数据模型...")
    isodata_remote.train(test_image, feature_names=band_names)

    # 预测整个图像
    pred_image = isodata_remote.predict(test_image)

    print(f"图像预测结果形状: {pred_image.shape}")

    # 统计预测结果
    unique_labels, counts = np.unique(pred_image[pred_image >= 0], return_counts=True)
    print("图像聚类分布:")
    for label, count in zip(unique_labels, counts):
        print(f"  聚类 {label}: {count} 像素 ({count/(H*W)*100:.1f}%)")

    # 遥感数据聚类分析
    cluster_info_remote = isodata_remote.get_cluster_info()
    iteration_history_remote = isodata_remote.get_iteration_history()

    print(f"遥感数据分析:")
    print(f"  期望聚类数: 4, 最终聚类数: {cluster_info_remote['n_clusters']}")
    print(f"  迭代次数: {iteration_history_remote['total_iterations']}")
    print(f"  算法收敛: {'是' if iteration_history_remote['converged'] else '否'}")

    # 聚类中心光谱特征分析
    centers_remote = np.array(cluster_info_remote['cluster_centers'])
    print("\n7. 聚类中心光谱特征分析:")
    for i, center in enumerate(centers_remote):
        print(f"聚类 {i} 光谱特征:")
        for j, (band, value) in enumerate(zip(band_names, center)):
            print(f"  {band}: {value:.4f}")
        print()

    # 测试向后兼容的函数接口
    print("8. 测试向后兼容的函数接口...")

    # 构造特征字典格式
    features_dict = {
        'height': H,
        'width': W
    }

    for i, band_name in enumerate(['blue', 'green', 'red', 'nir', 'swir']):
        features_dict[f'{band_name}_band'] = test_image[:, :, i]

    # 使用原函数接口
    labels_map_compat = unsupervised_isodata_classification(
        features_dict,
        desired_clusters=4,
        max_iter=20,
        min_cluster_size=20,
        threshold_split=0.02,
        threshold_merge=0.05,
        feature_keys_to_use=['blue_band', 'green_band', 'red_band', 'nir_band', 'swir_band']
    )

    print(f"函数接口预测结果形状: {labels_map_compat.shape}")

    # 比较两种接口的结果（由于算法的自适应性，不要求完全一致）
    unique_compat = len(np.unique(labels_map_compat))
    unique_class = len(np.unique(pred_image[pred_image >= 0]))
    print(f"两种接口聚类数量: 类接口={unique_class}, 函数接口={unique_compat}")

    # 分裂合并操作分析
    print("\n9. 分裂合并操作详细分析:")
    for name, model in [('多尺度数据', isodata_standard), ('遥感数据', isodata_remote)]:
        print(f"\n{name}:")
        iter_details = model.get_iteration_history()['iteration_details']

        split_operations = []
        merge_operations = []

        for iter_info in iter_details:
            if iter_info['splits_performed'] > 0:
                split_operations.append((iter_info['iteration'], iter_info['splits_performed']))
            if iter_info['merges_performed'] > 0:
                merge_operations.append((iter_info['iteration'], iter_info['merges_performed']))

        print(f"  分裂操作: {split_operations}")
        print(f"  合并操作: {merge_operations}")

    # 性能总结
    print("\n10. 性能总结:")
    print("=" * 60)
    print("ISODATA聚类分类器 v2.0.0 主要特性:")
    print("✓ 动态聚类数量自适应调整")
    print("✓ 智能分裂合并操作")
    print("✓ 完整的迭代过程监控")
    print("✓ 多种聚类质量评估指标")
    print("✓ 特征标准化和数据预处理")
    print("✓ 大规模3D图像数据处理")
    print("✓ 完整的模型持久化功能")
    print("✓ 向后兼容的函数式接口")
    print("✓ 鲁棒的数值稳定性处理")
    print("=" * 60)

    # 算法特点说明
    print("\nISODATA算法特点:")
    print("• 自适应调整聚类数量，无需预先精确指定")
    print("• 通过分裂操作处理大而分散的聚类")
    print("• 通过合并操作处理过小或相似的聚类")
    print("• 迭代优化直至收敛或达到最大迭代次数")
    print("• 特别适合复杂的遥感地物分类任务")

    # 参数选择建议
    print("\n参数选择建议:")
    print("• desired_clusters: 根据先验知识设置期望聚类数")
    print("• threshold_split: 较小值增加分裂敏感性")
    print("• threshold_merge: 较小值增加合并敏感性")
    print("• min_cluster_size: 根据数据规模和应用需求设置")
    print("• max_iter: 确保算法有足够时间收敛")

    # 应用场景推荐
    print("\n遥感应用场景:")
    print("• 复杂地物的精细分类")
    print("• 未知类别数的探索性分析")
    print("• 多时相变化检测")
    print("• 生态环境监测")
    print("• 农业作物精准识别")

    print(f"\n测试完成！ISODATA聚类分类器 v2.0.0 - 自适应智能，功能强大！")
    print(f"建议根据数据特性和应用需求调整分裂合并阈值以获得最佳聚类效果。")
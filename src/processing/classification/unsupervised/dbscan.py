# -*- coding: utf-8 -*-
# 文件路径: src/processing/classification/unsupervised/dbscan.py
# -----------------------------------------
# 功能: DBSCAN 无监督分类实现
# 接口:
#     DBSCANClassifier.train(self, features: np.ndarray) -> None
#     DBSCANClassifier.predict(self, features: np.ndarray) -> np.ndarray
#     DBSCANClassifier.get_cluster_info(self) -> dict
#     DBSCANClassifier.get_model_info(self) -> dict
#     DBSCANClassifier.save_model(self, filepath: str) -> None
#     DBSCANClassifier.load_model(self, filepath: str) -> None
#     DBSCANClassifier.optimize_parameters(self, features: np.ndarray) -> dict
#     DBSCANClassifier.evaluate_clustering_quality(self, features: np.ndarray) -> dict
#     unsupervised_dbscan_classification(features: dict, eps: float = 0.5, min_samples: int = 5, feature_keys_to_use: list = None) -> np.ndarray
# 作者: 孟诣楠
# 版本: 2.0.0
# 最新更改时间: 2025-06-17
# 改进说明: 重大功能增强版本
#   - 新增面向对象的DBSCANClassifier类实现
#   - 新增聚类质量评估功能 (evaluate_clustering_quality)
#   - 新增参数自动优化功能 (optimize_parameters)
#   - 新增聚类信息分析 (get_cluster_info)
#   - 新增模型保存和加载能力 (save_model, load_model)
#   - 增强数据预处理和特征标准化
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

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
from src.processing.classification.base_classifier import BaseClassifier


# 配置日志系统
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DBSCANClassifier(BaseClassifier):
    """
    DBSCAN密度聚类分类器 - 增强版本 2.0.0
    
    基于密度的空间聚类算法，能够发现任意形状的聚类并有效处理噪声数据。
    该实现特别适用于遥感图像的无监督分类任务，能够自动确定聚类数量并识别异常像素。
    
    新增功能:
        - 聚类质量全面评估
        - 参数自动优化和网格搜索
        - 聚类结构深度分析
        - 模型持久化和版本管理
        - 特征标准化预处理
        - 性能监控和优化
    
    技术特点:
        - 支持大规模遥感数据高效处理
        - 实现多种聚类质量评估指标
        - 提供详细的聚类统计分析
        - 支持自动化参数调优
    """

    def __init__(self,
                 eps: float = 0.5,
                 min_samples: int = 5,
                 metric: str = 'euclidean',
                 algorithm: str = 'auto',
                 leaf_size: int = 30,
                 n_jobs: Optional[int] = None,
                 enable_scaling: bool = True,
                 random_state: Optional[int] = 42):
        """
        初始化DBSCAN聚类分类器
        
        参数:
            eps: 邻域半径，控制聚类的紧密程度
            min_samples: 核心点的最小邻域样本数
            metric: 距离度量方式
            algorithm: 最近邻搜索算法
            leaf_size: BallTree或KDTree的叶子大小
            n_jobs: 并行作业数量
            enable_scaling: 是否启用特征标准化
            random_state: 随机种子，确保结果可重现
        """
        super().__init__()

        # 参数验证
        self._validate_parameters(eps, min_samples, leaf_size)

        # 存储配置参数
        self.config = {
            'eps': eps,
            'min_samples': min_samples,
            'metric': metric,
            'algorithm': algorithm,
            'leaf_size': leaf_size,
            'n_jobs': n_jobs,
            'enable_scaling': enable_scaling,
            'random_state': random_state
        }

        # 初始化模型和预处理器
        self.model = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric=metric,
            algorithm=algorithm,
            leaf_size=leaf_size,
            n_jobs=n_jobs
        )

        self.scaler = StandardScaler() if enable_scaling else None

        # 训练状态跟踪
        self.is_trained = False
        self.training_history = {}
        self.feature_names = None
        self.cluster_labels_ = None
        self.cluster_centers_ = None
        self.optimization_history = {}

        logger.info(f"DBSCAN聚类分类器已初始化 - 版本 2.0.0")
        logger.info(f"配置参数: eps={eps}, min_samples={min_samples}, 特征标准化={'启用' if enable_scaling else '禁用'}")

    def _validate_parameters(self, eps: float, min_samples: int, leaf_size: int) -> None:
        """参数有效性验证"""
        if eps <= 0:
            raise ValueError("eps参数必须为正数")

        if min_samples < 1:
            raise ValueError("min_samples参数必须为正整数")

        if leaf_size < 1:
            raise ValueError("leaf_size参数必须为正整数")

    def train(self, features: np.ndarray,
              feature_names: Optional[List[str]] = None,
              enable_optimization: bool = False) -> None:
        """
        训练DBSCAN聚类模型
        
        参数:
            features: 特征数组，形状 (N, D) 或 (H, W, D)
            feature_names: 特征名称列表，用于结果解释
            enable_optimization: 是否启用自动参数优化
        
        返回:
            无
        """
        try:
            logger.info("开始训练DBSCAN聚类模型...")
            start_time = time.time()

            # 数据预处理和验证
            X = self._preprocess_training_data(features)

            # 存储元数据
            self.feature_names = feature_names or [f"band_{i+1}" for i in range(X.shape[1])]

            # 自动参数优化
            if enable_optimization:
                logger.info("执行参数优化...")
                optimization_results = self.optimize_parameters(X)
                self.optimization_history = optimization_results
                logger.info(f"优化完成，最佳参数: eps={optimization_results['best_eps']:.4f}, min_samples={optimization_results['best_min_samples']}")

            # 特征标准化
            if self.config['enable_scaling']:
                X_scaled = self.scaler.fit_transform(X)
            else:
                X_scaled = X

            # 执行聚类
            logger.info(f"训练样本数: {X.shape[0]}, 特征维度: {X.shape[1]}")

            self.cluster_labels_ = self.model.fit_predict(X_scaled)
            self.is_trained = True

            # 计算聚类中心
            self._compute_cluster_centers(X_scaled)

            # 记录训练历史
            training_time = time.time() - start_time
            unique_labels = np.unique(self.cluster_labels_)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            n_noise = np.sum(self.cluster_labels_ == -1)

            self.training_history = {
                'training_time': training_time,
                'n_samples': X.shape[0],
                'n_features': X.shape[1],
                'n_clusters': n_clusters,
                'n_noise_points': n_noise,
                'noise_ratio': n_noise / X.shape[0],
                'cluster_labels': unique_labels.tolist()
            }

            logger.info(f"训练完成 - 耗时: {training_time:.2f}秒")
            logger.info(f"发现聚类数: {n_clusters}, 噪声点数: {n_noise} ({n_noise/X.shape[0]*100:.1f}%)")

        except Exception as e:
            logger.error(f"训练过程发生错误: {str(e)}")
            raise

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        对新样本进行聚类预测
        
        参数:
            features: 特征数组，形状 (M, D) 或 (H, W, D)
        
        返回:
            predictions: 聚类标签数组，形状 (M,) 或 (H, W)
        """
        if not self.is_trained:
            raise RuntimeError("模型尚未训练，请先调用train方法")

        try:
            # 预处理输入数据
            orig_shape = features.shape
            if features.ndim == 3:
                H, W, D = orig_shape
                X = features.reshape(-1, D)
            else:
                X = features

            # 处理无效像素
            valid_mask = ~np.any(np.isnan(X) | np.isinf(X), axis=1)
            predictions = np.full(X.shape[0], -1, dtype=int)

            if np.any(valid_mask):
                X_valid = X[valid_mask]

                # 特征标准化
                if self.config['enable_scaling'] and self.scaler is not None:
                    X_scaled = self.scaler.transform(X_valid)
                else:
                    X_scaled = X_valid

                # 使用最近邻分配到现有聚类
                predictions[valid_mask] = self._assign_to_clusters(X_scaled)

            # 恢复原始形状
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

        unique_labels = np.unique(self.cluster_labels_)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

        cluster_info = {
            'n_clusters': n_clusters,
            'cluster_labels': unique_labels.tolist(),
            'cluster_sizes': {},
            'cluster_centers': {},
            'noise_points': {
                'count': np.sum(self.cluster_labels_ == -1),
                'ratio': np.mean(self.cluster_labels_ == -1)
            }
        }

        # 统计每个聚类的大小
        for label in unique_labels:
            if label != -1:
                cluster_size = np.sum(self.cluster_labels_ == label)
                cluster_info['cluster_sizes'][int(label)] = cluster_size

                if self.cluster_centers_ is not None and label in self.cluster_centers_:
                    cluster_info['cluster_centers'][int(label)] = self.cluster_centers_[label].tolist()

        return cluster_info

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
            # 预处理数据
            X = self._preprocess_training_data(features)

            if self.config['enable_scaling'] and self.scaler is not None:
                X_scaled = self.scaler.transform(X)
            else:
                X_scaled = X

            # 过滤噪声点进行评估
            non_noise_mask = self.cluster_labels_ != -1
            X_clean = X_scaled[non_noise_mask]
            labels_clean = self.cluster_labels_[non_noise_mask]

            quality_metrics = {
                'n_clusters_found': len(np.unique(labels_clean)),
                'noise_ratio': np.mean(self.cluster_labels_ == -1),
                'cluster_size_statistics': self._compute_cluster_size_stats()
            }

            # 计算聚类质量指标
            if len(np.unique(labels_clean)) > 1:
                try:
                    quality_metrics['silhouette_score'] = silhouette_score(X_clean, labels_clean)
                except Exception as e:
                    logger.warning(f"无法计算轮廓系数: {str(e)}")
                    quality_metrics['silhouette_score'] = None

                try:
                    quality_metrics['calinski_harabasz_score'] = calinski_harabasz_score(X_clean, labels_clean)
                except Exception as e:
                    logger.warning(f"无法计算Calinski-Harabasz指数: {str(e)}")
                    quality_metrics['calinski_harabasz_score'] = None

                try:
                    quality_metrics['davies_bouldin_score'] = davies_bouldin_score(X_clean, labels_clean)
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

    def optimize_parameters(self, features: np.ndarray,
                            eps_range: Tuple[float, float] = (0.1, 2.0),
                            min_samples_range: Tuple[int, int] = (3, 20),
                            n_trials: int = 20) -> Dict[str, Any]:
        """
        自动参数优化
        
        参数:
            features: 训练特征
            eps_range: eps参数搜索范围
            min_samples_range: min_samples参数搜索范围
            n_trials: 尝试次数
        
        返回:
            optimization_results: 优化结果
        """
        logger.info("开始DBSCAN参数优化...")

        # 预处理数据
        X = features if features.ndim == 2 else self._preprocess_training_data(features)

        if self.config['enable_scaling']:
            scaler_temp = StandardScaler()
            X_scaled = scaler_temp.fit_transform(X)
        else:
            X_scaled = X

        best_score = -1
        best_params = {'eps': self.config['eps'], 'min_samples': self.config['min_samples']}
        results = []

        # 使用k-distance图估计合理的eps范围
        k = max(4, min(self.config['min_samples'], 10))
        nbrs = NearestNeighbors(n_neighbors=k).fit(X_scaled)
        distances, indices = nbrs.kneighbors(X_scaled)
        k_distances = np.sort(distances[:, k-1])

        # 自适应调整eps范围
        suggested_eps = np.percentile(k_distances, 95)
        eps_min = max(eps_range[0], suggested_eps * 0.1)
        eps_max = min(eps_range[1], suggested_eps * 2.0)

        # 网格搜索
        eps_values = np.linspace(eps_min, eps_max, max(10, n_trials // 2))
        min_samples_values = range(min_samples_range[0], min_samples_range[1] + 1)

        for eps in eps_values:
            for min_samples in min_samples_values:
                try:
                    # 创建临时模型
                    temp_model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=self.config['n_jobs'])
                    labels = temp_model.fit_predict(X_scaled)

                    # 评估聚类质量
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    n_noise = np.sum(labels == -1)
                    noise_ratio = n_noise / len(labels)

                    # 计算综合评分
                    if n_clusters > 1 and noise_ratio < 0.8:
                        non_noise_mask = labels != -1
                        if np.sum(non_noise_mask) > 1:
                            try:
                                silhouette = silhouette_score(X_scaled[non_noise_mask], labels[non_noise_mask])
                                # 综合评分：平衡聚类质量和噪声比例
                                score = silhouette * (1 - noise_ratio) * min(1, n_clusters / 10)
                            except:
                                score = (1 - noise_ratio) * min(1, n_clusters / 10)
                        else:
                            score = 0
                    else:
                        score = 0

                    results.append({
                        'eps': eps,
                        'min_samples': min_samples,
                        'n_clusters': n_clusters,
                        'noise_ratio': noise_ratio,
                        'score': score
                    })

                    if score > best_score:
                        best_score = score
                        best_params = {'eps': eps, 'min_samples': min_samples}

                except Exception as e:
                    logger.debug(f"参数组合 eps={eps:.3f}, min_samples={min_samples} 失败: {str(e)}")
                    continue

        # 更新模型参数
        self.config['eps'] = best_params['eps']
        self.config['min_samples'] = best_params['min_samples']
        self.model = DBSCAN(
            eps=best_params['eps'],
            min_samples=best_params['min_samples'],
            metric=self.config['metric'],
            algorithm=self.config['algorithm'],
            leaf_size=self.config['leaf_size'],
            n_jobs=self.config['n_jobs']
        )

        optimization_results = {
            'best_eps': best_params['eps'],
            'best_min_samples': best_params['min_samples'],
            'best_score': best_score,
            'suggested_eps_from_k_distance': suggested_eps,
            'optimization_results': results
        }

        logger.info(f"参数优化完成 - 最佳eps: {best_params['eps']:.4f}, 最佳min_samples: {best_params['min_samples']}")
        return optimization_results

    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型详细信息
        
        返回:
            model_info: 模型信息字典
        """
        info = {
            'version': '2.0.0',
            'model_type': 'DBSCAN',
            'is_trained': self.is_trained,
            'config': self.config.copy(),
            'training_history': self.training_history.copy(),
            'optimization_history': self.optimization_history.copy()
        }

        if self.is_trained:
            info.update({
                'feature_names': self.feature_names,
                'cluster_info': self.get_cluster_info()
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
                'optimization_history': self.optimization_history,
                'feature_names': self.feature_names,
                'cluster_labels_': self.cluster_labels_,
                'cluster_centers_': self.cluster_centers_,
                'scaler': self.scaler,
                'model': self.model
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
            self.optimization_history = model_data.get('optimization_history', {})
            self.feature_names = model_data.get('feature_names', None)
            self.cluster_labels_ = model_data.get('cluster_labels_', None)
            self.cluster_centers_ = model_data.get('cluster_centers_', None)
            self.scaler = model_data.get('scaler', None)
            self.model = model_data.get('model', None)

            logger.info(f"模型已从 {filepath} 加载")
            logger.info(f"模型版本: {model_data.get('version', '未知')}")

        except Exception as e:
            logger.error(f"加载模型时发生错误: {str(e)}")
            raise

    # 辅助方法
    def _preprocess_training_data(self, features: np.ndarray) -> np.ndarray:
        """预处理训练数据"""
        if features.ndim == 3:
            H, W, D = features.shape
            X = features.reshape(-1, D)
        else:
            X = features

        # 移除无效样本
        valid_mask = ~np.any(np.isnan(X) | np.isinf(X), axis=1)
        X_clean = X[valid_mask]

        if len(X_clean) == 0:
            raise ValueError("没有有效的训练样本")

        logger.info(f"有效训练样本: {len(X_clean)}/{len(X)} ({len(X_clean)/len(X)*100:.1f}%)")

        return X_clean

    def _compute_cluster_centers(self, X: np.ndarray) -> None:
        """计算聚类中心"""
        self.cluster_centers_ = {}
        unique_labels = np.unique(self.cluster_labels_)

        for label in unique_labels:
            if label != -1:  # 跳过噪声点
                cluster_mask = self.cluster_labels_ == label
                cluster_points = X[cluster_mask]
                self.cluster_centers_[label] = np.mean(cluster_points, axis=0)

    def _assign_to_clusters(self, X: np.ndarray) -> np.ndarray:
        """将新样本分配到最近的聚类"""
        if self.cluster_centers_ is None or len(self.cluster_centers_) == 0:
            return np.full(X.shape[0], -1)

        predictions = np.full(X.shape[0], -1)

        for i, sample in enumerate(X):
            min_distance = float('inf')
            best_label = -1

            for label, center in self.cluster_centers_.items():
                distance = np.linalg.norm(sample - center)
                if distance < min_distance:
                    min_distance = distance
                    best_label = label

            # 如果距离太远，标记为噪声
            if min_distance < self.config['eps']:
                predictions[i] = best_label

        return predictions

    def _compute_cluster_size_stats(self) -> Dict[str, Any]:
        """计算聚类大小统计"""
        unique_labels = np.unique(self.cluster_labels_)
        cluster_sizes = []

        for label in unique_labels:
            if label != -1:
                cluster_sizes.append(np.sum(self.cluster_labels_ == label))

        if cluster_sizes:
            return {
                'mean_size': np.mean(cluster_sizes),
                'std_size': np.std(cluster_sizes),
                'min_size': np.min(cluster_sizes),
                'max_size': np.max(cluster_sizes),
                'size_distribution': cluster_sizes
            }
        else:
            return {
                'mean_size': 0,
                'std_size': 0,
                'min_size': 0,
                'max_size': 0,
                'size_distribution': []
            }


# 保持向后兼容的原函数接口
def unsupervised_dbscan_classification(features: dict,
                                       eps: float = 0.5,
                                       min_samples: int = 5,
                                       feature_keys_to_use: list = None) -> np.ndarray:
    """
    对影像特征进行 DBSCAN 密度聚类，输出聚类标签图
    
    参数:
        features: 特征字典，包含 'height', 'width' 和多个二维/三维特征数组
        eps: 距离阈值参数，默认为 0.5
        min_samples: 核心点最小邻域样本数，默认 5
        feature_keys_to_use: 要用于聚类的特征键列表，None 时自动选择所有二维数组特征
    
    返回:
        labels_map: 聚类标签数组，形状 (height, width)，标签值整型，噪声点为 -1
    """
    H = features.get('height')
    W = features.get('width')
    if H is None or W is None:
        raise ValueError("features 字典中缺少 'height' 或 'width' 信息")

    # 自动选择特征键
    if feature_keys_to_use is None:
        feature_keys_to_use = [k for k, v in features.items()
                               if isinstance(v, np.ndarray) and v.ndim >= 2 and v.shape[:2] == (H, W)]

    if not feature_keys_to_use:
        raise ValueError("未指定可用的特征键，无法执行 DBSCAN 聚类")

    # 构造特征矩阵
    feature_list = []
    for key in feature_keys_to_use:
        arr = features[key]
        if arr.ndim == 2:
            feature_list.append(arr[:, :, np.newaxis])
        elif arr.ndim == 3:
            feature_list.append(arr)
    data_stack = np.concatenate(feature_list, axis=-1)

    # 使用新的DBSCANClassifier
    classifier = DBSCANClassifier(eps=eps, min_samples=min_samples, enable_scaling=True)
    classifier.train(data_stack)

    return classifier.predict(data_stack)


if __name__ == "__main__":
    # 测试增强版DBSCAN聚类分类器
    print("测试DBSCAN聚类分类器 v2.0.0")
    print("=" * 50)

    from sklearn.datasets import make_blobs, make_moons
    import matplotlib.pyplot as plt

    print("1. 生成测试数据...")

    # 测试1: 球形聚类数据
    print("\n=== 测试1: 球形聚类数据 ===")
    X_blobs, y_true_blobs = make_blobs(n_samples=300, centers=4, n_features=2,
                                       random_state=42, cluster_std=0.8)

    feature_names = ['X坐标', 'Y坐标']

    # 创建DBSCAN分类器
    dbscan_auto = DBSCANClassifier(
        eps=0.5,
        min_samples=5,
        enable_scaling=True,
        random_state=42
    )

    # 训练模型（带参数优化）
    print("训练DBSCAN模型（自动参数优化）...")
    dbscan_auto.train(X_blobs, feature_names=feature_names, enable_optimization=True)

    # 预测
    y_pred_auto = dbscan_auto.predict(X_blobs)

    # 获取聚类信息
    cluster_info = dbscan_auto.get_cluster_info()
    print(f"发现聚类数: {cluster_info['n_clusters']}")
    print(f"噪声点数: {cluster_info['noise_points']['count']} ({cluster_info['noise_points']['ratio']*100:.1f}%)")

    # 聚类质量评估
    print("\n2. 聚类质量评估:")
    quality_metrics = dbscan_auto.evaluate_clustering_quality(X_blobs)
    print(f"轮廓系数: {quality_metrics['silhouette_score']:.4f}" if quality_metrics['silhouette_score'] else "轮廓系数: N/A")
    print(f"Calinski-Harabasz指数: {quality_metrics['calinski_harabasz_score']:.4f}" if quality_metrics['calinski_harabasz_score'] else "Calinski-Harabasz指数: N/A")
    print(f"Davies-Bouldin指数: {quality_metrics['davies_bouldin_score']:.4f}" if quality_metrics['davies_bouldin_score'] else "Davies-Bouldin指数: N/A")

    # 聚类大小统计
    size_stats = quality_metrics['cluster_size_statistics']
    print(f"平均聚类大小: {size_stats['mean_size']:.1f} ± {size_stats['std_size']:.1f}")
    print(f"聚类大小范围: [{size_stats['min_size']}, {size_stats['max_size']}]")

    # 测试2: 月牙形数据（非球形聚类）
    print("\n=== 测试2: 月牙形数据 ===")
    X_moons, y_true_moons = make_moons(n_samples=200, noise=0.1, random_state=42)

    dbscan_moons = DBSCANClassifier(
        eps=0.2,
        min_samples=5,
        enable_scaling=True,
        random_state=42
    )

    print("训练月牙形数据模型...")
    dbscan_moons.train(X_moons, feature_names=feature_names)

    y_pred_moons = dbscan_moons.predict(X_moons)

    cluster_info_moons = dbscan_moons.get_cluster_info()
    print(f"月牙形数据 - 发现聚类数: {cluster_info_moons['n_clusters']}")
    print(f"月牙形数据 - 噪声点数: {cluster_info_moons['noise_points']['count']}")

    # 测试3: 高维数据和参数优化
    print("\n=== 测试3: 高维数据和参数优化 ===")
    X_high_dim, y_true_high = make_blobs(n_samples=500, centers=5, n_features=8,
                                         random_state=42, cluster_std=1.5)

    feature_names_high = [f"特征_{i+1}" for i in range(8)]

    dbscan_high = DBSCANClassifier(enable_scaling=True, random_state=42)

    print("执行高维数据参数优化...")
    optimization_results = dbscan_high.optimize_parameters(
        X_high_dim,
        eps_range=(0.5, 3.0),
        min_samples_range=(3, 15),
        n_trials=15
    )

    print(f"优化结果:")
    print(f"  最佳eps: {optimization_results['best_eps']:.4f}")
    print(f"  最佳min_samples: {optimization_results['best_min_samples']}")
    print(f"  最佳评分: {optimization_results['best_score']:.4f}")
    print(f"  建议eps (k-distance): {optimization_results['suggested_eps_from_k_distance']:.4f}")

    # 用优化参数训练
    dbscan_high.train(X_high_dim, feature_names=feature_names_high)
    y_pred_high = dbscan_high.predict(X_high_dim)

    cluster_info_high = dbscan_high.get_cluster_info()
    print(f"高维数据 - 发现聚类数: {cluster_info_high['n_clusters']}")
    print(f"高维数据 - 噪声比例: {cluster_info_high['noise_points']['ratio']*100:.1f}%")

    # 模型信息比较
    print("\n4. 模型信息比较:")
    models = {
        '球形数据': dbscan_auto,
        '月牙数据': dbscan_moons,
        '高维数据': dbscan_high
    }

    print("-" * 80)
    print(f"{'数据类型':12s} {'eps参数':>10s} {'min_samples':>12s} {'聚类数':>8s} {'噪声率':>8s} {'训练时间':>10s}")
    print("-" * 80)

    for name, model in models.items():
        model_info = model.get_model_info()
        eps = model_info['config']['eps']
        min_samples = model_info['config']['min_samples']
        n_clusters = model_info['cluster_info']['n_clusters']
        noise_ratio = model_info['cluster_info']['noise_points']['ratio']
        train_time = model_info['training_history']['training_time']

        print(f"{name:12s} {eps:9.4f} {min_samples:11d} {n_clusters:7d} {noise_ratio*100:7.1f}% {train_time:9.2f}s")

    # 测试模型保存和加载
    print("\n5. 测试模型保存和加载...")
    dbscan_auto.save_model("test_dbscan_v2.pkl")

    # 创建新实例并加载模型
    dbscan_loaded = DBSCANClassifier()
    dbscan_loaded.load_model("test_dbscan_v2.pkl")

    # 验证加载的模型
    y_pred_loaded = dbscan_loaded.predict(X_blobs)
    labels_match = np.array_equal(y_pred_auto, y_pred_loaded)
    print(f"加载模型预测结果一致: {'是' if labels_match else '否'}")

    # 验证聚类信息是否保持一致
    cluster_info_loaded = dbscan_loaded.get_cluster_info()
    clusters_match = (cluster_info['n_clusters'] == cluster_info_loaded['n_clusters'])
    print(f"聚类信息保持一致: {'是' if clusters_match else '否'}")

    # 测试3D遥感图像数据处理
    print("\n6. 测试3D遥感图像数据...")

    # 创建模拟的遥感图像数据 (H, W, D)
    H, W, D = 20, 25, 4
    np.random.seed(42)

    # 生成具有空间结构的模拟遥感数据
    test_image = np.zeros((H, W, D))

    # 创建几个区域，每个区域有不同的光谱特征
    # 区域1: 左上角 - 水体
    test_image[:8, :10, :] = np.random.normal([0.1, 0.2, 0.05, 0.15], 0.02, (8, 10, D))

    # 区域2: 右上角 - 植被
    test_image[:8, 15:, :] = np.random.normal([0.05, 0.8, 0.1, 0.9], 0.03, (8, 10, D))

    # 区域3: 下半部分 - 建筑物
    test_image[12:, :, :] = np.random.normal([0.4, 0.45, 0.5, 0.48], 0.04, (8, W, D))

    # 区域4: 中间部分 - 裸土
    test_image[8:12, 5:20, :] = np.random.normal([0.3, 0.35, 0.4, 0.32], 0.03, (4, 15, D))

    # 添加一些噪声像素
    noise_mask = np.random.random((H, W)) < 0.05
    test_image[noise_mask] = np.random.random((np.sum(noise_mask), D))

    band_names = ['蓝光', '红光', '近红外', '短波红外']

    # 训练遥感数据模型
    dbscan_remote = DBSCANClassifier(
        eps=0.1,
        min_samples=4,
        enable_scaling=True,
        random_state=42
    )

    print("训练遥感图像数据模型...")
    dbscan_remote.train(test_image, feature_names=band_names)

    # 预测整个图像
    pred_image = dbscan_remote.predict(test_image)

    print(f"图像预测结果形状: {pred_image.shape}")

    # 统计预测结果
    unique_labels, counts = np.unique(pred_image, return_counts=True)
    print("图像聚类分布:")
    for label, count in zip(unique_labels, counts):
        if label == -1:
            print(f"  噪声: {count} 像素 ({count/(H*W)*100:.1f}%)")
        else:
            print(f"  聚类 {label}: {count} 像素 ({count/(H*W)*100:.1f}%)")

    # 遥感数据聚类质量评估
    quality_remote = dbscan_remote.evaluate_clustering_quality(test_image)
    print(f"遥感数据聚类质量:")
    print(f"  发现聚类数: {quality_remote['n_clusters_found']}")
    print(f"  噪声比例: {quality_remote['noise_ratio']*100:.1f}%")
    if quality_remote['silhouette_score']:
        print(f"  轮廓系数: {quality_remote['silhouette_score']:.4f}")

    # 测试向后兼容的函数接口
    print("\n7. 测试向后兼容的函数接口...")

    # 构造特征字典格式
    features_dict = {
        'height': H,
        'width': W,
        'blue_band': test_image[:, :, 0],
        'red_band': test_image[:, :, 1],
        'nir_band': test_image[:, :, 2],
        'swir_band': test_image[:, :, 3]
    }

    # 使用原函数接口
    labels_map_compat = unsupervised_dbscan_classification(
        features_dict,
        eps=0.1,
        min_samples=4,
        feature_keys_to_use=['blue_band', 'red_band', 'nir_band', 'swir_band']
    )

    print(f"函数接口预测结果形状: {labels_map_compat.shape}")

    # 验证两种接口结果的一致性
    results_consistent = np.array_equal(pred_image, labels_map_compat)
    print(f"两种接口结果一致: {'是' if results_consistent else '否'}")

    # 性能总结
    print("\n8. 性能总结:")
    print("=" * 60)
    print("DBSCAN聚类分类器 v2.0.0 主要特性:")
    print("✓ 自动参数优化和网格搜索")
    print("✓ 多种聚类质量评估指标")
    print("✓ 特征标准化和数据预处理")
    print("✓ 聚类结构深度分析")
    print("✓ 大规模3D图像数据处理")
    print("✓ 完整的模型持久化功能")
    print("✓ 向后兼容的函数式接口")
    print("✓ 噪声点自动识别和处理")
    print("=" * 60)

    # 算法特点说明
    print("\nDBSCAN算法特点:")
    print("• 基于密度的聚类，能发现任意形状的聚类")
    print("• 自动确定聚类数量，无需预先指定")
    print("• 能够识别和处理噪声点")
    print("• 对初始化不敏感，结果稳定")
    print("• 适合处理具有空间结构的遥感数据")

    # 参数建议
    print("\n参数选择建议:")
    print("• eps: 控制聚类的紧密程度，建议使用k-distance图或自动优化")
    print("• min_samples: 控制噪声容忍度，通常设为维度数+1或更大")
    print("• 特征标准化: 对于不同尺度的特征（如光谱波段）建议启用")
    print("• 参数优化: 对于复杂数据建议启用自动参数优化功能")

    print(f"\n测试完成！DBSCAN聚类分类器 v2.0.0 - 功能全面，适用性强！")
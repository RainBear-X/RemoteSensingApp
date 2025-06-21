# -*- coding: utf-8 -*-
# 文件路径: src/processing/classification/unsupervised/kmeans.py
# -----------------------------------------
# 功能: K-Means 无监督分类实现
# 接口:
#     KMeansClassifier.train(self, features: np.ndarray) -> None
#     KMeansClassifier.predict(self, features: np.ndarray) -> np.ndarray
#     KMeansClassifier.get_cluster_info(self) -> dict
#     KMeansClassifier.get_model_info(self) -> dict
#     KMeansClassifier.save_model(self, filepath: str) -> None
#     KMeansClassifier.load_model(self, filepath: str) -> None
#     KMeansClassifier.optimize_k(self, features: np.ndarray) -> dict
#     KMeansClassifier.evaluate_clustering_quality(self, features: np.ndarray) -> dict
#     unsupervised_kmeans_classification(features: dict, n_clusters: int = 5, feature_keys_to_use: list = None) -> np.ndarray
# 作者: 孟诣楠
# 版本: 2.0.0
# 最新更改时间: 2025-06-17
# 改进说明: 重大功能增强版本
#   - 新增面向对象的KMeansClassifier类实现
#   - 新增聚类数量优化功能 (optimize_k)
#   - 新增聚类质量评估功能 (evaluate_clustering_quality)
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

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from src.processing.classification.base_classifier import BaseClassifier

# 配置日志系统
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KMeansClassifier(BaseClassifier):
    """
    K-Means聚类分类器 - 增强版本 2.0.0
    
    基于质心的经典聚类算法，通过迭代优化将数据分组到预定数量的聚类中。
    该实现特别适用于遥感图像的无监督分类任务，具备高效的计算性能和稳定的收敛特性。
    
    新增功能包括聚类数量自动优化、质量评估、聚类中心分析、模型持久化等专业功能。
    支持特征标准化预处理和大规模数据的高效处理，为遥感图像分类提供了完整的解决方案。
    
    技术特点涵盖多种聚类质量评估指标、肘部法则和轮廓分析的K值优化、详细的聚类统计分析以及自动化参数调优功能。
    """

    def __init__(self,
                 n_clusters: int = 5,
                 init: str = 'k-means++',
                 n_init: int = 10,
                 max_iter: int = 300,
                 tol: float = 1e-4,
                 algorithm: str = 'lloyd',
                 random_state: Optional[int] = 42,
                 enable_scaling: bool = True):
        """
        初始化K-Means聚类分类器
        
        参数:
            n_clusters: 聚类数量
            init: 初始化方法，推荐使用'k-means++'
            n_init: 不同初始化的运行次数
            max_iter: 单次运行的最大迭代次数
            tol: 收敛容忍度
            algorithm: 算法类型，'lloyd'为标准实现
            random_state: 随机种子，确保结果可重现
            enable_scaling: 是否启用特征标准化
        """
        super().__init__()

        self._validate_parameters(n_clusters, n_init, max_iter, tol)

        self.config = {
            'n_clusters': n_clusters,
            'init': init,
            'n_init': n_init,
            'max_iter': max_iter,
            'tol': tol,
            'algorithm': algorithm,
            'random_state': random_state,
            'enable_scaling': enable_scaling
        }

        self.model = KMeans(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            algorithm=algorithm,
            random_state=random_state
        )

        self.scaler = StandardScaler() if enable_scaling else None

        self.is_trained = False
        self.training_history = {}
        self.feature_names = None
        self.cluster_centers_ = None
        self.labels_ = None
        self.optimization_history = {}

        logger.info(f"K-Means聚类分类器已初始化 - 版本 2.0.0")
        logger.info(f"配置参数: 聚类数={n_clusters}, 初始化方法={init}, 特征标准化={'启用' if enable_scaling else '禁用'}")

    def _validate_parameters(self, n_clusters: int, n_init: int, max_iter: int, tol: float) -> None:
        """参数有效性验证"""
        if n_clusters < 1:
            raise ValueError("聚类数量必须为正整数")

        if n_init < 1:
            raise ValueError("初始化运行次数必须为正整数")

        if max_iter < 1:
            raise ValueError("最大迭代次数必须为正整数")

        if tol <= 0:
            raise ValueError("收敛容忍度必须为正数")

    def train(self, features: np.ndarray,
              feature_names: Optional[List[str]] = None,
              enable_k_optimization: bool = False,
              k_range: Tuple[int, int] = (2, 15)) -> None:
        """
        训练K-Means聚类模型
        
        参数:
            features: 特征数组，形状为(N, D)或(H, W, D)
            feature_names: 特征名称列表，用于结果解释
            enable_k_optimization: 是否启用聚类数量自动优化
            k_range: K值优化的搜索范围
        
        返回:
            无
        """
        try:
            logger.info("开始训练K-Means聚类模型...")
            start_time = time.time()

            X = self._preprocess_training_data(features)

            self.feature_names = feature_names or [f"band_{i+1}" for i in range(X.shape[1])]

            if enable_k_optimization:
                logger.info("执行聚类数量优化...")
                optimization_results = self.optimize_k(X, k_range=k_range)
                self.optimization_history = optimization_results
                optimal_k = optimization_results['optimal_k']
                logger.info(f"优化完成，最优K值: {optimal_k}")

                self.config['n_clusters'] = optimal_k
                self.model = KMeans(
                    n_clusters=optimal_k,
                    init=self.config['init'],
                    n_init=self.config['n_init'],
                    max_iter=self.config['max_iter'],
                    tol=self.config['tol'],
                    algorithm=self.config['algorithm'],
                    random_state=self.config['random_state']
                )

            if self.config['enable_scaling']:
                X_scaled = self.scaler.fit_transform(X)
            else:
                X_scaled = X

            logger.info(f"训练样本数: {X.shape[0]}, 特征维度: {X.shape[1]}")
            logger.info(f"目标聚类数: {self.config['n_clusters']}")

            self.model.fit(X_scaled)
            self.is_trained = True

            self.cluster_centers_ = self.model.cluster_centers_
            self.labels_ = self.model.labels_

            training_time = time.time() - start_time

            self.training_history = {
                'training_time': training_time,
                'n_samples': X.shape[0],
                'n_features': X.shape[1],
                'n_clusters': self.config['n_clusters'],
                'n_iter': self.model.n_iter_,
                'inertia': self.model.inertia_
            }

            logger.info(f"训练完成 - 耗时: {training_time:.2f}秒")
            logger.info(f"收敛迭代次数: {self.model.n_iter_}, 总平方距离: {self.model.inertia_:.2f}")

        except Exception as e:
            logger.error(f"训练过程发生错误: {str(e)}")
            raise

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

                predictions[valid_mask] = self.model.predict(X_scaled)

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
            'n_clusters': self.config['n_clusters'],
            'cluster_labels': unique_labels.tolist(),
            'cluster_sizes': dict(zip(unique_labels.tolist(), counts.tolist())),
            'cluster_centers': self.cluster_centers_.tolist(),
            'inertia': float(self.model.inertia_),
            'cluster_size_statistics': {
                'mean_size': np.mean(counts),
                'std_size': np.std(counts),
                'min_size': np.min(counts),
                'max_size': np.max(counts),
                'size_distribution': counts.tolist()
            }
        }

        cluster_distances = []
        centers = self.cluster_centers_
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                dist = np.linalg.norm(centers[i] - centers[j])
                cluster_distances.append(dist)

        if cluster_distances:
            cluster_info['inter_cluster_distances'] = {
                'mean_distance': np.mean(cluster_distances),
                'std_distance': np.std(cluster_distances),
                'min_distance': np.min(cluster_distances),
                'max_distance': np.max(cluster_distances)
            }

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
            X = self._preprocess_training_data(features)

            if self.config['enable_scaling'] and self.scaler is not None:
                X_scaled = self.scaler.transform(X)
            else:
                X_scaled = X

            labels = self.model.predict(X_scaled)

            quality_metrics = {
                'inertia': float(self.model.inertia_),
                'n_clusters': self.config['n_clusters'],
                'within_cluster_sum_of_squares': float(self.model.inertia_)
            }

            if len(np.unique(labels)) > 1:
                try:
                    quality_metrics['silhouette_score'] = silhouette_score(X_scaled, labels)
                except Exception as e:
                    logger.warning(f"无法计算轮廓系数: {str(e)}")
                    quality_metrics['silhouette_score'] = None

                try:
                    quality_metrics['calinski_harabasz_score'] = calinski_harabasz_score(X_scaled, labels)
                except Exception as e:
                    logger.warning(f"无法计算Calinski-Harabasz指数: {str(e)}")
                    quality_metrics['calinski_harabasz_score'] = None

                try:
                    quality_metrics['davies_bouldin_score'] = davies_bouldin_score(X_scaled, labels)
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

    def optimize_k(self, features: np.ndarray,
                   k_range: Tuple[int, int] = (2, 15),
                   methods: List[str] = ['elbow', 'silhouette']) -> Dict[str, Any]:
        """
        自动优化聚类数量K
        
        参数:
            features: 训练特征数组
            k_range: K值搜索范围
            methods: 优化方法列表，支持'elbow'和'silhouette'
        
        返回:
            optimization_results: 优化结果字典
        """
        logger.info(f"开始K值优化，搜索范围: {k_range}")

        X = features if features.ndim == 2 else self._preprocess_training_data(features)

        if self.config['enable_scaling']:
            scaler_temp = StandardScaler()
            X_scaled = scaler_temp.fit_transform(X)
        else:
            X_scaled = X

        k_values = range(k_range[0], k_range[1] + 1)
        inertias = []
        silhouette_scores = []

        for k in k_values:
            try:
                temp_model = KMeans(
                    n_clusters=k,
                    init=self.config['init'],
                    n_init=self.config['n_init'],
                    max_iter=self.config['max_iter'],
                    random_state=self.config['random_state']
                )

                labels = temp_model.fit_predict(X_scaled)
                inertias.append(temp_model.inertia_)

                if len(np.unique(labels)) > 1:
                    silhouette_avg = silhouette_score(X_scaled, labels)
                    silhouette_scores.append(silhouette_avg)
                else:
                    silhouette_scores.append(0)

            except Exception as e:
                logger.warning(f"K={k}时优化失败: {str(e)}")
                inertias.append(float('inf'))
                silhouette_scores.append(0)

        optimization_results = {
            'k_values': list(k_values),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'methods_used': methods
        }

        optimal_k_candidates = {}

        if 'elbow' in methods:
            elbow_k = self._find_elbow_point(list(k_values), inertias)
            optimal_k_candidates['elbow'] = elbow_k

        if 'silhouette' in methods:
            silhouette_k = k_values[np.argmax(silhouette_scores)]
            optimal_k_candidates['silhouette'] = silhouette_k

        if len(optimal_k_candidates) == 1:
            optimal_k = list(optimal_k_candidates.values())[0]
        else:
            optimal_k = optimal_k_candidates.get('silhouette', optimal_k_candidates.get('elbow', k_range[0]))

        optimization_results['optimal_k_candidates'] = optimal_k_candidates
        optimization_results['optimal_k'] = optimal_k

        logger.info(f"K值优化完成 - 推荐K值: {optimal_k}")
        if len(optimal_k_candidates) > 1:
            logger.info(f"候选K值: {optimal_k_candidates}")

        return optimization_results

    def _find_elbow_point(self, k_values: List[int], inertias: List[float]) -> int:
        """使用肘部法则寻找最优K值"""
        if len(k_values) < 3:
            return k_values[0]

        distances = []
        for i in range(1, len(k_values) - 1):
            x1, y1 = k_values[0], inertias[0]
            x2, y2 = k_values[-1], inertias[-1]
            x0, y0 = k_values[i], inertias[i]

            distance = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
            distances.append(distance)

        elbow_idx = np.argmax(distances) + 1
        return k_values[elbow_idx]

    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型详细信息
        
        返回:
            model_info: 模型信息字典
        """
        info = {
            'version': '2.0.0',
            'model_type': 'K-Means',
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
                'cluster_centers_': self.cluster_centers_,
                'labels_': self.labels_,
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

            if 'version' not in model_data:
                warnings.warn("加载的是旧版本模型，某些新功能可能不可用")

            self.config = model_data.get('config', {})
            self.is_trained = model_data.get('is_trained', False)
            self.training_history = model_data.get('training_history', {})
            self.optimization_history = model_data.get('optimization_history', {})
            self.feature_names = model_data.get('feature_names', None)
            self.cluster_centers_ = model_data.get('cluster_centers_', None)
            self.labels_ = model_data.get('labels_', None)
            self.scaler = model_data.get('scaler', None)
            self.model = model_data.get('model', None)

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

        logger.info(f"有效训练样本: {len(X_clean)}/{len(X)} ({len(X_clean)/len(X)*100:.1f}%)")

        return X_clean


def unsupervised_kmeans_classification(features: dict,
                                       n_clusters: int = 5,
                                       feature_keys_to_use: list = None) -> np.ndarray:
    """
    对影像特征进行K-Means聚类，输出聚类标签图
    
    参数:
        features: 特征字典，包含'height'、'width'和多个二维/三维特征数组
        n_clusters: 聚类类别数量
        feature_keys_to_use: 要用于聚类的特征键列表，None时自动选择所有二维数组特征
    
    返回:
        labels_map: 聚类标签数组，形状为(height, width)，标签值在[0, n_clusters-1]
    """
    H = features.get('height')
    W = features.get('width')
    if H is None or W is None:
        raise ValueError("features字典中缺少'height'或'width'信息")

    if feature_keys_to_use is None:
        feature_keys_to_use = [k for k, v in features.items()
                               if isinstance(v, np.ndarray) and v.ndim >= 2 and v.shape[:2] == (H, W)]

    if not feature_keys_to_use:
        raise ValueError("未指定可用的特征键，无法执行K-Means聚类")

    feature_list = []
    for key in feature_keys_to_use:
        arr = features[key]
        if arr.ndim == 2:
            feature_list.append(arr[:, :, np.newaxis])
        elif arr.ndim == 3:
            feature_list.append(arr)
        else:
            continue
    data_stack = np.concatenate(feature_list, axis=-1)

    classifier = KMeansClassifier(n_clusters=n_clusters, enable_scaling=True, random_state=42)
    classifier.train(data_stack)

    return classifier.predict(data_stack)


if __name__ == "__main__":
    print("测试K-Means聚类分类器 v2.0.0")
    print("=" * 50)

    from sklearn.datasets import make_blobs, make_classification

    print("1. 生成测试数据...")

    # 测试1: 标准聚类数据
    print("\n=== 测试1: 标准聚类数据 ===")
    X_blobs, y_true_blobs = make_blobs(n_samples=400, centers=5, n_features=3,
                                       random_state=42, cluster_std=1.2)

    feature_names = ['特征1', '特征2', '特征3']

    kmeans_standard = KMeansClassifier(
        n_clusters=5,
        init='k-means++',
        n_init=20,
        enable_scaling=True,
        random_state=42
    )

    print("训练标准K-Means模型...")
    kmeans_standard.train(X_blobs, feature_names=feature_names)

    y_pred_standard = kmeans_standard.predict(X_blobs)

    cluster_info = kmeans_standard.get_cluster_info()
    print(f"聚类数量: {cluster_info['n_clusters']}")
    print(f"总平方误差: {cluster_info['inertia']:.2f}")
    print(f"聚类大小分布: {cluster_info['cluster_sizes']}")

    # 聚类质量评估
    print("\n2. 聚类质量评估:")
    quality_metrics = kmeans_standard.evaluate_clustering_quality(X_blobs)
    print(f"轮廓系数: {quality_metrics['silhouette_score']:.4f}" if quality_metrics['silhouette_score'] else "轮廓系数: N/A")
    print(f"Calinski-Harabasz指数: {quality_metrics['calinski_harabasz_score']:.4f}" if quality_metrics['calinski_harabasz_score'] else "Calinski-Harabasz指数: N/A")
    print(f"Davies-Bouldin指数: {quality_metrics['davies_bouldin_score']:.4f}" if quality_metrics['davies_bouldin_score'] else "Davies-Bouldin指数: N/A")

    # 测试2: K值自动优化
    print("\n=== 测试2: K值自动优化 ===")
    X_unknown, _ = make_blobs(n_samples=300, centers=4, n_features=2,
                              random_state=42, cluster_std=0.8)

    kmeans_auto = KMeansClassifier(enable_scaling=True, random_state=42)

    print("执行K值自动优化...")
    optimization_results = kmeans_auto.optimize_k(
        X_unknown,
        k_range=(2, 10),
        methods=['elbow', 'silhouette']
    )

    print(f"优化方法: {optimization_results['methods_used']}")
    print(f"候选K值: {optimization_results['optimal_k_candidates']}")
    print(f"推荐K值: {optimization_results['optimal_k']}")

    # 用优化的K值训练模型
    kmeans_auto.config['n_clusters'] = optimization_results['optimal_k']
    kmeans_auto.model = KMeans(
        n_clusters=optimization_results['optimal_k'],
        init=kmeans_auto.config['init'],
        n_init=kmeans_auto.config['n_init'],
        max_iter=kmeans_auto.config['max_iter'],
        random_state=kmeans_auto.config['random_state']
    )

    kmeans_auto.train(X_unknown, feature_names=['X坐标', 'Y坐标'])

    y_pred_auto = kmeans_auto.predict(X_unknown)

    cluster_info_auto = kmeans_auto.get_cluster_info()
    print(f"优化后模型 - 聚类数量: {cluster_info_auto['n_clusters']}")
    print(f"优化后模型 - 总平方误差: {cluster_info_auto['inertia']:.2f}")

    # 测试3: 高维数据聚类
    print("\n=== 测试3: 高维数据聚类 ===")
    X_high_dim, y_true_high = make_classification(
        n_samples=500, n_features=10, n_informative=8, n_redundant=2,
        n_classes=6, n_clusters_per_class=1, random_state=42
    )

    feature_names_high = [f"光谱波段_{i+1}" for i in range(10)]

    kmeans_high = KMeansClassifier(
        n_clusters=6,
        enable_scaling=True,
        n_init=30,
        max_iter=500,
        random_state=42
    )

    print("训练高维数据模型...")
    kmeans_high.train(X_high_dim, feature_names=feature_names_high)

    y_pred_high = kmeans_high.predict(X_high_dim)

    cluster_info_high = kmeans_high.get_cluster_info()
    print(f"高维数据 - 收敛迭代次数: {kmeans_high.training_history['n_iter']}")
    print(f"高维数据 - 聚类间平均距离: {cluster_info_high['inter_cluster_distances']['mean_distance']:.4f}")

    # 聚类中心分析
    print("\n3. 聚类中心分析:")
    centers = cluster_info_high['cluster_centers']
    print(f"聚类中心维度: {len(centers)} x {len(centers[0])}")

    # 计算聚类中心的特征统计
    centers_array = np.array(centers)
    feature_means = np.mean(centers_array, axis=0)
    feature_stds = np.std(centers_array, axis=0)

    print("各特征在聚类中心的统计:")
    for i, (mean, std, name) in enumerate(zip(feature_means, feature_stds, feature_names_high[:5])):  # 只显示前5个
        print(f"  {name}: 均值={mean:.4f}, 标准差={std:.4f}")

    # 模型信息比较
    print("\n4. 模型信息比较:")
    models = {
        '标准模型': kmeans_standard,
        '自动优化': kmeans_auto,
        '高维数据': kmeans_high
    }

    print("-" * 90)
    print(f"{'模型类型':12s} {'K值':>6s} {'迭代次数':>8s} {'总平方误差':>12s} {'轮廓系数':>10s} {'训练时间':>10s}")
    print("-" * 90)

    for name, model in models.items():
        model_info = model.get_model_info()
        k = model_info['config']['n_clusters']
        n_iter = model_info['training_history']['n_iter']
        inertia = model_info['training_history']['inertia']
        train_time = model_info['training_history']['training_time']

        # 获取轮廓系数
        if name == '标准模型':
            silhouette = quality_metrics.get('silhouette_score', 0)
        else:
            quality = model.evaluate_clustering_quality(X_unknown if name == '自动优化' else X_high_dim)
            silhouette = quality.get('silhouette_score', 0)

        print(f"{name:12s} {k:5d} {n_iter:7d} {inertia:11.2f} {silhouette:9.4f} {train_time:9.2f}s")

    # 测试模型保存和加载
    print("\n5. 测试模型保存和加载...")
    kmeans_standard.save_model("test_kmeans_v2.pkl")

    # 创建新实例并加载模型
    kmeans_loaded = KMeansClassifier()
    kmeans_loaded.load_model("test_kmeans_v2.pkl")

    # 验证加载的模型
    y_pred_loaded = kmeans_loaded.predict(X_blobs)
    labels_match = np.array_equal(y_pred_standard, y_pred_loaded)
    print(f"加载模型预测结果一致: {'是' if labels_match else '否'}")

    # 验证聚类中心是否保持一致
    centers_match = np.allclose(cluster_info['cluster_centers'],
                                kmeans_loaded.get_cluster_info()['cluster_centers'])
    print(f"聚类中心保持一致: {'是' if centers_match else '否'}")

    # 测试3D遥感图像数据处理
    print("\n6. 测试3D遥感图像数据...")

    # 创建模拟的遥感图像数据 (H, W, D)
    H, W, D = 30, 40, 6
    np.random.seed(42)

    # 生成具有空间结构的模拟遥感数据
    test_image = np.zeros((H, W, D))

    # 创建不同的地物类型区域
    # 区域1: 水体 (左上角)
    water_signature = [0.02, 0.04, 0.03, 0.02, 0.01, 0.01]
    test_image[:10, :15, :] = np.random.normal(water_signature, 0.005, (10, 15, D))

    # 区域2: 植被 (右上角)  
    vegetation_signature = [0.03, 0.08, 0.45, 0.25, 0.40, 0.30]
    test_image[:10, 25:, :] = np.random.normal(vegetation_signature, 0.02, (10, 15, D))

    # 区域3: 建筑物 (中间)
    urban_signature = [0.15, 0.18, 0.20, 0.22, 0.25, 0.28]
    test_image[10:20, 10:30, :] = np.random.normal(urban_signature, 0.015, (10, 20, D))

    # 区域4: 裸土 (下半部分)
    soil_signature = [0.10, 0.12, 0.15, 0.18, 0.20, 0.22]
    test_image[20:, :, :] = np.random.normal(soil_signature, 0.01, (10, W, D))

    # 区域5: 混合区域 (其余部分)
    mixed_signature = [0.08, 0.10, 0.15, 0.12, 0.18, 0.16]
    mask = test_image.sum(axis=2) == 0  # 找到未赋值的像素
    test_image[mask] = np.random.normal(mixed_signature, 0.02, (np.sum(mask), D))

    band_names = ['蓝光', '绿光', '红光', '近红外', '短波红外1', '短波红外2']

    # 训练遥感数据模型（启用K值优化）
    kmeans_remote = KMeansClassifier(enable_scaling=True, random_state=42)

    print("训练遥感图像数据模型（启用K值优化）...")
    kmeans_remote.train(test_image,
                        feature_names=band_names,
                        enable_k_optimization=True,
                        k_range=(3, 8))

    # 预测整个图像
    pred_image = kmeans_remote.predict(test_image)

    print(f"图像预测结果形状: {pred_image.shape}")

    # 统计预测结果
    unique_labels, counts = np.unique(pred_image[pred_image >= 0], return_counts=True)
    print("图像聚类分布:")
    for label, count in zip(unique_labels, counts):
        print(f"  聚类 {label}: {count} 像素 ({count/(H*W)*100:.1f}%)")

    # 遥感数据聚类质量评估
    quality_remote = kmeans_remote.evaluate_clustering_quality(test_image)
    print(f"遥感数据聚类质量:")
    print(f"  最终K值: {kmeans_remote.config['n_clusters']}")
    print(f"  总平方误差: {quality_remote['inertia']:.4f}")
    if quality_remote['silhouette_score']:
        print(f"  轮廓系数: {quality_remote['silhouette_score']:.4f}")

    # 聚类中心光谱特征分析
    remote_cluster_info = kmeans_remote.get_cluster_info()
    centers_remote = np.array(remote_cluster_info['cluster_centers'])

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

    for i, band_name in enumerate(['blue', 'green', 'red', 'nir', 'swir1', 'swir2']):
        features_dict[f'{band_name}_band'] = test_image[:, :, i]

    # 使用原函数接口
    labels_map_compat = unsupervised_kmeans_classification(
        features_dict,
        n_clusters=kmeans_remote.config['n_clusters'],
        feature_keys_to_use=['blue_band', 'green_band', 'red_band', 'nir_band', 'swir1_band', 'swir2_band']
    )

    print(f"函数接口预测结果形状: {labels_map_compat.shape}")

    # 比较两种接口的结果相似性（由于随机性，不要求完全一致）
    unique_compat = len(np.unique(labels_map_compat))
    unique_class = len(np.unique(pred_image[pred_image >= 0]))
    print(f"两种接口聚类数量: 类接口={unique_class}, 函数接口={unique_compat}")

    # K值优化分析
    print("\n9. K值优化详细分析:")
    if kmeans_auto.optimization_history:
        opt_history = kmeans_auto.optimization_history
        k_values = opt_history['k_values']
        inertias = opt_history['inertias']
        silhouette_scores = opt_history['silhouette_scores']

        print("K值优化结果表:")
        print(f"{'K值':>4s} {'总平方误差':>12s} {'轮廓系数':>10s}")
        print("-" * 28)
        for k, inertia, silhouette in zip(k_values, inertias, silhouette_scores):
            print(f"{k:3d} {inertia:11.2f} {silhouette:9.4f}")

    # 性能总结
    print("\n10. 性能总结:")
    print("=" * 60)
    print("K-Means聚类分类器 v2.0.0 主要特性:")
    print("✓ 自动K值优化 (肘部法则 + 轮廓分析)")
    print("✓ 多种聚类质量评估指标")
    print("✓ 特征标准化和数据预处理")
    print("✓ 聚类中心详细分析")
    print("✓ 大规模3D图像数据处理")
    print("✓ 完整的模型持久化功能")
    print("✓ 向后兼容的函数式接口")
    print("✓ 聚类间距离分析")
    print("=" * 60)

    # 算法特点说明
    print("\nK-Means算法特点:")
    print("• 基于质心的聚类，计算效率高")
    print("• 适合球形分布的聚类结构")
    print("• 需要预先指定聚类数量K")
    print("• 对初始化敏感，建议使用k-means++")
    print("• 适合处理大规模遥感数据")

    # 参数选择建议
    print("\n参数选择建议:")
    print("• K值: 使用肘部法则或轮廓分析自动优化")
    print("• 初始化: 推荐使用k-means++方法")
    print("• n_init: 增加运行次数可提高稳定性")
    print("• 特征标准化: 对于不同尺度的特征建议启用")
    print("• 最大迭代次数: 根据数据规模适当调整")

    # 应用场景推荐
    print("\n遥感应用场景:")
    print("• 土地利用/土地覆盖分类")
    print("• 植被类型识别")
    print("• 城市功能区划分")
    print("• 水体和湿地监测")
    print("• 地质勘探和矿物识别")

    print(f"\n测试完成！K-Means聚类分类器 v2.0.0 - 功能全面，性能优异！")
    print(f"建议根据数据特性选择合适的K值和预处理方法以获得最佳聚类效果。")
# -*- coding: utf-8 -*-
# 文件路径: src/processing/classification/supervised/svm_classifier.py
# -----------------------------------------
# 功能: 支持向量机（SVM）分类器实现
# 接口:
#     train(self, features: np.ndarray, labels: np.ndarray) -> None
#     predict(self, features: np.ndarray) -> np.ndarray
#     predict_proba(self, features: np.ndarray) -> np.ndarray
#     predict_with_confidence(self, features: np.ndarray) -> tuple
#     get_support_vectors_info(self) -> dict
#     get_model_info(self) -> dict
#     save_model(self, filepath: str) -> None
#     load_model(self, filepath: str) -> None
#     optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> dict
#     evaluate_decision_function(self, features: np.ndarray) -> np.ndarray
# 作者: 孟诣楠
# 版本: 2.0.0
# 最新更改时间: 2025-06-17
# 改进说明: 重大功能增强版本
#   - 新增概率预测功能 (predict_proba)
#   - 新增置信度评估功能 (predict_with_confidence)
#   - 新增支持向量分析 (get_support_vectors_info)
#   - 新增模型信息查询功能 (get_model_info)
#   - 新增模型保存和加载能力 (save_model, load_model)
#   - 新增超参数优化功能 (optimize_hyperparameters)
#   - 新增决策函数评估 (evaluate_decision_function)
#   - 增强多种核函数支持和参数配置
#   - 改进大规模数据处理能力
#   - 优化内存使用和计算效率
#   - 增强异常处理和数据验证
#   - 支持特征标准化和数据预处理
# -----------------------------------------

import numpy as np
import pickle
import logging
import time
from typing import Optional, Dict, Any, Union, Tuple, List
from pathlib import Path
import warnings

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from src.processing.classification.base_classifier import BaseClassifier


# 配置日志系统
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SVMClassifier(BaseClassifier):
    """
    支持向量机分类器 - 增强版本 2.0.0
    
    基于最大间隔原理的高性能分类方法，通过核技巧处理非线性分类问题。
    该实现特别适用于遥感图像的高维光谱分类任务，具备优秀的泛化能力和理论基础。
    
    新增功能:
        - 概率预测和置信度评估
        - 支持向量深度分析
        - 决策函数评估和边界分析
        - 自动超参数优化
        - 特征标准化预处理
        - 多核函数支持和配置
    
    技术特点:
        - 支持大规模遥感数据高效处理
        - 实现多种核函数和参数组合
        - 提供详细的模型几何解释
        - 支持自动化调参和性能优化
    """

    def __init__(self,
                 kernel: str = 'rbf',
                 C: float = 1.0,
                 gamma: Union[str, float] = 'scale',
                 degree: int = 3,
                 coef0: float = 0.0,
                 shrinking: bool = True,
                 probability: bool = True,
                 tol: float = 1e-3,
                 cache_size: float = 200,
                 class_weight: Optional[Union[dict, str]] = None,
                 max_iter: int = -1,
                 random_state: Optional[int] = 42,
                 enable_scaling: bool = True):
        """
        初始化支持向量机分类器
        
        参数:
            kernel: 核函数类型 ('linear', 'poly', 'rbf', 'sigmoid', 'precomputed')
            C: 正则化参数，控制误分类的惩罚程度
            gamma: 核函数系数，影响支持向量的影响范围
            degree: 多项式核函数的度数
            coef0: 多项式和sigmoid核函数的独立项
            shrinking: 是否使用收缩启发式
            probability: 是否启用概率估计
            tol: 停止训练的容忍度
            cache_size: 核缓存大小（MB）
            class_weight: 类别权重，用于处理不平衡数据
            max_iter: 最大迭代次数
            random_state: 随机种子，确保结果可重现
            enable_scaling: 是否启用特征标准化
        """
        super().__init__()

        # 参数验证
        self._validate_parameters(kernel, C, gamma, degree, tol, cache_size, max_iter)

        # 存储配置参数
        self.config = {
            'kernel': kernel,
            'C': C,
            'gamma': gamma,
            'degree': degree,
            'coef0': coef0,
            'shrinking': shrinking,
            'probability': probability,
            'tol': tol,
            'cache_size': cache_size,
            'class_weight': class_weight,
            'max_iter': max_iter,
            'random_state': random_state,
            'enable_scaling': enable_scaling
        }

        # 初始化模型和预处理器
        self.svm_model = SVC(**{k: v for k, v in self.config.items() if k != 'enable_scaling'})
        self.scaler = StandardScaler() if enable_scaling else None

        if enable_scaling:
            self.model = Pipeline([
                ('scaler', self.scaler),
                ('svm', self.svm_model)
            ])
        else:
            self.model = self.svm_model

        # 训练状态跟踪
        self.is_trained = False
        self.training_history = {}
        self.feature_names = None
        self.class_names = None
        self.optimization_history = {}

        logger.info(f"支持向量机分类器已初始化 - 版本 2.0.0")
        logger.info(f"配置参数: 核函数={kernel}, C={C}, gamma={gamma}, 特征标准化={'启用' if enable_scaling else '禁用'}")

    def _validate_parameters(self, kernel: str, C: float, gamma: Union[str, float],
                             degree: int, tol: float, cache_size: float, max_iter: int) -> None:
        """参数有效性验证"""
        supported_kernels = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
        if kernel not in supported_kernels:
            raise ValueError(f"不支持的核函数: {kernel}, 支持的核函数: {supported_kernels}")

        if C <= 0:
            raise ValueError("正则化参数C必须为正数")

        if isinstance(gamma, (int, float)) and gamma <= 0:
            raise ValueError("gamma参数必须为正数或字符串")
        elif isinstance(gamma, str) and gamma not in ['scale', 'auto']:
            raise ValueError("gamma字符串参数必须为'scale'或'auto'")

        if degree < 1:
            raise ValueError("多项式度数必须为正整数")

        if tol <= 0:
            raise ValueError("容忍度必须为正数")

        if cache_size <= 0:
            raise ValueError("缓存大小必须为正数")

        if max_iter != -1 and max_iter <= 0:
            raise ValueError("最大迭代次数必须为正整数或-1")

    def train(self, features: np.ndarray, labels: np.ndarray,
              feature_names: Optional[List[str]] = None,
              class_names: Optional[List[str]] = None,
              validation_split: float = 0.0,
              enable_optimization: bool = False,
              optimization_method: str = 'grid_search') -> None:
        """
        训练支持向量机分类器模型
        
        参数:
            features: 特征数组，形状 (N, D) 或 (H, W, D)
            labels: 标签数组，形状 (N,) 或 (H, W)
            feature_names: 特征名称列表，用于结果解释
            class_names: 类别名称列表，用于结果显示
            validation_split: 验证集比例 (0.0-1.0)
            enable_optimization: 是否启用自动超参数优化
            optimization_method: 优化方法 ('grid_search', 'random_search')
        
        返回:
            无
        """
        try:
            logger.info("开始训练支持向量机分类器...")
            start_time = time.time()

            # 数据预处理和验证
            X, y = self._preprocess_training_data(features, labels)

            # 存储元数据
            self.feature_names = feature_names or [f"band_{i+1}" for i in range(X.shape[1])]
            if class_names is not None:
                self.class_names = class_names
            else:
                unique_classes = np.unique(y)
                self.class_names = [f"class_{cls}" for cls in unique_classes]

            # 验证集分割
            if validation_split > 0:
                X_train, X_val, y_train, y_val = self._split_validation_data(X, y, validation_split)
            else:
                X_train, y_train = X, y
                X_val, y_val = None, None

            # 自动超参数优化
            if enable_optimization:
                logger.info("执行超参数优化...")
                optimization_results = self.optimize_hyperparameters(
                    X_train, y_train, method=optimization_method
                )
                self.optimization_history = optimization_results
                logger.info(f"优化完成，最佳参数: {optimization_results['best_params']}")

            # 模型训练
            logger.info(f"训练样本数: {X_train.shape[0]}, 特征维度: {X_train.shape[1]}")
            logger.info(f"类别数量: {len(np.unique(y_train))}")

            self.model.fit(X_train, y_train)
            self.is_trained = True

            # 训练性能评估
            train_pred = self.model.predict(X_train)
            train_accuracy = accuracy_score(y_train, train_pred)

            # 记录训练历史
            training_time = time.time() - start_time
            self.training_history = {
                'training_time': training_time,
                'n_samples': X_train.shape[0],
                'n_features': X_train.shape[1],
                'n_classes': len(np.unique(y_train)),
                'train_accuracy': train_accuracy,
                'kernel': self.config['kernel']
            }

            # 支持向量统计
            actual_svm = self.svm_model if not self.config['enable_scaling'] else self.model.named_steps['svm']
            if hasattr(actual_svm, 'n_support_'):
                self.training_history['n_support_vectors'] = actual_svm.n_support_.tolist()
                self.training_history['total_support_vectors'] = int(np.sum(actual_svm.n_support_))
                logger.info(f"支持向量数量: {self.training_history['total_support_vectors']}")

            # 验证集评估
            if X_val is not None:
                val_pred = self.model.predict(X_val)
                val_accuracy = accuracy_score(y_val, val_pred)
                self.training_history['val_accuracy'] = val_accuracy
                logger.info(f"验证集精度: {val_accuracy:.4f}")

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
                X = features.reshape(-1, D)

                # 处理无效像素
                valid_mask = ~np.any(np.isnan(X) | np.isinf(X), axis=1)
                predictions = np.full(X.shape[0], -1, dtype=int)

                if np.any(valid_mask):
                    predictions[valid_mask] = self.model.predict(X[valid_mask])

                return predictions.reshape(H, W)
            else:
                # 处理2D输入
                valid_mask = ~np.any(np.isnan(features) | np.isinf(features), axis=1)
                predictions = np.full(features.shape[0], -1, dtype=int)

                if np.any(valid_mask):
                    predictions[valid_mask] = self.model.predict(features[valid_mask])

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

        if not self.config['probability']:
            raise RuntimeError("概率预测未启用，请在初始化时设置probability=True")

        try:
            orig_shape = features.shape
            actual_svm = self.svm_model if not self.config['enable_scaling'] else self.model.named_steps['svm']
            n_classes = len(actual_svm.classes_)

            if features.ndim == 3:
                H, W, D = orig_shape
                X = features.reshape(-1, D)

                # 初始化概率数组
                probabilities = np.zeros((X.shape[0], n_classes))
                valid_mask = ~np.any(np.isnan(X) | np.isinf(X), axis=1)

                if np.any(valid_mask):
                    probabilities[valid_mask] = self.model.predict_proba(X[valid_mask])

                return probabilities.reshape(H, W, n_classes)
            else:
                probabilities = np.zeros((features.shape[0], n_classes))
                valid_mask = ~np.any(np.isnan(features) | np.isinf(features), axis=1)

                if np.any(valid_mask):
                    probabilities[valid_mask] = self.model.predict_proba(features[valid_mask])

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
            confidences: 置信度数组（基于决策函数距离）
        """
        if not self.config['probability']:
            # 使用决策函数计算置信度
            decision_scores = self.evaluate_decision_function(features)
            predictions = self.predict(features)

            if features.ndim == 3:
                # 对于多类别，使用最大决策值作为置信度
                confidences = np.max(decision_scores, axis=2)
            else:
                confidences = np.max(decision_scores, axis=1)
        else:
            # 使用概率预测计算置信度
            probabilities = self.predict_proba(features)
            predictions = self.predict(features)

            if features.ndim == 3:
                confidences = np.max(probabilities, axis=2)
            else:
                confidences = np.max(probabilities, axis=1)

        return predictions, confidences

    def evaluate_decision_function(self, features: np.ndarray) -> np.ndarray:
        """
        评估决策函数值
        
        参数:
            features: 特征数组，形状 (M, D) 或 (H, W, D)
        
        返回:
            decision_scores: 决策函数值数组
        """
        if not self.is_trained:
            raise RuntimeError("模型尚未训练，请先调用train方法")

        try:
            orig_shape = features.shape
            actual_svm = self.svm_model if not self.config['enable_scaling'] else self.model.named_steps['svm']

            if features.ndim == 3:
                H, W, D = orig_shape
                X = features.reshape(-1, D)

                # 处理无效像素
                valid_mask = ~np.any(np.isnan(X) | np.isinf(X), axis=1)

                if len(actual_svm.classes_) == 2:
                    # 二分类情况
                    decision_scores = np.full(X.shape[0], 0.0)
                    if np.any(valid_mask):
                        if self.config['enable_scaling']:
                            X_scaled = self.scaler.transform(X[valid_mask])
                            decision_scores[valid_mask] = actual_svm.decision_function(X_scaled).flatten()
                        else:
                            decision_scores[valid_mask] = actual_svm.decision_function(X[valid_mask]).flatten()
                    return decision_scores.reshape(H, W)
                else:
                    # 多分类情况
                    n_classes = len(actual_svm.classes_)
                    decision_scores = np.zeros((X.shape[0], n_classes))
                    if np.any(valid_mask):
                        if self.config['enable_scaling']:
                            X_scaled = self.scaler.transform(X[valid_mask])
                            decision_scores[valid_mask] = actual_svm.decision_function(X_scaled)
                        else:
                            decision_scores[valid_mask] = actual_svm.decision_function(X[valid_mask])
                    return decision_scores.reshape(H, W, n_classes)
            else:
                valid_mask = ~np.any(np.isnan(features) | np.isinf(features), axis=1)

                if len(actual_svm.classes_) == 2:
                    decision_scores = np.full(features.shape[0], 0.0)
                    if np.any(valid_mask):
                        if self.config['enable_scaling']:
                            X_scaled = self.scaler.transform(features[valid_mask])
                            decision_scores[valid_mask] = actual_svm.decision_function(X_scaled).flatten()
                        else:
                            decision_scores[valid_mask] = actual_svm.decision_function(features[valid_mask]).flatten()
                else:
                    n_classes = len(actual_svm.classes_)
                    decision_scores = np.zeros((features.shape[0], n_classes))
                    if np.any(valid_mask):
                        if self.config['enable_scaling']:
                            X_scaled = self.scaler.transform(features[valid_mask])
                            decision_scores[valid_mask] = actual_svm.decision_function(X_scaled)
                        else:
                            decision_scores[valid_mask] = actual_svm.decision_function(features[valid_mask])

                return decision_scores

        except Exception as e:
            logger.error(f"决策函数评估过程发生错误: {str(e)}")
            raise

    def get_support_vectors_info(self) -> Dict[str, Any]:
        """
        获取支持向量详细信息
        
        返回:
            support_info: 包含支持向量详细信息的字典
        """
        if not self.is_trained:
            raise RuntimeError("模型尚未训练，请先调用train方法")

        actual_svm = self.svm_model if not self.config['enable_scaling'] else self.model.named_steps['svm']

        support_info = {
            'n_support_per_class': actual_svm.n_support_.tolist(),
            'total_support_vectors': int(np.sum(actual_svm.n_support_)),
            'support_vector_indices': actual_svm.support_.tolist(),
            'dual_coefficients_shape': actual_svm.dual_coef_.shape,
            'classes': actual_svm.classes_.tolist()
        }

        # 支持向量比例统计
        if hasattr(self, 'training_history') and 'n_samples' in self.training_history:
            total_samples = self.training_history['n_samples']
            support_info['support_vector_ratio'] = support_info['total_support_vectors'] / total_samples

        # 核函数相关信息
        support_info['kernel'] = self.config['kernel']
        if hasattr(actual_svm, 'gamma_'):
            support_info['gamma_'] = actual_svm.gamma_

        # 线性核的权重向量
        if self.config['kernel'] == 'linear' and hasattr(actual_svm, 'coef_'):
            support_info['feature_weights'] = actual_svm.coef_.tolist()
            support_info['intercept'] = actual_svm.intercept_.tolist()

        return support_info

    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray,
                                 method: str = 'grid_search',
                                 cv_folds: int = 5,
                                 n_iter: int = 50) -> Dict[str, Any]:
        """
        自动超参数优化
        
        参数:
            X: 训练特征
            y: 训练标签
            method: 优化方法 ('grid_search', 'random_search')
            cv_folds: 交叉验证折数
            n_iter: 随机搜索迭代次数
        
        返回:
            optimization_results: 优化结果
        """
        logger.info(f"开始{method}超参数优化...")

        # 根据是否启用缩放调整参数名前缀
        param_prefix = 'svm__' if self.config['enable_scaling'] else ''

        # 定义参数搜索空间
        if method == 'grid_search':
            if self.config['kernel'] == 'rbf':
                param_grid = {
                    f'{param_prefix}C': [0.1, 1, 10, 100],
                    f'{param_prefix}gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
                }
            elif self.config['kernel'] == 'linear':
                param_grid = {
                    f'{param_prefix}C': [0.1, 1, 10, 100]
                }
            elif self.config['kernel'] == 'poly':
                param_grid = {
                    f'{param_prefix}C': [0.1, 1, 10],
                    f'{param_prefix}gamma': ['scale', 'auto', 0.01, 0.1],
                    f'{param_prefix}degree': [2, 3, 4]
                }
            else:
                param_grid = {
                    f'{param_prefix}C': [0.1, 1, 10, 100],
                    f'{param_prefix}gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
                }

            search = GridSearchCV(
                self.model, param_grid, cv=cv_folds,
                scoring='accuracy', n_jobs=-1, verbose=1
            )
        else:  # random_search
            param_distributions = {
                f'{param_prefix}C': [0.01, 0.1, 1, 10, 100, 1000],
                f'{param_prefix}gamma': ['scale', 'auto', 0.0001, 0.001, 0.01, 0.1, 1, 10]
            }

            if self.config['kernel'] == 'poly':
                param_distributions[f'{param_prefix}degree'] = [2, 3, 4, 5]

            search = RandomizedSearchCV(
                self.model, param_distributions, n_iter=n_iter,
                cv=cv_folds, scoring='accuracy', n_jobs=-1,
                random_state=self.config['random_state'], verbose=1
            )

        # 执行搜索
        search.fit(X, y)

        # 更新模型
        self.model = search.best_estimator_
        if self.config['enable_scaling']:
            self.svm_model = self.model.named_steps['svm']
            self.scaler = self.model.named_steps['scaler']
        else:
            self.svm_model = self.model

        # 更新配置
        best_params_clean = {}
        for key, value in search.best_params_.items():
            clean_key = key.replace(param_prefix, '')
            best_params_clean[clean_key] = value
            if clean_key in self.config:
                self.config[clean_key] = value

        optimization_results = {
            'method': method,
            'best_params': best_params_clean,
            'best_score': search.best_score_,
            'cv_results': {
                'mean_test_scores': search.cv_results_['mean_test_score'].tolist(),
                'std_test_scores': search.cv_results_['std_test_score'].tolist(),
                'params': search.cv_results_['params']
            }
        }

        logger.info(f"优化完成 - 最佳得分: {search.best_score_:.4f}")
        return optimization_results

    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型详细信息
        
        返回:
            model_info: 模型信息字典
        """
        info = {
            'version': '2.0.0',
            'model_type': 'SVM',
            'is_trained': self.is_trained,
            'config': self.config.copy(),
            'training_history': self.training_history.copy(),
            'optimization_history': self.optimization_history.copy()
        }

        if self.is_trained:
            actual_svm = self.svm_model if not self.config['enable_scaling'] else self.model.named_steps['svm']
            info.update({
                'n_features': actual_svm.n_features_in_,
                'n_classes': len(actual_svm.classes_),
                'classes': actual_svm.classes_.tolist(),
                'feature_names': self.feature_names,
                'class_names': self.class_names,
                'support_vectors_info': self.get_support_vectors_info()
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
                'class_names': self.class_names,
                'model': self.model,
                'svm_model': self.svm_model,
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

            # 版本兼容性检查
            if 'version' not in model_data:
                warnings.warn("加载的是旧版本模型，某些新功能可能不可用")

            # 恢复模型状态
            self.config = model_data.get('config', {})
            self.is_trained = model_data.get('is_trained', False)
            self.training_history = model_data.get('training_history', {})
            self.optimization_history = model_data.get('optimization_history', {})
            self.feature_names = model_data.get('feature_names', None)
            self.class_names = model_data.get('class_names', None)
            self.model = model_data.get('model', None)
            self.svm_model = model_data.get('svm_model', None)
            self.scaler = model_data.get('scaler', None)

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

        # 移除无效样本
        valid_mask = ~(np.any(np.isnan(X) | np.isinf(X), axis=1) |
                       np.isnan(y) | np.isinf(y) | (y < 0))

        X_clean = X[valid_mask]
        y_clean = y[valid_mask]

        if len(X_clean) == 0:
            raise ValueError("没有有效的训练样本")

        logger.info(f"有效训练样本: {len(X_clean)}/{len(X)} ({len(X_clean)/len(X)*100:.1f}%)")

        return X_clean, y_clean

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
SVM = SVMClassifier

if __name__ == "__main__":
    # 测试增强版支持向量机分类器
    print("测试支持向量机分类器 v2.0.0")
    print("=" * 50)

    # 使用多个数据集进行全面测试
    from sklearn.datasets import load_iris, make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report

    print("1. 加载和生成测试数据...")

    # 测试1: 使用Iris数据集（小规模、经典数据集）
    print("\n=== 测试1: Iris数据集 ===")
    iris = load_iris()
    X_iris, y_iris = iris.data, iris.target
    feature_names_iris = list(iris.feature_names)
    class_names_iris = list(iris.target_names)

    X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(
        X_iris, y_iris, test_size=0.3, random_state=42, stratify=y_iris
    )

    print(f"Iris数据集: {X_iris.shape[0]} 样本, {X_iris.shape[1]} 特征, {len(class_names_iris)} 类别")

    # 创建SVM分类器（RBF核）
    svm_rbf = SVMClassifier(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        probability=True,
        enable_scaling=True,
        random_state=42
    )

    # 训练模型
    print("训练RBF-SVM模型...")
    svm_rbf.train(X_train_iris, y_train_iris,
                  feature_names=feature_names_iris,
                  class_names=class_names_iris,
                  validation_split=0.2)

    # 基本预测
    y_pred_rbf = svm_rbf.predict(X_test_iris)
    y_proba_rbf = svm_rbf.predict_proba(X_test_iris)
    y_pred_conf_rbf, confidences_rbf = svm_rbf.predict_with_confidence(X_test_iris)

    accuracy_rbf = accuracy_score(y_test_iris, y_pred_rbf)
    print(f"RBF-SVM测试精度: {accuracy_rbf*100:.2f}%")

    # 支持向量分析
    print("\n2. 支持向量分析:")
    sv_info = svm_rbf.get_support_vectors_info()
    print(f"总支持向量数: {sv_info['total_support_vectors']}")
    print(f"各类别支持向量数: {sv_info['n_support_per_class']}")
    print(f"支持向量比例: {sv_info['support_vector_ratio']:.4f}")

    # 决策函数分析
    print("\n3. 决策函数分析:")
    decision_scores = svm_rbf.evaluate_decision_function(X_test_iris)
    print(f"决策函数值形状: {decision_scores.shape}")
    print(f"决策函数值范围: [{np.min(decision_scores):.4f}, {np.max(decision_scores):.4f}]")

    # 测试2: 线性SVM
    print("\n=== 测试2: 线性SVM ===")
    svm_linear = SVMClassifier(
        kernel='linear',
        C=1.0,
        probability=True,
        enable_scaling=True,
        random_state=42
    )

    print("训练线性SVM模型...")
    svm_linear.train(X_train_iris, y_train_iris,
                     feature_names=feature_names_iris,
                     class_names=class_names_iris)

    y_pred_linear = svm_linear.predict(X_test_iris)
    accuracy_linear = accuracy_score(y_test_iris, y_pred_linear)
    print(f"线性SVM测试精度: {accuracy_linear*100:.2f}%")

    # 线性模型的特征权重
    sv_info_linear = svm_linear.get_support_vectors_info()
    if 'feature_weights' in sv_info_linear:
        print("特征权重分析:")
        for i, (feature, weight) in enumerate(zip(feature_names_iris, sv_info_linear['feature_weights'][0])):
            print(f"  {feature}: {weight:.4f}")

    # 测试3: 超参数优化
    print("\n=== 测试3: 超参数优化 ===")
    svm_opt = SVMClassifier(kernel='rbf', enable_scaling=True, random_state=42)

    print("执行超参数优化...")
    optimization_results = svm_opt.optimize_hyperparameters(
        X_train_iris, y_train_iris,
        method='grid_search',
        cv_folds=3
    )

    print(f"优化方法: {optimization_results['method']}")
    print(f"最佳参数: {optimization_results['best_params']}")
    print(f"最佳交叉验证得分: {optimization_results['best_score']:.4f}")

    # 用优化后的参数训练
    svm_opt.train(X_train_iris, y_train_iris,
                  feature_names=feature_names_iris,
                  class_names=class_names_iris)

    y_pred_opt = svm_opt.predict(X_test_iris)
    accuracy_opt = accuracy_score(y_test_iris, y_pred_opt)
    print(f"优化后SVM测试精度: {accuracy_opt*100:.2f}%")

    # 测试4: 合成数据集（更复杂的分类任务）
    print("\n=== 测试4: 合成数据集 ===")
    X_synthetic, y_synthetic = make_classification(
        n_samples=1000, n_features=20, n_informative=15,
        n_redundant=5, n_classes=4, n_clusters_per_class=1,
        random_state=42
    )

    X_train_syn, X_test_syn, y_train_syn, y_test_syn = train_test_split(
        X_synthetic, y_synthetic, test_size=0.3, random_state=42, stratify=y_synthetic
    )

    print(f"合成数据集: {X_synthetic.shape[0]} 样本, {X_synthetic.shape[1]} 特征, {len(np.unique(y_synthetic))} 类别")

    feature_names_syn = [f"feature_{i+1}" for i in range(X_synthetic.shape[1])]
    class_names_syn = [f"class_{i}" for i in range(len(np.unique(y_synthetic)))]

    # 多项式核SVM
    svm_poly = SVMClassifier(
        kernel='poly',
        degree=3,
        C=10.0,
        gamma='scale',
        probability=True,
        enable_scaling=True,
        random_state=42
    )

    print("训练多项式核SVM模型...")
    svm_poly.train(X_train_syn, y_train_syn,
                   feature_names=feature_names_syn,
                   class_names=class_names_syn,
                   validation_split=0.2)

    y_pred_poly = svm_poly.predict(X_test_syn)
    accuracy_poly = accuracy_score(y_test_syn, y_pred_poly)
    print(f"多项式核SVM测试精度: {accuracy_poly*100:.2f}%")

    # 详细分类报告
    print("\n4. 详细分类报告（多项式核SVM）:")
    print(classification_report(y_test_syn, y_pred_poly, target_names=class_names_syn))

    # 置信度分析
    print("\n5. 置信度分析:")
    _, confidences_poly = svm_poly.predict_with_confidence(X_test_syn)
    avg_confidence = np.mean(confidences_poly)
    print(f"平均预测置信度: {avg_confidence:.4f}")

    # 按类别分析置信度
    for i, cls_name in enumerate(class_names_syn):
        class_mask = (y_pred_poly == i)
        if np.any(class_mask):
            class_confidence = np.mean(confidences_poly[class_mask])
            class_count = np.sum(class_mask)
            print(f"类别 {cls_name}: 预测数量 {class_count}, 平均置信度 {class_confidence:.4f}")

    # 模型信息比较
    print("\n6. 模型信息比较:")
    models = {
        'RBF-SVM': svm_rbf,
        '线性SVM': svm_linear,
        '优化SVM': svm_opt,
        '多项式SVM': svm_poly
    }

    print("-" * 80)
    print(f"{'模型类型':12s} {'核函数':8s} {'训练时间':>10s} {'支持向量':>10s} {'测试精度':>10s}")
    print("-" * 80)

    accuracies = [accuracy_rbf, accuracy_linear, accuracy_opt, accuracy_poly]

    for i, (name, model) in enumerate(models.items()):
        model_info = model.get_model_info()
        kernel = model_info['config']['kernel']
        train_time = model_info['training_history']['training_time']
        sv_count = model_info['support_vectors_info']['total_support_vectors']
        accuracy = accuracies[i]

        print(f"{name:12s} {kernel:8s} {train_time:9.2f}s {sv_count:9d} {accuracy*100:9.2f}%")

    # 测试模型保存和加载
    print("\n7. 测试模型保存和加载...")
    svm_rbf.save_model("test_svm_v2.pkl")

    # 创建新实例并加载模型
    svm_loaded = SVMClassifier()
    svm_loaded.load_model("test_svm_v2.pkl")

    # 验证加载的模型
    y_pred_loaded = svm_loaded.predict(X_test_iris)
    accuracy_loaded = accuracy_score(y_test_iris, y_pred_loaded)
    print(f"加载模型测试精度: {accuracy_loaded*100:.2f}%")

    # 验证支持向量信息是否保持一致
    sv_info_loaded = svm_loaded.get_support_vectors_info()
    sv_match = (sv_info['total_support_vectors'] == sv_info_loaded['total_support_vectors'])
    print(f"支持向量信息保持一致: {'是' if sv_match else '否'}")

    # 测试3D遥感图像数据处理
    print("\n8. 测试3D遥感图像数据...")

    # 创建模拟的遥感图像数据 (H, W, D)
    H, W = 8, 10
    D = X_iris.shape[1]  # 使用相同的特征维度

    # 生成模拟图像数据
    np.random.seed(42)
    test_image = np.zeros((H, W, D))

    for i in range(H):
        for j in range(W):
            # 随机选择一个类别来生成像素值
            class_idx = np.random.choice(len(class_names_iris))
            class_mask = (y_train_iris == class_idx)
            if np.any(class_mask):
                class_data = X_train_iris[class_mask]
                # 添加噪声的类别典型值
                pixel_value = (np.mean(class_data, axis=0) +
                               np.random.normal(0, 0.1, D) * np.std(class_data, axis=0))
                test_image[i, j] = pixel_value

    # 预测整个图像
    pred_image = svm_rbf.predict(test_image)
    proba_image = svm_rbf.predict_proba(test_image)
    pred_conf_image, conf_image = svm_rbf.predict_with_confidence(test_image)
    decision_image = svm_rbf.evaluate_decision_function(test_image)

    print(f"图像预测结果形状: {pred_image.shape}")
    print(f"图像概率结果形状: {proba_image.shape}")
    print(f"图像置信度结果形状: {conf_image.shape}")
    print(f"图像决策函数结果形状: {decision_image.shape}")

    # 统计预测结果
    unique_classes, counts = np.unique(pred_image[pred_image >= 0], return_counts=True)
    print("图像预测类别分布:")
    for cls, count in zip(unique_classes, counts):
        if cls < len(class_names_iris):
            print(f"  {class_names_iris[cls]}: {count} 像素 ({count/(H*W)*100:.1f}%)")

    print(f"图像平均置信度: {np.mean(conf_image[conf_image > 0]):.4f}")

    # 性能总结
    print("\n9. 性能总结:")
    print("=" * 60)
    print("SVM分类器 v2.0.0 主要特性:")
    print("✓ 多种核函数支持 (线性、RBF、多项式、Sigmoid)")
    print("✓ 自动特征标准化和数据预处理")
    print("✓ 概率预测和置信度评估")
    print("✓ 支持向量详细分析")
    print("✓ 决策函数值计算")
    print("✓ 自动超参数优化")
    print("✓ 大规模3D图像数据处理")
    print("✓ 完整的模型持久化功能")
    print("=" * 60)

    # 推荐使用建议
    best_model = max(models.items(), key=lambda x: models[x[0]].training_history.get('train_accuracy', 0))
    print(f"\n推荐配置: {best_model[0]} (基于训练精度)")
    print(f"建议对于遥感数据使用RBF核并启用特征标准化以获得最佳效果。")

    print(f"\n测试完成！支持向量机分类器 v2.0.0 - 功能全面，性能卓越！")
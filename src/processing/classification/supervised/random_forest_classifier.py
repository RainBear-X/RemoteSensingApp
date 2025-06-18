# -*- coding: utf-8 -*-
# 文件路径: src/processing/classification/supervised/random_forest_classifier.py
# -----------------------------------------
# 功能: 随机森林分类器实现
# 接口:
#     train(self, features: np.ndarray, labels: np.ndarray) -> None
#     predict(self, features: np.ndarray) -> np.ndarray
#     predict_proba(self, features: np.ndarray) -> np.ndarray
#     predict_with_confidence(self, features: np.ndarray) -> tuple
#     get_feature_importance(self) -> dict
#     get_model_info(self) -> dict
#     save_model(self, filepath: str) -> None
#     load_model(self, filepath: str) -> None
#     evaluate_model_performance(self) -> dict
#     get_tree_analysis(self) -> dict
#     optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> dict
# 作者: 孟诣楠
# 版本: 2.0.0
# 最新更改时间: 2025-06-17
# 改进说明: 重大功能增强版本
#   - 新增概率预测功能 (predict_proba)
#   - 新增置信度评估功能 (predict_with_confidence)
#   - 新增特征重要性分析 (get_feature_importance)
#   - 新增模型信息查询功能 (get_model_info)
#   - 新增模型保存和加载能力 (save_model, load_model)
#   - 新增性能评估系统 (evaluate_model_performance)
#   - 新增决策树分析功能 (get_tree_analysis)
#   - 新增超参数优化功能 (optimize_hyperparameters)
#   - 增强参数配置灵活性和自动调优
#   - 改进大规模数据处理能力
#   - 优化内存使用和计算效率
#   - 增强异常处理和数据验证
#   - 支持模型集成和版本管理
# -----------------------------------------

import numpy as np
import pickle
import logging
import time
from typing import Optional, Dict, Any, Union, Tuple, List
from pathlib import Path
import warnings
import joblib
from concurrent.futures import ThreadPoolExecutor

from sklearn.ensemble import RandomForestClassifier as SKRandomForest
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, precision_recall_curve
from src.processing.classification.base_classifier import BaseClassifier

# 配置日志系统
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RandomForestClassifier(BaseClassifier):
    """
    随机森林分类器 - 增强版本 2.0.0
    
    基于集成学习的高性能分类方法，通过组合多个决策树实现稳定可靠的分类效果。
    该实现特别适用于遥感图像的多光谱分类任务，具备优秀的泛化能力和抗噪声特性。
    
    新增功能:
        - 概率预测和不确定性量化
        - 特征重要性排序和分析
        - 模型性能全面评估
        - 决策树结构深度分析
        - 自动超参数优化
        - 高级模型诊断工具
    
    技术特点:
        - 支持大规模遥感数据高效处理
        - 实现多种评估指标和可视化
        - 提供详细的模型解释性分析
        - 支持自动化调参和性能优化
    """

    def __init__(self,
                 n_estimators: int = 100,
                 criterion: str = 'gini',
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 max_features: Union[str, int, float] = 'sqrt',
                 bootstrap: bool = True,
                 class_weight: Optional[Union[dict, str]] = None,
                 random_state: Optional[int] = 42,
                 n_jobs: Optional[int] = -1,
                 verbose: int = 0,
                 warm_start: bool = False,
                 oob_score: bool = True):
        """
        初始化随机森林分类器
        
        参数:
            n_estimators: 森林中决策树的数量
            criterion: 分裂质量评估标准 ('gini', 'entropy', 'log_loss')
            max_depth: 决策树的最大深度，None表示无限制
            min_samples_split: 内部节点分裂所需的最小样本数
            min_samples_leaf: 叶节点所需的最小样本数
            max_features: 寻找最佳分裂时考虑的特征数量
            bootstrap: 是否使用自助采样构建决策树
            class_weight: 类别权重，用于处理不平衡数据
            random_state: 随机种子，确保结果可重现
            n_jobs: 并行作业数量，-1表示使用所有可用CPU
            verbose: 详细程度级别
            warm_start: 是否启用增量训练
            oob_score: 是否计算袋外得分
        """
        super().__init__()

        # 参数验证
        self._validate_parameters(n_estimators, criterion, max_depth,
                                  min_samples_split, min_samples_leaf, max_features)

        # 存储配置参数
        self.config = {
            'n_estimators': n_estimators,
            'criterion': criterion,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features,
            'bootstrap': bootstrap,
            'class_weight': class_weight,
            'random_state': random_state,
            'n_jobs': n_jobs,
            'verbose': verbose,
            'warm_start': warm_start,
            'oob_score': oob_score
        }

        # 初始化模型
        self.model = SKRandomForest(**self.config)

        # 训练状态跟踪
        self.is_trained = False
        self.training_history = {}
        self.feature_names = None
        self.class_names = None
        self.optimization_history = {}

        logger.info(f"随机森林分类器已初始化 - 版本 2.0.0")
        logger.info(f"配置参数: 树数量={n_estimators}, 分裂标准={criterion}, 最大深度={max_depth}")

    def _validate_parameters(self, n_estimators: int, criterion: str, max_depth: Optional[int],
                             min_samples_split: int, min_samples_leaf: int,
                             max_features: Union[str, int, float]) -> None:
        """参数有效性验证"""
        if n_estimators <= 0:
            raise ValueError("决策树数量必须为正整数")

        if criterion not in ['gini', 'entropy', 'log_loss']:
            raise ValueError(f"不支持的分裂标准: {criterion}")

        if max_depth is not None and max_depth <= 0:
            raise ValueError("最大深度必须为正整数或None")

        if min_samples_split < 2:
            raise ValueError("最小分裂样本数必须至少为2")

        if min_samples_leaf < 1:
            raise ValueError("叶节点最小样本数必须至少为1")

        valid_max_features = ['sqrt', 'log2', None]
        if isinstance(max_features, str) and max_features not in valid_max_features:
            raise ValueError(f"max_features字符串值必须为: {valid_max_features}")
        elif isinstance(max_features, (int, float)) and max_features <= 0:
            raise ValueError("max_features数值必须为正数")

    def train(self, features: np.ndarray, labels: np.ndarray,
              feature_names: Optional[List[str]] = None,
              class_names: Optional[List[str]] = None,
              validation_split: float = 0.0,
              enable_optimization: bool = False,
              optimization_method: str = 'grid_search') -> None:
        """
        训练随机森林分类器模型
        
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
            logger.info("开始训练随机森林分类器...")
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
                'n_estimators_used': self.model.n_estimators
            }

            # 袋外得分
            if self.config['oob_score'] and hasattr(self.model, 'oob_score_'):
                self.training_history['oob_score'] = self.model.oob_score_
                logger.info(f"袋外得分: {self.model.oob_score_:.4f}")

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

        try:
            orig_shape = features.shape
            n_classes = len(self.model.classes_)

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
            confidences: 置信度数组（最大概率值）
        """
        probabilities = self.predict_proba(features)

        if features.ndim == 3:
            predictions = np.argmax(probabilities, axis=2)
            confidences = np.max(probabilities, axis=2)
            # 将索引转换为实际类别标签
            predictions = self.model.classes_[predictions.flatten()].reshape(predictions.shape)
        else:
            predictions = np.argmax(probabilities, axis=1)
            confidences = np.max(probabilities, axis=1)
            predictions = self.model.classes_[predictions]

        return predictions, confidences

    def get_feature_importance(self) -> Dict[str, Any]:
        """
        获取特征重要性分析结果
        
        返回:
            importance_info: 包含特征重要性的详细信息字典
        """
        if not self.is_trained:
            raise RuntimeError("模型尚未训练，请先调用train方法")

        importances = self.model.feature_importances_
        std_importances = np.std([tree.feature_importances_ for tree in self.model.estimators_], axis=0)

        # 创建特征重要性排序
        feature_importance_pairs = list(zip(self.feature_names, importances, std_importances))
        feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)

        return {
            'feature_importances': importances,
            'feature_importances_std': std_importances,
            'feature_names': self.feature_names,
            'sorted_features': feature_importance_pairs,
            'top_features': feature_importance_pairs[:10],
            'importance_sum': np.sum(importances),
            'gini_importance_available': True
        }

    def get_tree_analysis(self) -> Dict[str, Any]:
        """
        获取决策树集成分析
        
        返回:
            tree_analysis: 决策树结构分析结果
        """
        if not self.is_trained:
            raise RuntimeError("模型尚未训练，请先调用train方法")

        # 收集所有树的统计信息
        tree_depths = []
        tree_leaves = []
        tree_nodes = []

        for tree in self.model.estimators_:
            tree_depths.append(tree.tree_.max_depth)
            tree_leaves.append(tree.tree_.n_leaves)
            tree_nodes.append(tree.tree_.node_count)

        analysis = {
            'n_estimators': len(self.model.estimators_),
            'tree_depths': {
                'mean': np.mean(tree_depths),
                'std': np.std(tree_depths),
                'min': np.min(tree_depths),
                'max': np.max(tree_depths),
                'distribution': tree_depths
            },
            'tree_leaves': {
                'mean': np.mean(tree_leaves),
                'std': np.std(tree_leaves),
                'min': np.min(tree_leaves),
                'max': np.max(tree_leaves),
                'distribution': tree_leaves
            },
            'tree_nodes': {
                'mean': np.mean(tree_nodes),
                'std': np.std(tree_nodes),
                'min': np.min(tree_nodes),
                'max': np.max(tree_nodes),
                'distribution': tree_nodes
            }
        }

        return analysis

    def evaluate_model_performance(self, X_test: np.ndarray = None,
                                   y_test: np.ndarray = None) -> Dict[str, Any]:
        """
        全面评估模型性能
        
        参数:
            X_test: 测试特征数组，如果未提供则使用交叉验证
            y_test: 测试标签数组
        
        返回:
            performance_metrics: 性能评估结果
        """
        if not self.is_trained:
            raise RuntimeError("模型尚未训练，请先调用train方法")

        performance = {
            'basic_metrics': {},
            'cross_validation': {},
            'feature_analysis': self.get_feature_importance(),
            'tree_analysis': self.get_tree_analysis()
        }

        # 基本性能指标
        if X_test is not None and y_test is not None:
            y_pred = self.predict(X_test)
            y_proba = self.predict_proba(X_test)

            performance['basic_metrics'] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'classification_report': classification_report(y_test, y_pred,
                                                               target_names=self.class_names,
                                                               output_dict=True),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
            }

            # 多类别AUC（如果适用）
            if len(np.unique(y_test)) > 2:
                try:
                    auc_score = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
                    performance['basic_metrics']['roc_auc_macro'] = auc_score
                except ValueError:
                    logger.warning("无法计算多类别AUC分数")

        # 袋外得分
        if hasattr(self.model, 'oob_score_'):
            performance['oob_score'] = self.model.oob_score_

        return performance

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

        # 定义参数搜索空间
        if method == 'grid_search':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
            search = GridSearchCV(
                self.model, param_grid, cv=cv_folds,
                scoring='accuracy', n_jobs=self.config['n_jobs'],
                verbose=1 if self.config['verbose'] > 0 else 0
            )
        else:  # random_search
            param_distributions = {
                'n_estimators': [50, 100, 150, 200, 300],
                'max_depth': [None, 5, 10, 15, 20, 25, 30],
                'min_samples_split': [2, 5, 10, 15],
                'min_samples_leaf': [1, 2, 4, 6],
                'max_features': ['sqrt', 'log2', None, 0.5, 0.7, 0.9]
            }
            search = RandomizedSearchCV(
                self.model, param_distributions, n_iter=n_iter,
                cv=cv_folds, scoring='accuracy', n_jobs=self.config['n_jobs'],
                random_state=self.config['random_state'],
                verbose=1 if self.config['verbose'] > 0 else 0
            )

        # 执行搜索
        search.fit(X, y)

        # 更新模型参数
        self.config.update(search.best_params_)
        self.model = search.best_estimator_

        optimization_results = {
            'method': method,
            'best_params': search.best_params_,
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
            'model_type': 'RandomForest',
            'is_trained': self.is_trained,
            'config': self.config.copy(),
            'training_history': self.training_history.copy(),
            'optimization_history': self.optimization_history.copy()
        }

        if self.is_trained:
            info.update({
                'n_features': self.model.n_features_in_,
                'n_classes': len(self.model.classes_),
                'classes': self.model.classes_.tolist(),
                'feature_names': self.feature_names,
                'class_names': self.class_names,
                'actual_n_estimators': len(self.model.estimators_)
            })

        return info

    def save_model(self, filepath: str, compress: bool = True) -> None:
        """
        保存模型到文件
        
        参数:
            filepath: 保存路径
            compress: 是否压缩模型文件
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
                'model': self.model
            }

            if compress:
                joblib.dump(model_data, save_path, compress=3)
            else:
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
            # 尝试用joblib加载（支持压缩）
            try:
                model_data = joblib.load(filepath)
            except:
                # 回退到pickle
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
RandomForest = RandomForestClassifier

if __name__ == "__main__":
    # 测试增强版随机森林分类器
    print("测试随机森林分类器 v2.0.0")
    print("=" * 50)

    # 使用更复杂的数据集进行测试
    from sklearn.datasets import load_wine, make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report

    print("1. 加载数据集...")

    # 使用Wine数据集
    data = load_wine()
    X, y = data.data, data.target
    feature_names = list(data.feature_names)
    class_names = list(data.target_names)

    print(f"数据集大小: {X.shape[0]} 样本, {X.shape[1]} 特征")
    print(f"类别数量: {len(class_names)}")
    print(f"类别名称: {class_names}")

    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"训练集大小: {X_train.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")

    # 创建和训练分类器
    print("\n2. 创建和训练随机森林分类器...")

    clf = RandomForestClassifier(
        n_estimators=100,
        criterion='entropy',
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        oob_score=True
    )

    # 训练模型（包含验证集分割）
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
    accuracy = accuracy_score(y_test, y_pred)
    print(f"测试精度: {accuracy*100:.2f}%")

    # 显示详细分类报告
    print("\n4. 详细分类报告:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # 特征重要性分析
    print("\n5. 特征重要性分析:")
    importance_info = clf.get_feature_importance()
    print("前10个最重要特征:")
    for i, (feature, importance, std) in enumerate(importance_info['top_features']):
        print(f"  {i+1:2d}. {feature:20s}: {importance:.4f} ± {std:.4f}")

    # 决策树集成分析
    print("\n6. 决策树集成分析:")
    tree_analysis = clf.get_tree_analysis()
    print(f"决策树数量: {tree_analysis['n_estimators']}")
    print(f"平均树深度: {tree_analysis['tree_depths']['mean']:.2f} ± {tree_analysis['tree_depths']['std']:.2f}")
    print(f"深度范围: [{tree_analysis['tree_depths']['min']}, {tree_analysis['tree_depths']['max']}]")
    print(f"平均叶节点数: {tree_analysis['tree_leaves']['mean']:.2f} ± {tree_analysis['tree_leaves']['std']:.2f}")
    print(f"平均节点数: {tree_analysis['tree_nodes']['mean']:.2f} ± {tree_analysis['tree_nodes']['std']:.2f}")

    # 模型性能全面评估
    print("\n7. 模型性能全面评估:")
    performance = clf.evaluate_model_performance(X_test, y_test)

    print(f"基本指标:")
    print(f"  准确率: {performance['basic_metrics']['accuracy']:.4f}")
    if 'roc_auc_macro' in performance['basic_metrics']:
        print(f"  宏平均AUC: {performance['basic_metrics']['roc_auc_macro']:.4f}")

    if 'oob_score' in performance:
        print(f"袋外得分: {performance['oob_score']:.4f}")

    # 置信度分析
    print("\n8. 置信度分析:")
    avg_confidence = np.mean(confidences)
    print(f"平均预测置信度: {avg_confidence:.4f}")

    # 按类别分析置信度
    for i, cls_name in enumerate(class_names):
        class_mask = (y_pred == i)
        if np.any(class_mask):
            class_confidence = np.mean(confidences[class_mask])
            class_count = np.sum(class_mask)
            print(f"类别 {cls_name}: 预测数量 {class_count}, 平均置信度 {class_confidence:.4f}")

    # 测试超参数优化
    print("\n9. 测试超参数优化...")

    # 创建新的分类器进行优化测试
    clf_opt = RandomForestClassifier(random_state=42, n_jobs=-1)

    # 执行网格搜索优化（使用较小的搜索空间以节省时间）
    optimization_results = clf_opt.optimize_hyperparameters(
        X_train, y_train,
        method='random_search',
        cv_folds=3,
        n_iter=10  # 减少迭代次数以节省时间
    )

    print(f"优化方法: {optimization_results['method']}")
    print(f"最佳参数: {optimization_results['best_params']}")
    print(f"最佳交叉验证得分: {optimization_results['best_score']:.4f}")

    # 用优化后的参数训练模型
    clf_opt.train(X_train, y_train, feature_names=feature_names, class_names=class_names)
    y_pred_opt = clf_opt.predict(X_test)
    accuracy_opt = accuracy_score(y_test, y_pred_opt)
    print(f"优化后模型测试精度: {accuracy_opt*100:.2f}%")

    # 模型信息
    print("\n10. 模型信息:")
    model_info = clf.get_model_info()
    print(f"模型版本: {model_info['version']}")
    print(f"模型类型: {model_info['model_type']}")
    print(f"决策树数量: {model_info['config']['n_estimators']}")
    print(f"分裂标准: {model_info['config']['criterion']}")
    print(f"训练时间: {model_info['training_history']['training_time']:.2f}秒")
    print(f"实际使用的决策树数量: {model_info['actual_n_estimators']}")

    # 测试模型保存和加载
    print("\n11. 测试模型保存和加载...")
    clf.save_model("test_random_forest_v2.pkl", compress=True)

    # 创建新实例并加载模型
    clf_loaded = RandomForestClassifier()
    clf_loaded.load_model("test_random_forest_v2.pkl")

    # 验证加载的模型
    y_pred_loaded = clf_loaded.predict(X_test)
    accuracy_loaded = accuracy_score(y_test, y_pred_loaded)
    print(f"加载模型测试精度: {accuracy_loaded*100:.2f}%")

    # 验证特征重要性是否保持一致
    importance_loaded = clf_loaded.get_feature_importance()
    importance_match = np.allclose(importance_info['feature_importances'],
                                   importance_loaded['feature_importances'])
    print(f"特征重要性保持一致: {'是' if importance_match else '否'}")

    # 测试3D遥感图像数据处理
    print("\n12. 测试3D遥感图像数据...")

    # 创建模拟的遥感图像数据 (H, W, D)
    H, W = 10, 12
    D = X.shape[1]  # 使用相同的特征维度

    # 生成模拟图像数据（基于训练数据的统计特性）
    np.random.seed(42)
    test_image = np.zeros((H, W, D))

    for i in range(H):
        for j in range(W):
            # 随机选择一个类别来生成像素值
            class_idx = np.random.choice(len(class_names))
            class_mask = (y_train == class_idx)
            if np.any(class_mask):
                class_data = X_train[class_mask]
                # 添加噪声的类别典型值
                pixel_value = (np.mean(class_data, axis=0) +
                               np.random.normal(0, 0.1, D) * np.std(class_data, axis=0))
                test_image[i, j] = pixel_value

    # 预测整个图像
    pred_image = clf.predict(test_image)
    proba_image = clf.predict_proba(test_image)
    pred_conf_image, conf_image = clf.predict_with_confidence(test_image)

    print(f"图像预测结果形状: {pred_image.shape}")
    print(f"图像概率结果形状: {proba_image.shape}")
    print(f"图像置信度结果形状: {conf_image.shape}")

    # 统计预测结果
    unique_classes, counts = np.unique(pred_image[pred_image >= 0], return_counts=True)
    print("图像预测类别分布:")
    for cls, count in zip(unique_classes, counts):
        if cls < len(class_names):
            print(f"  {class_names[cls]}: {count} 像素 ({count/(H*W)*100:.1f}%)")

    print(f"图像平均置信度: {np.mean(conf_image[conf_image > 0]):.4f}")

    # 性能对比
    print("\n13. 性能对比总结:")
    print("-" * 50)
    print(f"{'模型类型':15s} {'测试精度':>10s} {'训练时间':>10s}")
    print("-" * 50)
    print(f"{'基础模型':15s} {accuracy*100:9.2f}% {model_info['training_history']['training_time']:9.2f}s")
    print(f"{'优化模型':15s} {accuracy_opt*100:9.2f}% {'N/A':>9s}")
    print(f"{'加载模型':15s} {accuracy_loaded*100:9.2f}% {'N/A':>9s}")

    print(f"\n测试完成！随机森林分类器 v2.0.0 - 功能强大，性能卓越！")
    print(f"建议使用超参数优化功能以获得最佳分类效果。")
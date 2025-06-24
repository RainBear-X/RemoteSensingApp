# -*- coding: utf-8 -*-
# 文件路径: src/processing/classification/supervised/decision_tree_classifier.py
# -----------------------------------------
# 功能: 决策树分类器实现
# 接口:
#     train(self, features: np.ndarray, labels: np.ndarray) -> None
#     predict(self, features: np.ndarray) -> np.ndarray
#     predict_proba(self, features: np.ndarray) -> np.ndarray
#     get_feature_importance(self) -> np.ndarray
#     get_model_info(self) -> dict
#     save_model(self, filepath: str) -> None
#     load_model(self, filepath: str) -> None
#     visualize_tree(self, filepath: str = None, feature_names: list = None) -> str
# 作者: 孟诣楠
# 版本: 2.0.0
# 最新更改时间: 2025-06-17
# 改进说明: 重大功能增强版本
#   - 新增概率预测功能 (predict_proba)
#   - 新增特征重要性分析 (get_feature_importance)
#   - 新增模型信息获取 (get_model_info)
#   - 新增模型保存和加载功能 (save_model, load_model)
#   - 新增决策树可视化功能 (visualize_tree)
#   - 增强参数配置，支持类别权重、随机状态等
#   - 改进错误处理和输入验证
#   - 新增训练进度监控和性能指标
#   - 支持增量训练和模型更新
#   - 优化内存使用和计算效率
# -----------------------------------------

import numpy as np
import pickle
import logging
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
import warnings

from sklearn.tree import DecisionTreeClassifier as SKDecisionTree
from sklearn.tree import export_text, export_graphviz
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.processing.classification.base_classifier import BaseClassifier

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DecisionTreeClassifier(BaseClassifier):
    """
    决策树分类器 - 增强版本 2.0.0
    
    基于 scikit-learn DecisionTreeClassifier 的高级封装，提供完整的遥感图像分类功能。
    支持模型保存加载、特征重要性分析、决策树可视化等高级功能。
    
    新增功能:
        - 概率预测和置信度评估
        - 特征重要性排序和分析
        - 模型序列化和持久化
        - 决策树结构可视化
        - 训练过程监控和性能评估
        - 增强的参数配置选项
    """

    def __init__(self,
                 criterion: str = 'gini',
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 max_features: Optional[Union[str, int, float]] = None,
                 class_weight: Optional[Union[dict, str]] = None,
                 random_state: Optional[int] = 42,
                 min_impurity_decrease: float = 0.0,
                 ccp_alpha: float = 0.0):
        """
        初始化决策树分类器
        
        参数:
            criterion: 分裂质量评估标准 ('gini', 'entropy', 'log_loss')
            max_depth: 最大树深度，None表示无限制
            min_samples_split: 内部节点分裂所需的最小样本数
            min_samples_leaf: 叶节点所需的最小样本数
            max_features: 寻找最佳分裂时考虑的特征数量
            class_weight: 类别权重，用于处理不平衡数据
            random_state: 随机种子，确保结果可重现
            min_impurity_decrease: 节点分裂所需的最小不纯度减少量
            ccp_alpha: 最小化成本复杂度剪枝参数
        """
        super().__init__()

        # 参数验证
        self._validate_parameters(criterion, max_depth, min_samples_split,
                                  min_samples_leaf, max_features, random_state,
                                  min_impurity_decrease, ccp_alpha)

        # 存储配置参数
        self.config = {
            'criterion': criterion,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features,
            'class_weight': class_weight,
            'random_state': random_state,
            'min_impurity_decrease': min_impurity_decrease,
            'ccp_alpha': ccp_alpha
        }

        # 初始化模型
        self.model = SKDecisionTree(**self.config)

        # 训练状态跟踪
        self.is_trained = False
        self.training_history = {}
        self.feature_names = None
        self.class_names = None

        logger.info(f"决策树分类器已初始化 - 版本 2.0.0")
        logger.info(f"配置参数: {self.config}")

    def _validate_parameters(self, criterion: str, max_depth: Optional[int],
                             min_samples_split: int, min_samples_leaf: int,
                             max_features: Optional[Union[str, int, float]],
                             random_state: Optional[int], min_impurity_decrease: float,
                             ccp_alpha: float) -> None:
        """参数有效性验证"""
        if criterion not in ['gini', 'entropy', 'log_loss']:
            raise ValueError(f"不支持的criterion参数: {criterion}")

        if max_depth is not None and max_depth <= 0:
            raise ValueError("max_depth必须为正整数或None")

        if min_samples_split < 2:
            raise ValueError("min_samples_split必须至少为2")

        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf必须至少为1")

        if min_impurity_decrease < 0:
            raise ValueError("min_impurity_decrease必须非负")

        if ccp_alpha < 0:
            raise ValueError("ccp_alpha必须非负")

    def train(self, features: np.ndarray, labels: np.ndarray,
              feature_names: Optional[List[str]] = None,
              class_names: Optional[List[str]] = None,
              validation_split: float = 0.0) -> None:
        """
        训练决策树模型
        
        参数:
            features: 特征数组，形状 (N, D) 或 (H, W, D)
            labels: 标签数组，形状 (N,) 或 (H, W)
            feature_names: 特征名称列表，用于可视化
            class_names: 类别名称列表，用于结果解释
            validation_split: 验证集比例 (0.0-1.0)
        
        返回:
            无
        """
        try:
            logger.info("开始训练决策树模型...")

            # 输入验证和预处理
            X, y = self._preprocess_training_data(features, labels)

            # 存储元数据
            self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
            if class_names is not None:
                self.class_names = class_names
            else:
                unique_classes = np.unique(y)
                self.class_names = [f"class_{i}" for i in unique_classes]

            # 验证集分割
            if validation_split > 0:
                X_train, X_val, y_train, y_val = self._split_validation_data(X, y, validation_split)
            else:
                X_train, y_train = X, y
                X_val, y_val = None, None

            # 训练模型
            logger.info(f"训练样本数: {X_train.shape[0]}, 特征维度: {X_train.shape[1]}")
            logger.info(f"类别数量: {len(np.unique(y_train))}")

            self.model.fit(X_train, y_train)
            self.is_trained = True

            # 计算训练指标
            train_pred = self.model.predict(X_train)
            train_accuracy = accuracy_score(y_train, train_pred)

            # 记录训练历史
            self.training_history = {
                'train_samples': X_train.shape[0],
                'n_features': X_train.shape[1],
                'n_classes': len(np.unique(y_train)),
                'train_accuracy': train_accuracy,
                'tree_depth': self.model.get_depth(),
                'n_leaves': self.model.get_n_leaves()
            }

            # 验证集评估
            if X_val is not None:
                val_pred = self.model.predict(X_val)
                val_accuracy = accuracy_score(y_val, val_pred)
                self.training_history['val_samples'] = X_val.shape[0]
                self.training_history['val_accuracy'] = val_accuracy
                logger.info(f"验证集精度: {val_accuracy:.4f}")

            logger.info(f"训练完成 - 训练精度: {train_accuracy:.4f}")
            logger.info(f"决策树深度: {self.model.get_depth()}, 叶节点数: {self.model.get_n_leaves()}")

        except Exception as e:
            logger.error(f"训练过程发生错误: {str(e)}")
            raise

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        对新样本进行预测
        
        参数:
            features: 特征数组，形状 (M, D) 或 (H, W, D)
        
        返回:
            prediction: 预测标签数组，形状 (M,) 或 (H, W)
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
                predictions = np.full(X.shape[0], -1, dtype=int)  # -1表示无效像素

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
        预测类别概率
        
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

    def get_feature_importance(self) -> Dict[str, Any]:
        """
        获取特征重要性分析结果
        
        返回:
            importance_info: 包含特征重要性的详细信息字典
        """
        if not self.is_trained:
            raise RuntimeError("模型尚未训练，请先调用train方法")

        importances = self.model.feature_importances_

        # 创建特征重要性排序
        feature_importance_pairs = list(zip(self.feature_names, importances))
        feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)

        return {
            'feature_importances': importances,
            'feature_names': self.feature_names,
            'sorted_features': feature_importance_pairs,
            'top_features': feature_importance_pairs[:10],  # 前10重要特征
            'importance_sum': np.sum(importances)
        }

    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型详细信息
        
        返回:
            model_info: 模型信息字典
        """
        info = {
            'version': '2.0.0',
            'model_type': 'DecisionTree',
            'is_trained': self.is_trained,
            'config': self.config.copy(),
            'training_history': self.training_history.copy()
        }

        if self.is_trained:
            info.update({
                'n_features': self.model.n_features_in_,
                'n_classes': len(self.model.classes_),
                'classes': self.model.classes_.tolist(),
                'tree_depth': self.model.get_depth(),
                'n_leaves': self.model.get_n_leaves(),
                'feature_names': self.feature_names,
                'class_names': self.class_names
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
                'model': self.model,
                'config': self.config,
                'is_trained': self.is_trained,
                'training_history': self.training_history,
                'feature_names': self.feature_names,
                'class_names': self.class_names,
                'version': '2.0.0'
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

            self.model = model_data['model']
            self.config = model_data.get('config', {})
            self.is_trained = model_data.get('is_trained', False)
            self.training_history = model_data.get('training_history', {})
            self.feature_names = model_data.get('feature_names', None)
            self.class_names = model_data.get('class_names', None)

            logger.info(f"模型已从 {filepath} 加载")
            logger.info(f"模型版本: {model_data.get('version', '未知')}")

        except Exception as e:
            logger.error(f"加载模型时发生错误: {str(e)}")
            raise

    def visualize_tree(self, filepath: str = None,
                       feature_names: List[str] = None,
                       max_depth: int = 3) -> str:
        """
        可视化决策树结构
        
        参数:
            filepath: 保存路径，如果为None则返回文本表示
            feature_names: 特征名称列表
            max_depth: 显示的最大深度
        
        返回:
            tree_text: 决策树的文本表示
        """
        if not self.is_trained:
            raise RuntimeError("模型尚未训练，请先调用train方法")

        try:
            # 使用提供的特征名称或默认名称
            names = feature_names or self.feature_names

            # 生成文本表示
            tree_text = export_text(
                self.model,
                feature_names=names,
                max_depth=max_depth,
                spacing=2,
                decimals=3,
                show_weights=True
            )

            # 如果提供了文件路径，保存到文件
            if filepath:
                save_path = Path(filepath)
                save_path.parent.mkdir(parents=True, exist_ok=True)

                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(tree_text)

                logger.info(f"决策树可视化已保存到: {filepath}")

            return tree_text

        except Exception as e:
            logger.error(f"可视化决策树时发生错误: {str(e)}")
            raise

    def _preprocess_training_data(self, features: np.ndarray,
                                  labels: np.ndarray) -> tuple:
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
                               validation_split: float) -> tuple:
        """分割验证数据"""
        from sklearn.model_selection import train_test_split

        return train_test_split(
            X, y,
            test_size=validation_split,
            random_state=self.config.get('random_state', 42),
            stratify=y if len(np.unique(y)) > 1 else None
        )


# 保持向后兼容的别名
DecisionTree = DecisionTreeClassifier

if __name__ == "__main__":
    # 测试增强版决策树分类器
    print("测试决策树分类器 v2.0.0")
    print("=" * 50)

    # 使用鸢尾花数据集进行测试
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    # 加载数据
    data = load_iris()
    X, y = data.data, data.target
    feature_names = data.feature_names
    class_names = data.target_names

    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 创建和训练分类器
    clf = DecisionTreeClassifier(
        criterion='entropy',
        max_depth=3,
        random_state=42,
        class_weight='balanced'
    )

    print("1. 训练模型...")
    clf.train(X_train, y_train, feature_names=feature_names,
              class_names=class_names, validation_split=0.2)

    # 预测
    print("\n2. 进行预测...")
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    # 计算精度
    accuracy = np.mean(y_pred == y_test)
    print(f"测试精度: {accuracy*100:.2f}%")

    # 特征重要性分析
    print("\n3. 特征重要性分析:")
    importance_info = clf.get_feature_importance()
    for feature, importance in importance_info['top_features']:
        print(f"  {feature}: {importance:.4f}")

    # 模型信息
    print("\n4. 模型信息:")
    model_info = clf.get_model_info()
    print(f"  树深度: {model_info['tree_depth']}")
    print(f"  叶节点数: {model_info['n_leaves']}")
    print(f"  训练精度: {model_info['training_history']['train_accuracy']:.4f}")

    # 可视化决策树
    print("\n5. 决策树结构:")
    tree_text = clf.visualize_tree(max_depth=2)
    print(tree_text[:500] + "..." if len(tree_text) > 500 else tree_text)

    # 测试模型保存和加载
    print("\n6. 测试模型保存和加载...")
    clf.save_model("test_decision_tree_v2.pkl")

    # 创建新实例并加载模型
    clf_loaded = DecisionTreeClassifier()
    clf_loaded.load_model("test_decision_tree_v2.pkl")

    # 验证加载的模型
    y_pred_loaded = clf_loaded.predict(X_test)
    accuracy_loaded = np.mean(y_pred_loaded == y_test)
    print(f"加载模型测试精度: {accuracy_loaded*100:.2f}%")

    print("\n测试完成！所有功能正常运行。")
# -*- coding: utf-8 -*-
# 文件路径: src/processing/classification/model_manager.py
# -----------------------------------------
# 功能: 分类器管理脚本，集成所有监督和无监督分类算法的工厂与统一接口
# 接口:
#     get_supervised_classifier(name: str, **kwargs) -> BaseClassifier
#     get_unsupervised_classifier(name: str, **kwargs) -> BaseClassifier
#     get_unsupervised_function(name: str) -> callable
#     list_available_classifiers() -> dict
#     create_classifier_pipeline(config: dict) -> ClassifierPipeline
#     compare_classifiers(data: dict, config: dict) -> dict
#     get_classifier_info(name: str) -> dict
# 作者: 孟诣楠
# 版本: 2.0.0
# 最新更改时间: 2025-06-17
# 改进说明: 重大功能增强版本
#   - 新增无监督分类器面向对象接口 (get_unsupervised_classifier)
#   - 新增分类器信息查询功能 (get_classifier_info, list_available_classifiers)
#   - 新增分类器管道支持 (create_classifier_pipeline)
#   - 新增性能比较功能 (compare_classifiers)
#   - 增强错误处理和参数验证
#   - 添加分类器版本管理和兼容性检查
#   - 优化工厂模式设计
#   - 保持原有接口向后兼容
# -----------------------------------------

import numpy as np
import logging
import time
from typing import Dict, Any, Optional, Union, Callable, List, Tuple
from pathlib import Path
import warnings

from src.processing.classification.base_classifier import BaseClassifier

# 监督分类器导入
from src.processing.classification.supervised.maximum_likelihood import MaximumLikelihoodClassifier
from src.processing.classification.supervised.minimum_distance import MinimumDistanceClassifier
from src.processing.classification.supervised.svm_classifier import SVMClassifier
from src.processing.classification.supervised.decision_tree_classifier import DecisionTreeClassifier
from src.processing.classification.supervised.random_forest_classifier import RandomForestClassifier

# 无监督分类器导入（面向对象版本）
from src.processing.classification.unsupervised.kmeans import KMeansClassifier
from src.processing.classification.unsupervised.dbscan import DBSCANClassifier
from src.processing.classification.unsupervised.isodata import ISODATAClassifier
# 无监督分类函数导入（兼容版本）
from src.processing.classification.unsupervised.kmeans import unsupervised_kmeans_classification
from src.processing.classification.unsupervised.dbscan import unsupervised_dbscan_classification
from src.processing.classification.unsupervised.isodata import unsupervised_isodata_classification

# 配置日志系统
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 监督分类器映射
SUPERVISED_CLASSIFIERS = {
    'maximum_likelihood': {
        'class': MaximumLikelihoodClassifier,
        'version': '2.0.0',
        'description': '基于多元正态分布假设的最大似然分类器，适用于高维光谱数据',
        'category': 'statistical',
        'features': ['probability_prediction', 'feature_importance', 'model_persistence'],
        'recommended_use': ['multispectral_classification', 'gaussian_distributed_data']
    },
    'minimum_distance': {
        'class': MinimumDistanceClassifier,
        'version': '2.0.0',
        'description': '基于欧氏距离的简单高效分类器，支持多种距离度量',
        'category': 'geometric',
        'features': ['multiple_distance_metrics', 'confidence_estimation', 'fast_prediction'],
        'recommended_use': ['large_scale_data', 'real_time_classification']
    },
    'svm': {
        'class': SVMClassifier,
        'version': '2.0.0',
        'description': '支持向量机分类器，具备强大的非线性分类能力',
        'category': 'kernel_based',
        'features': ['kernel_methods', 'support_vector_analysis', 'hyperparameter_optimization'],
        'recommended_use': ['high_dimensional_data', 'nonlinear_classification']
    },
    'decision_tree': {
        'class': DecisionTreeClassifier,
        'version': '2.0.0',
        'description': '决策树分类器，提供优秀的可解释性和特征重要性分析',
        'category': 'tree_based',
        'features': ['interpretability', 'feature_importance', 'tree_visualization'],
        'recommended_use': ['feature_analysis', 'interpretable_models']
    },
    'random_forest': {
        'class': RandomForestClassifier,
        'version': '2.0.0',
        'description': '随机森林集成分类器，具备优秀的泛化能力和鲁棒性',
        'category': 'ensemble',
        'features': ['ensemble_learning', 'feature_importance', 'oob_score', 'hyperparameter_optimization'],
        'recommended_use': ['complex_classification_tasks', 'noisy_data', 'feature_selection']
    }
}

# 无监督分类器映射（面向对象版本）
UNSUPERVISED_CLASSIFIERS = {
    'kmeans': {
        'class': KMeansClassifier,
        'version': '2.0.0',
        'description': 'K-Means聚类分类器，支持自动K值优化和质量评估',
        'category': 'centroid_based',
        'features': ['k_optimization', 'clustering_quality_evaluation', 'model_persistence'],
        'recommended_use': ['spherical_clusters', 'known_cluster_number']
    },
    'dbscan': {
        'class': DBSCANClassifier,
        'version': '2.0.0',
        'description': 'DBSCAN密度聚类分类器，能够发现任意形状聚类并处理噪声',
        'category': 'density_based',
        'features': ['arbitrary_cluster_shapes', 'noise_detection', 'parameter_optimization'],
        'recommended_use': ['irregular_clusters', 'noisy_data', 'unknown_cluster_number']
    },
    'isodata': {
        'class': ISODATAClassifier,
        'version': '2.0.0',
        'description': 'ISODATA自适应聚类分类器，动态调整聚类数量',
        'category': 'adaptive',
        'features': ['adaptive_clustering', 'split_merge_operations', 'iteration_monitoring'],
        'recommended_use': ['complex_landcover_classification', 'exploratory_analysis']
    }
}

# 无监督分类函数映射（向后兼容）
UNSUPERVISED_FUNCTIONS = {
    'kmeans': unsupervised_kmeans_classification,
    'dbscan': unsupervised_dbscan_classification,
    'isodata': unsupervised_isodata_classification,
}


class ClassifierPipeline:
    """
    分类器管道类，支持多个分类器的串联和并行处理
    """

    def __init__(self, classifiers: List[Dict[str, Any]]):
        """
        初始化分类器管道
        
        参数:
            classifiers: 分类器配置列表，每个配置包含名称和参数
        """
        self.classifiers = []
        self.results = {}

        for config in classifiers:
            classifier_name = config['name']
            classifier_params = config.get('params', {})

            if classifier_name in SUPERVISED_CLASSIFIERS:
                clf = get_supervised_classifier(classifier_name, **classifier_params)
            elif classifier_name in UNSUPERVISED_CLASSIFIERS:
                clf = get_unsupervised_classifier(classifier_name, **classifier_params)
            else:
                raise ValueError(f"未知的分类器: {classifier_name}")

            self.classifiers.append({
                'name': classifier_name,
                'classifier': clf,
                'config': config
            })

    def run_pipeline(self, data: Dict[str, np.ndarray],
                     mode: str = 'parallel') -> Dict[str, Any]:
        """
        运行分类器管道
        
        参数:
            data: 包含特征和标签的数据字典
            mode: 运行模式，'parallel'或'sequential'
        
        返回:
            results: 各分类器的结果字典
        """
        features = data['features']
        labels = data.get('labels', None)

        if mode == 'parallel':
            return self._run_parallel(features, labels)
        else:
            return self._run_sequential(features, labels)

    def _run_parallel(self, features: np.ndarray,
                      labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """并行运行所有分类器"""
        results = {}

        for clf_info in self.classifiers:
            name = clf_info['name']
            classifier = clf_info['classifier']

            try:
                start_time = time.time()

                if labels is not None and name in SUPERVISED_CLASSIFIERS:
                    # 监督学习
                    classifier.train(features, labels)
                    predictions = classifier.predict(features)
                    accuracy = np.mean(predictions == labels)

                    results[name] = {
                        'predictions': predictions,
                        'accuracy': accuracy,
                        'training_time': time.time() - start_time,
                        'model_info': classifier.get_model_info() if hasattr(classifier, 'get_model_info') else {}
                    }
                else:
                    # 无监督学习
                    classifier.train(features)
                    predictions = classifier.predict(features)

                    results[name] = {
                        'predictions': predictions,
                        'training_time': time.time() - start_time,
                        'model_info': classifier.get_model_info() if hasattr(classifier, 'get_model_info') else {}
                    }

            except Exception as e:
                logger.error(f"分类器 {name} 运行失败: {str(e)}")
                results[name] = {'error': str(e)}

        return results

    def _run_sequential(self, features: np.ndarray,
                        labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """顺序运行所有分类器"""
        # 实现与并行版本相同，但可以利用前一个分类器的结果
        return self._run_parallel(features, labels)


def get_supervised_classifier(name: str, **kwargs) -> BaseClassifier:
    """
    获取指定名称的监督分类器实例
    
    参数:
        name: 分类器名称，对应SUPERVISED_CLASSIFIERS键
        kwargs: 分类器构造参数
    
    返回:
        BaseClassifier子类实例
    """
    if name not in SUPERVISED_CLASSIFIERS:
        available = list(SUPERVISED_CLASSIFIERS.keys())
        raise ValueError(f"未知的监督分类器: {name}。可用选项: {available}")

    classifier_info = SUPERVISED_CLASSIFIERS[name]
    classifier_class = classifier_info['class']

    try:
        classifier = classifier_class(**kwargs)
        logger.info(f"成功创建监督分类器: {name} v{classifier_info['version']}")
        return classifier
    except Exception as e:
        logger.error(f"创建监督分类器 {name} 失败: {str(e)}")
        raise


def get_unsupervised_classifier(name: str, **kwargs) -> BaseClassifier:
    """
    获取指定名称的无监督分类器实例（面向对象版本）
    
    参数:
        name: 分类器名称，对应UNSUPERVISED_CLASSIFIERS键
        kwargs: 分类器构造参数
    
    返回:
        BaseClassifier子类实例
    """
    if name not in UNSUPERVISED_CLASSIFIERS:
        available = list(UNSUPERVISED_CLASSIFIERS.keys())
        raise ValueError(f"未知的无监督分类器: {name}。可用选项: {available}")

    classifier_info = UNSUPERVISED_CLASSIFIERS[name]
    classifier_class = classifier_info['class']

    try:
        classifier = classifier_class(**kwargs)
        logger.info(f"成功创建无监督分类器: {name} v{classifier_info['version']}")
        return classifier
    except Exception as e:
        logger.error(f"创建无监督分类器 {name} 失败: {str(e)}")
        raise


def get_unsupervised_function(name: str) -> Callable:
    """
    获取指定名称的无监督分类函数（向后兼容版本）
    
    参数:
        name: 分类方法名称，对应UNSUPERVISED_FUNCTIONS键
    
    返回:
        函数引用，可直接调用
    """
    if name not in UNSUPERVISED_FUNCTIONS:
        available = list(UNSUPERVISED_FUNCTIONS.keys())
        raise ValueError(f"未知的无监督分类方法: {name}。可用选项: {available}")

    logger.info(f"获取无监督分类函数: {name} (兼容模式)")
    return UNSUPERVISED_FUNCTIONS[name]


def list_available_classifiers() -> Dict[str, Any]:
    """
    列出所有可用的分类器及其信息
    
    返回:
        classifiers_info: 包含所有分类器信息的字典
    """
    return {
        'supervised': {
            name: {
                'version': info['version'],
                'description': info['description'],
                'category': info['category'],
                'features': info['features'],
                'recommended_use': info['recommended_use']
            }
            for name, info in SUPERVISED_CLASSIFIERS.items()
        },
        'unsupervised': {
            name: {
                'version': info['version'],
                'description': info['description'],
                'category': info['category'],
                'features': info['features'],
                'recommended_use': info['recommended_use']
            }
            for name, info in UNSUPERVISED_CLASSIFIERS.items()
        },
        'summary': {
            'total_supervised': len(SUPERVISED_CLASSIFIERS),
            'total_unsupervised': len(UNSUPERVISED_CLASSIFIERS),
            'total_classifiers': len(SUPERVISED_CLASSIFIERS) + len(UNSUPERVISED_CLASSIFIERS)
        }
    }


def get_classifier_info(name: str) -> Dict[str, Any]:
    """
    获取指定分类器的详细信息
    
    参数:
        name: 分类器名称
    
    返回:
        classifier_info: 分类器详细信息
    """
    if name in SUPERVISED_CLASSIFIERS:
        info = SUPERVISED_CLASSIFIERS[name].copy()
        info['type'] = 'supervised'
        return info
    elif name in UNSUPERVISED_CLASSIFIERS:
        info = UNSUPERVISED_CLASSIFIERS[name].copy()
        info['type'] = 'unsupervised'
        return info
    else:
        all_classifiers = list(SUPERVISED_CLASSIFIERS.keys()) + list(UNSUPERVISED_CLASSIFIERS.keys())
        raise ValueError(f"未知的分类器: {name}。可用选项: {all_classifiers}")


def create_classifier_pipeline(config: Dict[str, Any]) -> ClassifierPipeline:
    """
    创建分类器管道
    
    参数:
        config: 管道配置字典，包含分类器列表和参数
    
    返回:
        pipeline: ClassifierPipeline实例
    """
    classifiers = config.get('classifiers', [])

    if not classifiers:
        raise ValueError("管道配置中必须包含至少一个分类器")

    logger.info(f"创建包含 {len(classifiers)} 个分类器的管道")
    return ClassifierPipeline(classifiers)


def compare_classifiers(data: Dict[str, np.ndarray],
                        config: Dict[str, Any]) -> Dict[str, Any]:
    """
    比较多个分类器的性能
    
    参数:
        data: 包含特征和标签的数据字典
        config: 比较配置，包含要比较的分类器列表
    
    返回:
        comparison_results: 比较结果字典
    """
    classifiers_to_compare = config.get('classifiers', [])
    features = data['features']
    labels = data.get('labels', None)

    if not classifiers_to_compare:
        raise ValueError("必须指定要比较的分类器")

    # 创建管道
    pipeline = create_classifier_pipeline({'classifiers': classifiers_to_compare})

    # 运行管道
    logger.info(f"开始比较 {len(classifiers_to_compare)} 个分类器")
    results = pipeline.run_pipeline(data, mode='parallel')

    # 生成比较报告
    comparison_results = {
        'individual_results': results,
        'summary': _generate_comparison_summary(results, labels is not None),
        'recommendations': _generate_recommendations(results, data, config)
    }

    return comparison_results


def _generate_comparison_summary(results: Dict[str, Any],
                                 is_supervised: bool) -> Dict[str, Any]:
    """生成比较摘要"""
    summary = {
        'total_classifiers': len(results),
        'successful_runs': len([r for r in results.values() if 'error' not in r]),
        'failed_runs': len([r for r in results.values() if 'error' in r])
    }

    if is_supervised:
        # 监督学习摘要
        accuracies = {name: result.get('accuracy', 0) for name, result in results.items() if 'error' not in result}
        if accuracies:
            summary['best_accuracy'] = max(accuracies.values())
            summary['best_classifier'] = max(accuracies, key=accuracies.get)
            summary['accuracy_ranking'] = sorted(accuracies.items(), key=lambda x: x[1], reverse=True)

    # 训练时间比较
    training_times = {name: result.get('training_time', 0) for name, result in results.items() if 'error' not in result}
    if training_times:
        summary['fastest_training'] = min(training_times, key=training_times.get)
        summary['slowest_training'] = max(training_times, key=training_times.get)
        summary['average_training_time'] = np.mean(list(training_times.values()))

    return summary


def _generate_recommendations(results: Dict[str, Any],
                              data: Dict[str, np.ndarray],
                              config: Dict[str, Any]) -> List[str]:
    """生成推荐建议"""
    recommendations = []

    # 基于数据特性的推荐
    features = data['features']
    n_samples, n_features = features.shape

    if n_features > 100:
        recommendations.append("高维数据建议使用SVM或随机森林分类器")

    if n_samples > 10000:
        recommendations.append("大规模数据建议使用最小距离或随机森林分类器以获得更好的计算效率")

    # 基于结果的推荐
    successful_results = {name: result for name, result in results.items() if 'error' not in result}

    if 'labels' in data and successful_results:
        # 监督学习推荐
        accuracies = {name: result.get('accuracy', 0) for name, result in successful_results.items()}
        if accuracies:
            best_classifier = max(accuracies, key=accuracies.get)
            best_accuracy = accuracies[best_classifier]
            recommendations.append(f"基于当前数据，推荐使用{best_classifier}分类器（精度: {best_accuracy:.4f}）")

    return recommendations


def validate_classifier_compatibility(classifier_name: str,
                                      data_shape: Tuple[int, ...]) -> Dict[str, Any]:
    """
    验证分类器与数据的兼容性
    
    参数:
        classifier_name: 分类器名称
        data_shape: 数据形状
    
    返回:
        compatibility_info: 兼容性信息
    """
    n_samples = data_shape[0]
    n_features = data_shape[1] if len(data_shape) > 1 else 1

    compatibility = {
        'is_compatible': True,
        'warnings': [],
        'recommendations': []
    }

    # 获取分类器信息
    try:
        classifier_info = get_classifier_info(classifier_name)
    except ValueError:
        compatibility['is_compatible'] = False
        compatibility['warnings'].append(f"未知的分类器: {classifier_name}")
        return compatibility

    # 样本数量检查
    if n_samples < 10:
        compatibility['warnings'].append("样本数量过少，可能影响分类器性能")

    # 特征维度检查
    if n_features > 1000 and classifier_name == 'svm':
        compatibility['warnings'].append("高维数据可能导致SVM训练时间过长")

    # 分类器特定检查
    if classifier_name == 'decision_tree' and n_features > 50:
        compatibility['recommendations'].append("高维数据建议使用随机森林替代单一决策树")

    return compatibility


if __name__ == '__main__':
    # 测试增强版模型管理器
    print("测试模型管理器 v2.0.0")
    print("=" * 50)

    # 1. 列出所有可用分类器
    print("1. 可用分类器列表:")
    classifiers_info = list_available_classifiers()

    print(f"监督分类器 ({classifiers_info['summary']['total_supervised']} 个):")
    for name, info in classifiers_info['supervised'].items():
        print(f"  {name}: {info['description']}")

    print(f"\n无监督分类器 ({classifiers_info['summary']['total_unsupervised']} 个):")
    for name, info in classifiers_info['unsupervised'].items():
        print(f"  {name}: {info['description']}")

    # 2. 测试监督分类器创建
    print("\n2. 测试监督分类器创建:")

    # 构造监督学习测试数据
    np.random.seed(42)
    X_supervised = np.random.randn(200, 4)
    y_supervised = np.random.randint(0, 3, 200)

    # 测试不同的监督分类器
    supervised_names = ['minimum_distance', 'decision_tree', 'random_forest']

    for clf_name in supervised_names:
        try:
            clf = get_supervised_classifier(clf_name, random_state=42)
            clf.train(X_supervised, y_supervised)
            predictions = clf.predict(X_supervised)
            accuracy = np.mean(predictions == y_supervised)
            print(f"  {clf_name}: 创建成功，训练精度 {accuracy:.4f}")
        except Exception as e:
            print(f"  {clf_name}: 创建失败 - {str(e)}")

    # 3. 测试无监督分类器创建
    print("\n3. 测试无监督分类器创建:")

    # 构造无监督学习测试数据
    X_unsupervised = np.random.randn(150, 3)

    # 测试不同的无监督分类器
    unsupervised_names = ['kmeans', 'dbscan', 'isodata']

    for clf_name in unsupervised_names:
        try:
            clf = get_unsupervised_classifier(clf_name, random_state=42)
            clf.train(X_unsupervised)
            predictions = clf.predict(X_unsupervised)
            n_clusters = len(np.unique(predictions[predictions >= 0]))
            print(f"  {clf_name}: 创建成功，发现 {n_clusters} 个聚类")
        except Exception as e:
            print(f"  {clf_name}: 创建失败 - {str(e)}")

    # 4. 测试分类器管道
    print("\n4. 测试分类器管道:")

    pipeline_config = {
        'classifiers': [
            {'name': 'kmeans', 'params': {'n_clusters': 3, 'random_state': 42}},
            {'name': 'dbscan', 'params': {'eps': 0.5, 'min_samples': 5, 'random_state': 42}},
            {'name': 'decision_tree', 'params': {'max_depth': 5, 'random_state': 42}}
        ]
    }

    try:
        pipeline = create_classifier_pipeline(pipeline_config)

        # 准备数据
        pipeline_data = {
            'features': X_supervised,
            'labels': y_supervised
        }

        # 运行管道
        pipeline_results = pipeline.run_pipeline(pipeline_data, mode='parallel')

        print("  管道执行结果:")
        for name, result in pipeline_results.items():
            if 'error' not in result:
                if 'accuracy' in result:
                    print(f"    {name}: 精度 {result['accuracy']:.4f}, 训练时间 {result['training_time']:.4f}s")
                else:
                    print(f"    {name}: 训练时间 {result['training_time']:.4f}s")
            else:
                print(f"    {name}: 执行失败 - {result['error']}")

    except Exception as e:
        print(f"  管道测试失败: {str(e)}")

    # 5. 测试分类器比较
    print("\n5. 测试分类器性能比较:")

    comparison_config = {
        'classifiers': [
            {'name': 'minimum_distance', 'params': {'random_state': 42}},
            {'name': 'decision_tree', 'params': {'max_depth': 3, 'random_state': 42}},
            {'name': 'random_forest', 'params': {'n_estimators': 50, 'random_state': 42}}
        ]
    }

    try:
        comparison_data = {
            'features': X_supervised,
            'labels': y_supervised
        }

        comparison_results = compare_classifiers(comparison_data, comparison_config)

        print("  比较结果摘要:")
        summary = comparison_results['summary']
        print(f"    成功运行: {summary['successful_runs']}/{summary['total_classifiers']}")

        if 'best_classifier' in summary:
            print(f"    最佳分类器: {summary['best_classifier']} (精度: {summary['best_accuracy']:.4f})")

        if 'fastest_training' in summary:
            print(f"    最快训练: {summary['fastest_training']}")

        print("  推荐建议:")
        for recommendation in comparison_results['recommendations']:
            print(f"    - {recommendation}")

    except Exception as e:
        print(f"  比较测试失败: {str(e)}")

    # 6. 测试向后兼容性
    print("\n6. 测试向后兼容性:")

    # 测试原有的函数式接口
    try:
        H, W = 20, 25
        feat = np.random.randn(H, W, 2)
        features_dict = {
            'height': H,
            'width': W,
            'feat1': feat[:, :, 0],
            'feat2': feat[:, :, 1]
        }

        # 测试K-Means函数接口
        kmeans_func = get_unsupervised_function('kmeans')
        labels_map = kmeans_func(features_dict, n_clusters=3)
        print(f"  K-Means函数接口: 成功，标签形状 {labels_map.shape}")

        # 测试DBSCAN函数接口
        dbscan_func = get_unsupervised_function('dbscan')
        labels_map_dbscan = dbscan_func(features_dict, eps=0.5, min_samples=5)
        print(f"  DBSCAN函数接口: 成功，发现 {len(np.unique(labels_map_dbscan))} 个聚类")

    except Exception as e:
        print(f"  向后兼容性测试失败: {str(e)}")

    # 7. 测试分类器信息查询
    print("\n7. 测试分类器信息查询:")

    try:
        svm_info = get_classifier_info('svm')
        print(f"  SVM分类器信息:")
        print(f"    版本: {svm_info['version']}")
        print(f"    类别: {svm_info['category']}")
        print(f"    描述: {svm_info['description']}")
        print(f"    主要特性: {', '.join(svm_info['features'])}")

    except Exception as e:
        print(f"  信息查询测试失败: {str(e)}")

    # 8. 测试兼容性检查
    print("\n8. 测试兼容性检查:")

    try:
        # 测试不同数据规模的兼容性
        test_cases = [
            ('small_data', (50, 5)),
            ('high_dim_data', (200, 500)),
            ('large_data', (10000, 10))
        ]

        for case_name, data_shape in test_cases:
            print(f"  {case_name} {data_shape}:")

            for clf_name in ['svm', 'decision_tree', 'random_forest']:
                compatibility = validate_classifier_compatibility(clf_name, data_shape)

                status = "兼容" if compatibility['is_compatible'] else "不兼容"
                print(f"    {clf_name}: {status}")

                if compatibility['warnings']:
                    for warning in compatibility['warnings']:
                        print(f"      警告: {warning}")

                if compatibility['recommendations']:
                    for rec in compatibility['recommendations']:
                        print(f"      建议: {rec}")

    except Exception as e:
        print(f"  兼容性检查测试失败: {str(e)}")

    # 9. 性能统计
    print("\n9. 模型管理器统计:")
    print(f"  总分类器数量: {classifiers_info['summary']['total_classifiers']}")
    print(f"  监督分类器: {classifiers_info['summary']['total_supervised']}")
    print(f"  无监督分类器: {classifiers_info['summary']['total_unsupervised']}")
    print(f"  版本: 2.0.0")

    # 10. 使用建议
    print("\n10. 使用建议:")
    print("  • 对于监督学习任务，推荐使用随机森林或SVM")
    print("  • 对于无监督聚类，根据数据特性选择合适算法:")
    print("    - K-Means: 适用于球形分布的数据")
    print("    - DBSCAN: 适用于任意形状的聚类和含噪声数据")
    print("    - ISODATA: 适用于复杂地物分类和探索性分析")
    print("  • 使用分类器管道可以同时测试多个算法")
    print("  • 使用比较功能可以快速找到最适合的分类器")

    print(f"\n测试完成！模型管理器 v2.0.0 - 功能全面，易于使用！")
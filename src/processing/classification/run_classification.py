# -*- coding: utf-8 -*-
"""
文件: run_classification.py
模块: src.processing.classification.run_classification
功能: 分类模块入口，兼容额外参数并自动读取 config.classification_params
作者: 孟诣楠
版本: v1.0.3
最近更新: 2025-06-18

更新说明:
    - 修正传入 data 中路径类型的 features 和 labels 时，自动加载为 numpy.ndarray
"""

from typing import Any, Dict, List
from src.processing.task_result import TaskResult
from src.processing.classification.model_manager import create_classifier_pipeline, compare_classifiers

import numpy as np  # 用于加载 .npy 文件

def run(
        config: Any,
        data: Dict[str, Any],
        pipeline_config: Dict[str, Any],
        mode: str = "parallel",
        **kwargs
) -> TaskResult:
    """
    分类模块统一入口：
      - 根据 pipeline_config 创建并执行分类器管道
      - 自动读取并应用 config.classification_params 中的额外参数
      - 支持 compare 标志执行分类器性能比较

    参数:
        config: 引擎配置对象，需包含可选属性 classification_params
        data: 数据字典，需包含 'features': np.ndarray 或 str，可选 'labels': np.ndarray 或 str
        pipeline_config: 管道配置，示例：
            {
              'classifiers': [{ 'name':'random_forest','params':{'n_estimators':100}}],
              'compare': False
            }
        mode: 执行模式，'parallel' 或 'sequential'
        **kwargs: 透传额外参数，如 class_map_path
    返回:
        TaskResult，outputs 包含分类结果或比较结果字典，logs 为执行日志列表
    """
    logs: List[str] = []
    outputs: List[Any] = []

    # 合并 config.classification_params
    if config is not None and hasattr(config, 'classification_params'):
        classification_params = getattr(config, 'classification_params') or {}
        kwargs.update(classification_params)
        logs.append(f"已读取 config.classification_params: {classification_params}")

    # —— 加载 features 和 labels（支持路径和数组） ——
    if isinstance(data.get('features'), str):
        fp = data['features']
        try:
            data['features'] = np.load(fp)
            logs.append(f"已加载 features 数组 从 '{fp}'")
        except Exception as e:
            logs.append(f"加载 features 时出错: {e}")
            return TaskResult(status='failure', message=str(e), outputs=[], logs=logs)

    if 'labels' in data and isinstance(data['labels'], str):
        lp = data['labels']
        try:
            data['labels'] = np.load(lp)
            logs.append(f"已加载 labels 数组 从 '{lp}'")
        except Exception as e:
            logs.append(f"加载 labels 时出错: {e}")
            return TaskResult(status='failure', message=str(e), outputs=[], logs=logs)

    # 处理 model 参数，若未在 pipeline_config 中定义 classifiers，则创建默认分类器
    model_name = kwargs.pop('model', None)
    if model_name:
        if not pipeline_config.get('classifiers'):
            pipeline_config['classifiers'] = [{'name': model_name, 'params': {}}]
            logs.append(f"基于 model 参数创建分类器配置: {model_name}")
        else:
            logs.append(f"忽略 model 参数，因为 pipeline_config 已定义 classifiers: {pipeline_config['classifiers']}")
    try:
        # 性能比较
        if pipeline_config.get('compare', False):
            results = compare_classifiers(data, pipeline_config)
            logs.append('分类器性能比较完成')
            outputs.append(results)
            return TaskResult(status='success', message='分类比较完成', outputs=outputs, logs=logs)

        # 创建并运行分类管道
        pipeline = create_classifier_pipeline(pipeline_config)
        logs.append(f"已创建分类管道，分类器数量: {len(pipeline_config.get('classifiers', []))}")
        results = pipeline.run_pipeline(data, mode=mode)
        logs.append('分类管道执行完成')

        # 兼容 class_map_path 参数（如有使用，可用于后续保存）
        class_map_path = kwargs.get('class_map_path')
        if class_map_path:
            logs.append(f"参数 class_map_path 已接收: {class_map_path}")
            # 如需自动保存，可在此处添加 pipeline.save_classification_map(results, class_map_path)

        outputs.append(results)
        return TaskResult(status='success', message='分类完成', outputs=outputs, logs=logs)
    except Exception as e:
        err = f"分类模块运行失败: {e}"
        logs.append(err)
        return TaskResult(status='failure', message=str(e), outputs=[], logs=logs)


# ———— 单独测试示例 ————
if __name__ == '__main__':
    # 准备伪数据
    X = np.random.randn(100, 4)
    y = np.random.randint(0, 3, 100)
    data = {'features': X, 'labels': y}
    # 示例 pipeline_config
    pipeline_cfg = {
        'classifiers': [],  # 留空以测试 model 参数
        'compare': False
    }
    # 测试自动创建默认分类器
    class Config: pass
    cfg = Config()
    cfg.classification_params = {
        'model': 'random_forest',
        'class_map_path': 'out/class_map.npy'
    }
    result = run(config=cfg, data=data, pipeline_config=pipeline_cfg, mode='parallel')
    print(result)
    # 测试传入已有 classifiers
    pipeline_cfg2 = {
        'classifiers': [{'name': 'svc', 'params': {'kernel': 'linear'}}],
        'compare': False
    }
    cfg2 = Config()
    cfg2.classification_params = {'model': 'random_forest'}
    result2 = run(config=cfg2, data=data, pipeline_config=pipeline_cfg2, mode='parallel')
    print(result2)

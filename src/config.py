# config.py
#!/usr/bin/env python3.12
# -*- coding: utf-8 -*-
"""
文件: config.py
模块: 根目录
功能: 默认配置示例，用于测试数据跑通全流程
作者: 孟诣楠
版本: v1.0.0
创建时间: 2025-06-19
最近更新: 2025-06-19
较上一版改进:
  - 首次创建
"""
from src.constants import TEST_DATA_DIR, OUTPUT_DIR
import os
import glob

class EngineConfig:
    """配置容器，所有任务参数以子属性形式挂载"""
    pass

# 实例化配置
config = EngineConfig()
# 文件操作任务
config.file_operation_params = {
    "input_dir": TEST_DATA_DIR,
    "output_dir": os.path.join(OUTPUT_DIR, "file_operation")
}
# 文件保存任务
config.file_saver_params = {
    "save_dir": os.path.join(OUTPUT_DIR, "saved_files")
}
# 图像显示任务
config.image_display_params = {
    # 直接读取 file_operation 生成的 .npy 文件列表
    "paths": glob.glob(os.path.join(OUTPUT_DIR, "file_operation", "*.npy")),
    # 要展示的波段（基于 1），默认可写 [1,2,3]
    "bands": [1, 2, 3],
    # 保存 PNG 的目录
    "output_dir": os.path.join(OUTPUT_DIR, "display")
}
# 图像处理任务
config.image_processing_params = {
    "paths": glob.glob(os.path.join(OUTPUT_DIR, "file_operation", "*.npy")),
    "methods": ["equalization", "stretch"],
    "output_dir": os.path.join(OUTPUT_DIR, "processed"),
    "options": {
        "stretch": { "in_range": (2,98), "out_range": (0,255) }
    }
}
# 矢量处理任务
config.vector_processing_params = {
    # 将单文件改成列表
    "input_paths": [
        os.path.join(TEST_DATA_DIR, "sample.shp"),
        # 可以加更多 .shp
    ],
    # 对应批量接口里的 output_dir
    "output_dir": os.path.join(OUTPUT_DIR, "vector"),
    # 要执行的操作列表；如果只是想“加载→保存”，可以留空列表 []
    "operations": [
        # 举例：先重投影到工程 CRS
        {"type": "reproject", "params": {"crs": getattr(config, "target_crs", None)}},
        # 再做缓冲
        # {"type": "buffer",    "params": {"distance": 50}},
        # 再做裁剪
         # {"type": "clip",      "params": {"mask": os.path.join(TEST_DATA_DIR, "mask.shp")}},
    ],
    # 可选的加载/保存参数
    "options": {
        "load": None,
        "save": {"driver": "ESRI Shapefile"}
    }
}
# 分类任务
model = "random_forest"
config.classification_params = {
    # 训练数据输入：features 与 labels 均为 .npy 文件路径
    "data": {
        "features": os.path.join(OUTPUT_DIR, "processed", "features.npy"),
        "labels":   os.path.join(OUTPUT_DIR, "processed", "labels.npy")
    },
    # 管道配置：根据 model 生成 classifiers 列表
    "pipeline_config": {
        "classifiers": [
            {"name": model, "params": {"random_state": 0}}
        ],
        "compare": False
    },
    # 分类结果保存路径
    "class_map_path": os.path.join(OUTPUT_DIR, "processed", "class_map.npy"),
    # 额外标记模型名（可选，run() 中已兼容）
    "model": model
}
# 精度评估任务
config.evaluation_params = {
    "class_map_path": os.path.join(OUTPUT_DIR, "processed", "class_map.npy"),
    "roi_mask_path": os.path.join(TEST_DATA_DIR, "mask.npy"),
    "output_dir": os.path.join(OUTPUT_DIR, "evaluation")
}

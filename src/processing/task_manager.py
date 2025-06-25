# -*- coding: utf-8 -*-
"""
文件: task_manager.py
模块: src.processing.task_manager
功能: 提供任务管理接口，封装 RemoteSensingEngine 以便 GUI 调用
作者: 孟诣楠
版本: v1.1.0
最新更改时间: 2025-06-25
较上一版改进:
  1. 新增监督分类时自动从 .tif 生成内存特征功能（若未显式提供 features）
"""
import logging
from typing import Any, Dict

from src.processing.engine import load_config, RemoteSensingEngine
from src.processing.task_result import TaskResult

from src.utils.image_utils import load_tif_as_numpy

class TaskManager:
    """
    任务管理器，封装遥感处理引擎，提供给 GUI 调用的统一接口。
    """

    def __init__(self, config_path: str = None):
        """
        初始化 TaskManager。

        参数:
            config_path: 可选，配置文件路径，支持 YAML 或 Python 脚本，不指定则使用根目录 config.py。
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        try:
            self.config = load_config(config_path)
            self.engine = RemoteSensingEngine(self.config)
            self.logger.info(f"加载配置成功: {config_path or '根目录 config.py'}")
        except Exception as e:
            self.logger.error(f"加载配置失败: {e}")
            raise

    def run_task(self, task_name: str, params: Dict[str, Any] = None) -> TaskResult:
        """
        运行单个任务。

        参数:
            task_name: 任务名称，需与引擎注册的任务名一致
            params:    任务参数字典

        返回:
            TaskResult
        """
        if params is None:
            params = {}
        self.logger.info(f"开始运行任务: {task_name}，参数: {params}")

        # ✅ 分类任务支持直接从 .tif 加载为特征
        if task_name == "classification":
            data = params.get("data", {})
            features = data.get("features")
            image_path = data.get("image_path") or params.get("image_path")
    
            if not features:
                if image_path and image_path.endswith(".tif"):
                    self.logger.info(f"未提供 features，自动从 {image_path} 加载像素矩阵")
                    try:
                        # 直接从 .tif 转 numpy(H, W, C)
                        features_array = load_tif_as_numpy(image_path)
    
                        # 保存临时 .npy 文件供分类器读取
                        import tempfile, os, numpy as np
                        out_dir = self.config.classification_params.get('output_dir', './')
                        os.makedirs(out_dir, exist_ok=True)
                        tmp_path = tempfile.mktemp(prefix="raw_pixel_", suffix=".npy", dir=out_dir)
                        np.save(tmp_path, features_array)
    
                        data["features"] = tmp_path
                        self.logger.info(f"已生成像素特征文件: {tmp_path}")
                    except Exception as e:
                        self.logger.error(f"从 .tif 加载失败: {e}")
                        raise RuntimeError("无法从 .tif 读取影像像素，分类任务中止。")
                else:
                    raise ValueError("未提供 features，且缺少有效 image_path (.tif)")
    
            params["data"] = data  # 确保回填后的 data 写回 params
        
        result = self.engine.run_task(task_name, **params)
        return result

    def run_all(self) -> Dict[str, TaskResult]:
        """
        运行全流程任务。

        返回:
            各任务名称到 TaskResult 的映射
        """
        self.logger.info("开始运行全流程任务")
        results = self.engine.run()
        return results

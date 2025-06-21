#!/usr/bin/env python3.12
# -*- coding: utf-8 -*-
"""
文件: constants.py
模块: 根目录
功能: 常量定义
作者: 孟诣楠
版本: v1.0.1
创建时间: 2025-06-19
最近更新: 2025-06-20
较上一版改进:
  1. 修改测试数据目录为 data/test_data；
"""
from pathlib import Path

# 项目根目录（src 上一级）
ROOT_DIR = Path(__file__).resolve().parent
BASE_DIR = str(ROOT_DIR)
# 测试数据目录：位于 src/data/test_data
TEST_DATA_DIR = str(ROOT_DIR / "data" / "test_data")
# 默认结果输出目录
OUTPUT_DIR = str(ROOT_DIR / "results")
print(f"测试数据目录: {TEST_DATA_DIR}")
# 其他常量
SUPPORTED_TASKS = [
    "file_operation", "file_saver", "image_display",
    "image_processing", "vector_processing",
    "classification", "evaluation"
]

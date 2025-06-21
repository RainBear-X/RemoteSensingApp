# app.py
#!/usr/bin/env python3.12
# -*- coding: utf-8 -*-
"""
文件: app.py
模块: 根目录
功能: 应用入口，加载默认配置并执行遥感处理全流程或单任务
作者: 孟诣楠
版本: v1.0.0
创建时间: 2025-06-19
最近更新: 2025-06-19
较上一版改进:
  - 首次创建
"""
import sys
from pathlib import Path

if __package__ is None:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
from src.processing.engine import run

if __name__ == "__main__":
    # 直接调用引擎 CLI
    run()
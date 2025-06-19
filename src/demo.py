#!/usr/bin/env python3.12
# -*- coding: utf-8 -*-
"""\
简单示例脚本：加载默认配置并按预定顺序执行全部任务。
运行后会在 src/results 目录下生成处理结果。
"""
from src.processing.engine import load_config, RemoteSensingEngine


def main() -> None:
    config = load_config(None)
    engine = RemoteSensingEngine(config)
    summary = engine.run()
    for name, res in summary.items():
        print(f"{name}: {res.status} - {res.message}")


if __name__ == "__main__":
    main()

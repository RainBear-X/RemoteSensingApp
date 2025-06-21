# -*- coding: utf-8 -*-
"""
文件: run_vector_processing.py
模块: src.processing.vector_processing.run_vector_processing
功能: 批量执行矢量数据处理操作（重投影、缓冲、裁剪等），并保存结果
作者: 孟诣楠
版本: v1.0.1
最近更新: 2025-06-18
较上一版改进:
  - 修复 save_vector_file_as 重复 driver 参数导致保存失败的问题
"""
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import geopandas as gpd
if __package__ is None or __package__ == "":
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "src").is_dir():
            sys.path.insert(0, str(parent))
            break
from src.processing.task_result import TaskResult
from src.processing.file_operations.vector_loader import open_vector_file
from src.processing.file_operations.vector_saver import save_vector_file_as


def run(
        config: Any,
        input_paths: List[str],
        output_dir: str,
        operations: List[Dict[str, Any]],
        options: Optional[Dict[str, Any]] = None
) -> TaskResult:
    """
    批量执行矢量数据处理操作，并保存处理结果。

    参数:
        config: 配置对象，可包含全局目标 CRS 等
        input_paths: 要处理的矢量文件路径列表
        output_dir: 处理结果保存目录
        operations: 操作列表，每项为 dict，示例：
            [
                {"type":"reproject", "params":{"crs":"EPSG:3857"}},
                {"type":"buffer",    "params":{"distance":100.0}},
                {"type":"clip",      "params":{"mask":"mask.shp"}},
            ]
        options: 可选参数字典，可细分为 load/save 等子选项，如
            {"load":{...}, "save":{...}}

    返回:
        TaskResult: 包含 status、message、outputs(保存路径)、logs(日志)
    """
    logs: List[str] = []
    outputs: List[str] = []
    opts = options or {}

    # 创建输出目录
    try:
        os.makedirs(output_dir, exist_ok=True)
        logs.append(f"创建输出目录: {output_dir}")
    except Exception as e:
        msg = f"无法创建输出目录 [{output_dir}]: {e}"
        return TaskResult(status="failure", message=msg, outputs=outputs, logs=[msg])

    # 批量加载与处理
    for fp in input_paths:
        logs.append(f"加载矢量文件: {fp}")
        try:
            gdf = open_vector_file(fp, opts.get("load"))
        except Exception as e:
            err = f"加载失败 [{fp}]: {e}"
            logs.append(err)
            return TaskResult(status="failure", message=str(e), outputs=outputs, logs=logs)

        # 应用操作
        for op in operations:
            typ = op.get("type")
            params = op.get("params", {})
            if typ == "reproject":
                crs = params.get("crs") or getattr(config, "target_crs", None) or gdf.crs
                gdf = gdf.to_crs(crs)
                logs.append(f"重投影至: {crs}")
            elif typ == "buffer":
                dist = params.get("distance", 0)
                gdf["geometry"] = gdf.geometry.buffer(dist)
                logs.append(f"应用缓冲, 距离: {dist}")
            elif typ == "clip":
                mask_fp = params.get("mask")
                logs.append(f"加载裁剪掩膜: {mask_fp}")
                mask_gdf = open_vector_file(mask_fp, opts.get("load"))
                gdf = gdf.clip(mask_gdf)
                logs.append(f"裁剪至掩膜: {mask_fp}")
            else:
                err = f"未知操作类型: {typ}"
                logs.append(err)
                return TaskResult(status="failure", message=err, outputs=outputs, logs=logs)

        # 保存结果
        base = os.path.splitext(os.path.basename(fp))[0]
        out_name = f"{base}_processed.shp"
        out_path = os.path.join(output_dir, out_name)
        try:
            # 避免重复传递 driver 参数
            save_opts = (opts.get("save") or {}).copy()
            save_opts.pop("driver", None)
            save_vector_file_as(gdf, out_path, save_opts)
            logs.append(f"已保存处理文件: {out_path}")
            outputs.append(out_path)
        except Exception as e:
            err = f"保存失败 [{out_path}]: {e}"
            logs.append(err)
            return TaskResult(status="failure", message=str(e), outputs=outputs, logs=logs)

    return TaskResult(status="success", message="矢量处理完成", outputs=outputs, logs=logs)

# ———— 单元测试示例 ————
if __name__ == '__main__':
    class Config:
        target_crs = "EPSG:3857"
    cfg = Config()
    test_files = ["data/roads.shp"]
    ops = [
        {"type":"reproject", "params":{}},
        {"type":"buffer",    "params":{"distance":50}},
    ]
    result = run(
        config=cfg,
        input_paths=test_files,
        output_dir="output/vector",
        operations=ops,
        options={"save":{"driver":"ESRI Shapefile"}}
    )
    print(result)
# ===============================================
# 模块名称：vector_loader.py
# 接口说明：矢量数据加载器
# 作者：9(冒浩溶)
# 版本：v1.0.1
# 功能：支持主流矢量数据格式（如shp, geojson等）的导入
# 创建时间：2025-06-17
# 最后修改时间：2025-06-17
# ===============================================

from __future__ import annotations

import os
from typing import Optional, Dict, Any
import geopandas as gpd


SUPPORTED_VECTOR_FORMATS = ('.shp', '.geojson', '.json', '.gpkg')

class VectorLoaderError(Exception):
    """自定义异常：矢量加载失败"""
    pass

def load_vector(file_path: str, options: Optional[Dict[str, Any]] = None) -> "gpd.GeoDataFrame":
    """
    矢量数据加载主接口，支持多种主流格式。

    Args:
        file_path (str): 矢量文件路径。
        options (dict, 可选): 额外参数（预留）。

    Returns:
        GeoDataFrame: 加载后的矢量数据

    Raises:
        VectorLoaderError: 如果加载失败或格式不支持。
    """
    if not os.path.isfile(file_path):
        raise VectorLoaderError(f"文件不存在: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    if ext not in SUPPORTED_VECTOR_FORMATS:
        raise VectorLoaderError(f"不支持的矢量格式: {ext}")

    if gpd is None:
        raise VectorLoaderError("geopandas 未安装，请先安装 geopandas 库。")

    try:
        gdf = gpd.read_file(file_path)
        return gdf
    except Exception as e:
        raise VectorLoaderError(f"矢量数据加载失败: {str(e)}")

# ========== 预留接口示例 ==========
def open_vector_file(file_path: str, options: Optional[Dict[str, Any]] = None):
    """
    对外接口，加载矢量数据并返回 GeoDataFrame
    """
    return load_vector(file_path, options)

# ========== 独立测试 ==========
if __name__ == "__main__":
    test_path = "340000.shp"
    try:
        gdf = open_vector_file(test_path)
        print("矢量数据加载成功！条目数：", len(gdf))
        print(gdf.head())
    except Exception as e:
        print("测试失败:", e)
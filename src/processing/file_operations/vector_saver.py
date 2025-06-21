# ===============================================
# 模块名称：vector_saver.py
# 接口说明：矢量数据保存器
# 作者：9(冒浩溶)
# 版本：v1.0.0
# 功能：支持矢量数据保存与格式转换，可配置参数
# 创建时间：2025-06-17
# 最后修改时间：2025-06-17
# ===============================================

import os
from typing import Any, Optional, Dict

try:
    import geopandas as gpd
except ImportError:
    gpd = None

SUPPORTED_SAVE_FORMATS = ('.shp', '.geojson', '.json', '.gpkg')

class VectorSaverError(Exception):
    """自定义异常：矢量保存失败"""
    pass

def save_vector(gdf: Any, save_path: str, options: Optional[Dict] = None) -> None:
    """
    矢量数据保存主接口，支持格式转换与参数配置。

    Args:
        gdf (GeoDataFrame): 矢量数据。
        save_path (str): 保存路径。
        options (dict, 可选): 保存参数。

    Raises:
        VectorSaverError: 保存失败时抛出。
    """
    ext = os.path.splitext(save_path)[1].lower()
    if ext not in SUPPORTED_SAVE_FORMATS:
        raise VectorSaverError(f"不支持的保存格式: {ext}")

    if gpd is None:
        raise VectorSaverError("geopandas 未安装，请先安装 geopandas 库。")

    try:
        gdf.to_file(save_path, driver=None, **(options or {}))
    except Exception as e:
        raise VectorSaverError(f"矢量保存失败: {str(e)}")

# ========== 预留接口示例 ==========
def save_vector_file_as(gdf: Any, save_path: str, options: Optional[Dict] = None):
    """
    对外接口，保存矢量数据到指定路径
    """
    return save_vector(gdf, save_path, options)

# ========== 独立测试 ==========
if __name__ == "__main__":
    import geopandas as gpd
    from shapely.geometry import Point

    # 构建测试数据
    gdf = gpd.GeoDataFrame({'id': [1, 2]}, geometry=[Point(0, 0), Point(1, 1)], crs="EPSG:4326")
    test_save_path = "3400000.shp"
    try:
        save_vector_file_as(gdf, test_save_path)
        print("矢量数据保存成功:", test_save_path)
    except Exception as e:
        print("测试失败:", e)
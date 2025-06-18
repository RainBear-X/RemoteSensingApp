# 文件: src/processing/vector_processing/roi_saver.py
# 作者: 徐新宇、孟诣楠
# 版本: 1.0.0

# 最新更改时间: 2025-06-18
# 功能: 将 ROI 几何对象保存为矢量文件 (Shapefile/GeoJSON 等)

import geopandas as gpd
import os


def save_roi_to_file(roi_geom, filepath, crs="EPSG:4326", driver=None):
    """
    保存 ROI 几何对象到矢量文件

    参数:
        roi_geom (shapely.geometry.Polygon): ROI 多边形
        filepath (str): 输出文件路径，应带后缀 .shp/.geojson
        crs (str): 坐标参考系，默认 WGS84
        driver (str): 可选指定驱动，如 "ESRI Shapefile" 或 "GeoJSON"
    """
    if roi_geom is None:
        raise ValueError("roi_geom 不能为空")
    if not filepath:
        raise ValueError("filepath 不能为空")

    # 构建 GeoDataFrame
    gdf = gpd.GeoDataFrame({'geometry': [roi_geom]}, crs=crs)

    # 根据后缀推断驱动
    ext = os.path.splitext(filepath)[1].lower()
    if driver is None:
        if ext == '.geojson':
            driver = 'GeoJSON'
        elif ext == '.shp':
            driver = 'ESRI Shapefile'
        else:
            raise ValueError("无法根据文件后缀推断格式，请指定 driver")

    # 创建输出目录
    out_dir = os.path.dirname(filepath)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # 保存文件
    gdf.to_file(filepath, driver=driver)
    print(f"✅ ROI 已保存: {filepath}")

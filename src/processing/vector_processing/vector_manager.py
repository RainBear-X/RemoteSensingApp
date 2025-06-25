# 文件: src/processing/vector_processing/vector_manager.py
# 作者: 徐新宇、孟诣楠
# 版本: 1.0.0

# 最新更改时间: 2025-06-18
# 功能: 矢量数据管理，包括加载、保存、查询和新增要素

import geopandas as gpd
import pandas as pd


def load_vector(filepath):
    """
    读取矢量文件，支持 shapefile、GeoJSON 等

    参数:
        filepath (str): 矢量文件路径
    返回:
        GeoDataFrame: 包含要素的 GeoDataFrame
    """
    gdf = gpd.read_file(filepath)
    print(f"✅ 矢量数据已加载: {filepath}, 要素数: {len(gdf)}")
    return gdf


def save_vector(gdf, filepath, driver=None):
    """
    将 GeoDataFrame 保存为矢量文件

    参数:
        gdf (GeoDataFrame): 输入要素
        filepath (str): 输出文件路径
        driver (str): 可选指定驱动
    """
    if driver:
        gdf.to_file(filepath, driver=driver)
    else:
        gdf.to_file(filepath)
    print(f"✅ 矢量数据已保存: {filepath}")


def query_features(gdf, query_expr):
    """
    根据表达式查询要素

    参数:
        gdf (GeoDataFrame): 输入要素
        query_expr (str): Pandas 查询表达式
    返回:
        GeoDataFrame: 查询结果
    """
    result = gdf.query(query_expr)
    print(f"查询完成，匹配要素数: {len(result)}")
    return result


def add_feature(gdf, geom, properties=None):
    """
    向 GeoDataFrame 添加一个要素

    参数:
        gdf (GeoDataFrame): 原始要素
        geom (shapely.geometry.base.BaseGeometry): 要添加的几何对象
        properties (dict): 属性字典
    返回:
        GeoDataFrame: 包含新增要素的 GeoDataFrame
    """
    props = properties or {}
    new = gpd.GeoDataFrame([props], geometry=[geom], crs=gdf.crs)
    updated = pd.concat([gdf, new], ignore_index=True)
    print(f"✅ 添加要素: {geom}, 共 {len(updated)} 个要素")
    return updated
# 文件: src/processing/vector_processing/vector_creation/point_creator.py
# 作者: 徐新宇、孟诣楠
# 版本: 1.0.0

# 最新更改时间: 2025-06-18
# 功能: 创建点要素

from shapely.geometry import Point


def create_point_feature(x, y):
    """
    创建点要素

    参数:
        x (float): X 坐标
        y (float): Y 坐标
    返回:
        Point: Shapely 点对象
    """
    pt = Point(x, y)
    return pt
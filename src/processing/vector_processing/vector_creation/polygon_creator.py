# 文件: src/processing/vector_processing/vector_creation/polygon_creator.py
# 作者: 徐新宇、孟诣楠
# 版本: 1.0.0

# 最新更改时间: 2025-06-18
# 功能: 创建面多边形要素

from shapely.geometry import Polygon


def create_polygon_feature(coords):
    """
    创建面要素

    参数:
        coords (List[Tuple[float,float]]): 点列表
    返回:
        Polygon: Shapely 多边形对象
    """
    if len(coords) < 3:
        raise ValueError("多边形至少需要三个点")
    poly = Polygon(coords)
    return poly
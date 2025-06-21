# 文件: src/processing/vector_processing/vector_creation/polyline_creator.py
# 作者: 徐新宇、孟诣楠
# 版本: 1.0.0

# 最新更改时间: 2025-06-18
# 功能: 创建折线要素

from shapely.geometry import LineString


def create_polyline_feature(coords):
    """
    创建折线要素

    参数:
        coords (List[Tuple[float,float]]): 点列表
    返回:
        LineString: Shapely 折线对象
    """
    if len(coords) < 2:
        raise ValueError("折线至少需要两个点")
    line = LineString(coords)
    return line

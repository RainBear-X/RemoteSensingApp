# 文件: src/processing/vector_processing/roi_creator.py
# 作者: 徐新宇、孟诣楠
# 版本: 1.0.0

# 最新更改时间: 2025-06-18
# 功能: 根据用户提供的坐标列表创建 ROI 多边形对象

from shapely.geometry import Polygon


def create_roi_polygon(coords):
    """
    根据用户点击的坐标列表创建 ROI 多边形

    参数:
        coords (List[Tuple[float,float]]): 点列表 [(x1,y1), (x2,y2), ...]
    返回:
        Polygon: Shapely 多边形对象
    """
    if not isinstance(coords, (list, tuple)):
        raise TypeError("coords 必须是列表或元组")
    if len(coords) < 3:
        raise ValueError("至少需要三个点来创建多边形 ROI")
    poly = Polygon(coords)
    return poly


if __name__ == "__main__":
    # 测试示例
    pts = [(0, 0), (10, 0), (10, 10), (0, 10)]
    roi = create_roi_polygon(pts)
    print(f"创建的 ROI: {roi}")

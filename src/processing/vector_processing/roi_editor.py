# 文件: src/processing/vector_processing/roi_editor.py
# 作者: 徐新宇、孟诣楠
# 版本: 1.0.0

# 最新更改时间: 2025-06-18
# 功能: 编辑已有 ROI 的几何形状

from shapely.geometry import Polygon


def edit_roi_polygon(existing_roi, new_coords):
    """
    更新 ROI 的几何形状

    参数:
        existing_roi (shapely.geometry.Polygon): 原 ROI
        new_coords (List[Tuple[float,float]]): 新坐标点列表
    返回:
        Polygon: 更新后的 ROI 多边形
    """
    if len(new_coords) < 3:
        raise ValueError("多边形至少需要三个点")
    updated = Polygon(new_coords)
    return updated


if __name__ == "__main__":
    # 测试：将正方形编辑为三角形
    orig = Polygon([(0,0),(1,0),(1,1),(0,1)])
    new = [(0,0),(2,0),(1,2)]
    roi2 = edit_roi_polygon(orig, new)
    print(f"更新后的 ROI: {roi2}")



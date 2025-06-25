# ===============================================
# 模块名称：projection.py
# 接口说明：遥感影像投影转换模块
# 作者：YangQC
# 版本：v1.1.0
# 功能：支持遥感影像投影变换，自动检测PROJ环境
# 创建时间：2025-06-17
# 最后修改时间：2025-06-17
# ===============================================

import os
import sys
from osgeo import gdal, osr

gdal.UseExceptions()
osr.UseExceptions()

def set_proj_lib():
    """
    自动检测当前Python环境下的proj库路径，并设置PROJ_LIB，避免外部GDAL/PROJ库冲突。
    """
    possible_paths = []
    # Conda环境
    conda_proj = os.path.join(sys.prefix, "Library", "share", "proj")
    possible_paths.append(conda_proj)
    # pip安装osgeo
    pip_proj = os.path.join(sys.prefix, "Lib", "site-packages", "osgeo", "data", "proj")
    possible_paths.append(pip_proj)
    # 其它常见路径可按需添加

    for path in possible_paths:
        if os.path.isdir(path) and os.path.isfile(os.path.join(path, "proj.db")):
            os.environ["PROJ_LIB"] = path
            print(f"Set PROJ_LIB to: {path}")
            return
    print("Warning: Could not automatically set PROJ_LIB，投影转换可能失败。")

set_proj_lib()

class ProjectionTransformer:
    """投影变换器，支持自动判断地理参考"""
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.dataset = gdal.Open(filepath)
        if self.dataset is None:
            raise FileNotFoundError(f"无法打开文件: {filepath}")

        # 鲁棒性检测：地理参考判断
        gt = self.dataset.GetGeoTransform()
        has_gcp = bool(self.dataset.GetGCPs())
        if (gt is None or gt == (0, 1, 0, 0, 0, 1)) and not has_gcp:
            raise ValueError(
                f"影像 {filepath} 没有地理参考，无法投影变换！\n"
                f"请先用ENVI、QGIS或gdal_translate等工具赋予地理参考信息。"
            )

    def reproject(self, dst_path: str, dst_srs_wkt: str, resample_method=gdal.GRA_NearestNeighbour) -> str:
        """
        影像投影转换，输出到新文件
        :param dst_path: 输出文件路径
        :param dst_srs_wkt: 目标投影WKT
        :param resample_method: 重采样方法
        :return: 新文件路径
        """
        warp_options = gdal.WarpOptions(dstSRS=dst_srs_wkt, resampleAlg=resample_method)
        out_ds = gdal.Warp(dst_path, self.dataset, options=warp_options)
        if out_ds is None:
            raise RuntimeError(f"投影转换失败: {dst_path}")
        out_ds = None
        return dst_path

    def close(self):
        self.dataset = None

# ===============================================
# 预留接口（供系统UI调用）
# ===============================================
def reproject_image(filepath: str, dst_path: str, dst_srs_wkt: str, resample_method=gdal.GRA_NearestNeighbour) -> str:
    """
    外部接口：影像投影转换
    """
    transformer = ProjectionTransformer(filepath)
    try:
        out = transformer.reproject(dst_path, dst_srs_wkt, resample_method)
    finally:
        transformer.close()
    return out

# ===============================================
# 单元测试
# ===============================================
if __name__ == "__main__":
    test_file = "example.tif"
    test_out = "reprojected.tif"
    # WGS84投影WKT
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    dst_wkt = srs.ExportToWkt()
    try:
        out_path = reproject_image(test_file, test_out, dst_wkt)
        print(f"投影转换输出: {out_path}")
    except Exception as e:
        print(f"投影转换失败: {e}")
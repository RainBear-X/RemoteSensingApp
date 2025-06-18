# ===============================================
# 模块名称：image_cutting.py
# 接口说明：遥感影像裁剪模块
# 作者：YangQC
# 版本：v1.2.0
# 功能：支持像素窗口、地理/投影坐标、矢量/ROI裁剪
# 创建时间：2025-06-17
# 最后修改时间：2025-06-17
# ===============================================

import os
from osgeo import gdal, ogr, osr
import numpy as np
from typing import Tuple, Optional, Union

gdal.UseExceptions()
osr.UseExceptions()

class ImageCutter:
    """
    影像裁剪器，支持像素窗口、地理/投影坐标、矢量/ROI掩模裁剪
    任意方法均可单独使用，无需全部实现
    """
    def __init__(self, filepath: str):
        """
        初始化，打开影像文件，准备后续操作
        :param filepath: 影像文件路径
        """
        self.filepath = filepath
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"影像文件不存在: {filepath}")
        self.dataset = gdal.Open(filepath)
        if self.dataset is None:
            raise FileNotFoundError(f"无法打开文件: {filepath}")
        self.xsize = self.dataset.RasterXSize
        self.ysize = self.dataset.RasterYSize

    def cut_by_window(self, xoff: int, yoff: int, xsize: int, ysize: int) -> np.ndarray:
        """
        按像素窗口裁剪
        :param xoff: 左上角X像素索引（从0开始）
        :param yoff: 左上角Y像素索引（从0开始）
        :param xsize: 裁剪窗口宽度（像素数）
        :param ysize: 裁剪窗口高度（像素数）
        :return: 裁剪后的影像数据（numpy数组，单波段为2D，多波段为3D）
        """
        if xsize <= 0 or ysize <= 0:
            raise ValueError(f"裁剪窗口尺寸必须为正，当前xsize={xsize}，ysize={ysize}")
        # 边界检查，防止越界
        xoff = max(0, xoff)
        yoff = max(0, yoff)
        xsize = min(xsize, self.xsize - xoff)
        ysize = min(ysize, self.ysize - yoff)
        if xsize <= 0 or ysize <= 0:
            raise ValueError(f"像素窗口裁剪越界(xoff={xoff}, yoff={yoff}, xsize={xsize}, ysize={ysize})")
        bands = self.dataset.RasterCount
        result = []
        for i in range(1, bands + 1):
            band = self.dataset.GetRasterBand(i)
            arr = band.ReadAsArray(xoff, yoff, xsize, ysize)
            result.append(arr)
        if bands == 1:
            return result[0]
        return np.stack(result, axis=-1)

    def geo_to_pixel(self, x: float, y: float) -> Tuple[int, int]:
        """
        地理/投影坐标转像素行列号
        :param x: X坐标（经度或投影坐标，不是像素）
        :param y: Y坐标（纬度或投影坐标，不是像素）
        :return: (col, row) 像素索引
        """
        gt = self.dataset.GetGeoTransform()
        px = int(round((x - gt[0]) / gt[1]))
        py = int(round((y - gt[3]) / gt[5]))
        return px, py

    def cut_by_geo(self, x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
        """
        按地理/投影坐标裁剪，自动排序，边界检查
        :param x1, y1: 一角地理/投影坐标
        :param x2, y2: 对角地理/投影坐标（顺序任意，自动纠正）
        :return: 裁剪后的影像数据
        """
        ulx, lrx = sorted([x1, x2])
        uly, lry = sorted([y1, y2], reverse=True)  # uly > lry（以左上-右下为主）
        # 转为像素窗口
        px0, py0 = self.geo_to_pixel(ulx, uly)
        px1, py1 = self.geo_to_pixel(lrx, lry)
        xoff = min(px0, px1)
        yoff = min(py0, py1)
        xsize = abs(px1 - px0)
        ysize = abs(py1 - py0)
        # 边界检查，防止越界
        if xoff < 0 or yoff < 0 or xoff >= self.xsize or yoff >= self.ysize:
            raise ValueError(f"地理裁剪超出影像范围(xoff={xoff}, yoff={yoff})")
        xsize = min(xsize, self.xsize - xoff)
        ysize = min(ysize, self.ysize - yoff)
        if xsize <= 0 or ysize <= 0:
            raise ValueError(f"地理坐标裁剪计算得窗口尺寸非法(xsize={xsize}, ysize={ysize})，请检查输入坐标顺序和范围！")
        return self.cut_by_window(xoff, yoff, xsize, ysize)

    def cut_by_vector(self, vector_path: str, out_path: Optional[str] = None, nodata: Union[int, float] = 0) -> Optional[str]:
        """
        按矢量范围（shapefile/geojson等）掩模裁剪，输出为新文件
        :param vector_path: 矢量文件路径（如shp、geojson等，需带全路径）
        :param out_path: 输出文件路径（如不指定则自动生成与输入同目录的 _clip 后缀文件）
        :param nodata: 掩模外像元填充值（默认为0）
        :return: 输出文件路径
        """
        if not os.path.exists(vector_path):
            raise FileNotFoundError(f"矢量文件不存在: {vector_path}")
        if out_path is None:
            base, ext = os.path.splitext(self.filepath)
            out_path = f"{base}_clip{ext}"
        # 尝试打开矢量文件（支持shp/geojson等），提前报错更友好
        try:
            # 只检查文件可读性，不做数据读取
            ogr_ds = ogr.Open(vector_path)
            if ogr_ds is None:
                raise RuntimeError("无法打开矢量文件，驱动不支持或文件损坏")
        except Exception as e:
            raise RuntimeError(f"矢量文件打开失败: {e}")

        # 使用GDAL.Warp执行掩模裁剪
        try:
            gdal.Warp(
                out_path,
                self.dataset,
                cutlineDSName=vector_path,
                cropToCutline=True,
                dstNodata=nodata,
                multithread=True
            )
        except Exception as e:
            raise RuntimeError(f"GDAL矢量掩模裁剪失败: {e}")
        return out_path

    def close(self):
        """
        释放数据集资源
        """
        self.dataset = None

# ===============================================
# 预留接口（供系统UI或外部调用）
# ===============================================
def cut_image(filepath: str, xoff: int, yoff: int, xsize: int, ysize: int) -> np.ndarray:
    """
    外部接口：像素窗口裁剪
    :return: 裁剪后numpy数组
    """
    cutter = ImageCutter(filepath)
    try:
        result = cutter.cut_by_window(xoff, yoff, xsize, ysize)
    finally:
        cutter.close()
    return result

def cut_image_by_geo(filepath: str, x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    """
    外部接口：地理/投影坐标裁剪
    :return: 裁剪后numpy数组
    """
    cutter = ImageCutter(filepath)
    try:
        result = cutter.cut_by_geo(x1, y1, x2, y2)
    finally:
        cutter.close()
    return result

def cut_image_by_vector(filepath: str, vector_path: str, out_path: Optional[str] = None, nodata: Union[int, float] = 0) -> Optional[str]:
    """
    外部接口：矢量/ROI掩模裁剪
    :return: 输出文件路径
    """
    cutter = ImageCutter(filepath)
    try:
        result = cutter.cut_by_vector(vector_path, out_path, nodata)
    finally:
        cutter.close()
    return result

# ===============================================
# 单元测试
# ===============================================
if __name__ == "__main__":
    test_file = "sample.tif"
    # 测试像素窗口裁剪
    try:
        cut = cut_image(test_file, 0, 0, 100, 100)
        print(f"像素窗口裁剪shape: {cut.shape}")
    except Exception as e:
        print(f"像素窗口裁剪失败: {e}")
    # 测试地理/投影坐标裁剪
    try:
        cut_geo = cut_image_by_geo(test_file, 457024.15625, 4332804.5, 557024.15625, 4232804.5)
        print(f"地理坐标裁剪shape: {cut_geo.shape}")
    except Exception as e:
        print(f"地理坐标裁剪失败: {e}")
    # 测试矢量掩模裁剪
    try:
        out_file = cut_image_by_vector(test_file, "bound.shp", "clip_out.tif")
        print(f"矢量裁剪输出: {out_file}")
    except Exception as e:
        print(f"矢量裁剪失败: {e}")
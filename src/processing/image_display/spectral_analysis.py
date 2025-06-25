# ===============================================
# 模块名称：spectral_analysis.py
# 接口说明：波谱特征分析模块
# 作者：YangQC
# 版本：v1.2.0
# 功能：对指定像元、ROI或矢量区域输出各波段的波谱值/统计信息，并增强鲁棒性
# 创建时间：2025-06-17
# 最后修改时间：2025-06-17
# ===============================================

from osgeo import gdal, ogr, osr
import numpy as np
from typing import Tuple, List, Dict, Optional, Union
import os

gdal.UseExceptions()
osr.UseExceptions()

class SpectralAnalyzer:
    """
    波谱特征分析器
    支持获取：
      - 单像元多波段光谱
      - ROI/矢量掩模区域的光谱均值/极值
    """

    def __init__(self, filepath: str):
        self.filepath = os.path.abspath(filepath)
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"影像文件不存在: {filepath}")
        self.dataset = gdal.Open(self.filepath)
        if self.dataset is None:
            raise FileNotFoundError(f"无法打开文件: {filepath}")
        self.xsize = self.dataset.RasterXSize
        self.ysize = self.dataset.RasterYSize

    def get_nodata(self, band) -> Optional[float]:
        """
        获取波段的NoData值，如果设置了
        """
        nodata = band.GetNoDataValue()
        # 某些格式可能返回None
        return nodata

    def get_spectrum(self, row: int, col: int) -> Dict[int, Union[float, str]]:
        """
        获取某像元的多波段波谱值，自动处理NoData
        :param row: 行号（从0开始）
        :param col: 列号（从0开始）
        :return: {band_id: value 或 'NoData'}
        """
        if not (0 <= row < self.ysize and 0 <= col < self.xsize):
            raise IndexError(f"像元位置越界 (row={row}, col={col})，影像尺寸=({self.ysize},{self.xsize})")
        bands = self.dataset.RasterCount
        result = {}
        for i in range(1, bands + 1):
            band = self.dataset.GetRasterBand(i)
            arr = band.ReadAsArray(col, row, 1, 1)
            val = float(arr[0, 0])
            nodata = self.get_nodata(band)
            # 如果NoData设置且相等，或检查常见空值
            if nodata is not None and np.isclose(val, nodata):
                result[i] = 'NoData'
            elif np.isclose(val, -3.4028235e+38):  # 常见GDAL空值
                result[i] = 'NoData'
            else:
                result[i] = val
        return result

    def get_roi_statistics(self, vector_path: str, stat: str = "mean") -> Dict[int, Union[float, str]]:
        """
        在矢量区域（shp/geojson）内统计各波段光谱信息
        :param vector_path: ROI/矢量文件路径
        :param stat: 统计方式，支持'mean', 'min', 'max'
        :return: {band_id: 统计值 或 'NoData'}
        """
        if not os.path.exists(vector_path):
            raise FileNotFoundError(f"矢量文件不存在: {vector_path}")

        # 检查矢量和影像空间参考的一致性，给出警告
        try:
            ds = ogr.Open(vector_path)
            layer = ds.GetLayer()
            vector_srs = layer.GetSpatialRef()
            raster_srs_wkt = self.dataset.GetProjection()
            raster_srs = osr.SpatialReference()
            raster_srs.ImportFromWkt(raster_srs_wkt)
            if vector_srs is not None and raster_srs_wkt:
                if not vector_srs.IsSame(raster_srs):
                    print("Warning: ROI/矢量数据的空间参考与栅格影像不一致，掩模结果可能不准确。")
        except Exception as e:
            print(f"Warning: 无法读取矢量文件空间参考信息：{e}")

        # 将矢量转为掩模
        try:
            mask_ds = gdal.Warp(
                '', self.dataset,
                format='MEM',
                cutlineDSName=vector_path,
                cropToCutline=False,
                dstNodata=0,
                outputType=gdal.GDT_Byte
            )
        except Exception as e:
            raise RuntimeError(f"GDAL掩模生成失败: {e}")

        mask_band = mask_ds.GetRasterBand(1).ReadAsArray()  # 掩模：掩模区为1，外为0
        bands = self.dataset.RasterCount
        result = {}
        for i in range(1, bands + 1):
            arr = self.dataset.GetRasterBand(i).ReadAsArray()
            nodata = self.get_nodata(self.dataset.GetRasterBand(i))
            arr_masked = arr[mask_band != 0]
            # 剔除NoData
            if nodata is not None:
                arr_masked = arr_masked[arr_masked != nodata]
            arr_masked = arr_masked[arr_masked != -3.4028235e+38]
            if arr_masked.size == 0:
                result[i] = 'NoData'
            elif stat == "mean":
                result[i] = float(np.mean(arr_masked))
            elif stat == "min":
                result[i] = float(np.min(arr_masked))
            elif stat == "max":
                result[i] = float(np.max(arr_masked))
            else:
                raise ValueError(f"不支持的统计方式: {stat}")
        return result

    def close(self):
        self.dataset = None

# ===============================================
# 预留接口（供系统UI调用）
# ===============================================
def pixel_spectrum(filepath: str, row: int, col: int) -> Dict[int, Union[float, str]]:
    """
    外部接口：获取像元波谱
    :param filepath: 影像主文件
    :param row: 行号
    :param col: 列号
    :return: {band_id: value 或 'NoData'}
    """
    analyzer = SpectralAnalyzer(filepath)
    try:
        result = analyzer.get_spectrum(row, col)
    finally:
        analyzer.close()
    return result

def roi_spectrum(filepath: str, vector_path: str, stat: str = "mean") -> Dict[int, Union[float, str]]:
    """
    外部接口：获取ROI/矢量区内波谱统计
    :param filepath: 影像主文件
    :param vector_path: ROI/矢量文件路径
    :param stat: 统计方式（mean/min/max）
    :return: {band_id: 统计值 或 'NoData'}
    """
    analyzer = SpectralAnalyzer(filepath)
    try:
        result = analyzer.get_roi_statistics(vector_path, stat)
    finally:
        analyzer.close()
    return result

# ===============================================
# 单元测试
# ===============================================
if __name__ == "__main__":
    test_file = "AA"
    test_row, test_col = 100, 100
    test_shp = "bound.shp"
    # 测试单像元光谱
    try:
        spectrum = pixel_spectrum(test_file, test_row, test_col)
        print(f"像元({test_row},{test_col})光谱: {spectrum}")
    except Exception as e:
        print(f"波谱分析失败: {e}")
    # 测试ROI/矢量区光谱均值
    try:
        spectrum_roi = roi_spectrum(test_file, test_shp, "mean")
        print(f"ROI区域波谱均值: {spectrum_roi}")
    except Exception as e:
        print(f"ROI波谱分析失败: {e}")
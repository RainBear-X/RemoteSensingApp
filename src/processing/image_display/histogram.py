# ===============================================
# 模块名称：histogram.py
# 接口说明：波段直方图统计模块
# 作者：YangQC
# 版本：v1.0.2
# 功能：支持单波段/多波段直方图统计与返回
# 创建时间：2025-06-17
# 最后修改时间：2025-06-17
# ===============================================

from osgeo import gdal
import numpy as np
from typing import Union, List, Dict
from osgeo import gdal, osr

gdal.UseExceptions()
osr.UseExceptions()

class HistogramAnalyzer:
    """
    波段直方图统计器，支持单波段和多波段
    """
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.dataset = gdal.Open(filepath)
        # 增加鲁棒性，判断是否为有效栅格影像
        if self.dataset is None or self.dataset.RasterCount == 0:
            raise ValueError(f"文件格式不受支持或不是有效的栅格影像: {filepath}")

    def histogram(self, bands: Union[int, List[int]], bins: int = 256) -> Dict[int, np.ndarray]:
        """
        统计指定波段的直方图
        :param bands: 波段编号（1开始）或编号列表
        :param bins: 直方图分bin数
        :return: {band_id: hist}
        """
        if isinstance(bands, int):
            bands = [bands]
        result = {}
        for b in bands:
            if b < 1 or b > self.dataset.RasterCount:
                raise ValueError(f"波段编号{b}超出范围(1-{self.dataset.RasterCount})")
            band = self.dataset.GetRasterBand(b)
            arr = band.ReadAsArray()
            hist, _ = np.histogram(arr, bins=bins, range=(arr.min(), arr.max()))
            result[b] = hist
        return result

    def close(self):
        self.dataset = None

# ===============================================
# 预留接口（供系统UI调用）
# ===============================================
def band_histogram(filepath: str, bands: Union[int, List[int]], bins: int = 256) -> Dict[int, np.ndarray]:
    """
    外部接口：波段直方图统计
    """
    analyzer = HistogramAnalyzer(filepath)
    try:
        result = analyzer.histogram(bands, bins)
    finally:
        analyzer.close()
    return result

# ===============================================
# 单元测试
# ===============================================
if __name__ == "__main__":
    test_file = "AA"
    try:
        ds = gdal.Open(test_file)
        if ds is not None and ds.RasterCount > 0:
            test_bands = list(range(1, ds.RasterCount + 1))
            ds = None
            try:
                hists = band_histogram(test_file, test_bands)
                for b, hist in hists.items():
                    print(f"Band {b} 直方图: {hist[:10]} ...")
            except Exception as e:
                print(f"直方图测试失败: {e}")
        else:
            print("测试文件不是有效的栅格影像或格式不受支持！")
    except Exception as e:
        print(f"文件格式不受支持或不是有效的栅格影像: {e}")
# ===============================================
# 模块名称：band_extraction.py
# 接口说明：多光谱影像波段提取模块
# 作者：YangQC
# 版本：v1.0.2
# 功能：支持多光谱影像的单波段或多波段组合提取，适配遥感图像显示系统
# 创建时间：2025-06-17
# 最后修改时间：2025-06-17
# ===============================================

import numpy as np
from typing import List, Union
from osgeo import gdal, osr

gdal.UseExceptions()
osr.UseExceptions()

class BandExtractor:
    """
    波段提取器，支持单波段和多波段组合提取
    """
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.dataset = gdal.Open(filepath)
        # 增加鲁棒性，判断是否为有效栅格影像
        if self.dataset is None or self.dataset.RasterCount == 0:
            raise ValueError(f"文件格式不受支持或不是有效的栅格影像: {filepath}")
        self.band_count = self.dataset.RasterCount

    def extract(self, bands: Union[int, List[int]]) -> np.ndarray:
        """
        提取指定波段，支持单波段与多波段组合
        :param bands: int 或 list(int), 需要提取的波段编号（从1开始）
        :return: np.ndarray, shape=(H, W) 或 (H, W, N)
        """
        if isinstance(bands, int):
            bands = [bands]
        result = []
        for b in bands:
            if b < 1 or b > self.band_count:
                raise ValueError(f"波段编号{b}超出范围(1-{self.band_count})")
            band = self.dataset.GetRasterBand(b)
            arr = band.ReadAsArray()
            result.append(arr)
        if len(result) == 1:
            return result[0]
        return np.stack(result, axis=-1)

    def close(self):
        """释放GDAL资源"""
        self.dataset = None

# ===============================================
# 预留接口（供系统UI调用）
# ===============================================
def extract_band(filepath: str, bands: Union[int, List[int]]) -> np.ndarray:
    """
    外部接口：提取指定波段
    :param filepath: 文件路径
    :param bands: 波段编号或编号列表
    :return: 波段数组
    """
    extractor = BandExtractor(filepath)
    try:
        result = extractor.extract(bands)
    finally:
        extractor.close()
    return result

# ===============================================
# 单元测试
# ===============================================
if __name__ == "__main__":
    test_file = "AA"
    try:
        # 自动检测波段数并提取
        ds = gdal.Open(test_file)
        if ds is not None and ds.RasterCount > 0:
            test_bands = list(range(1, ds.RasterCount + 1))
            ds = None
            try:
                bands_data = extract_band(test_file, test_bands)
                print(f"全部波段提取shape: {bands_data.shape}")
            except Exception as e:
                print(f"波段提取测试失败: {e}")
        else:
            print("测试文件不是有效的栅格影像或格式不受支持！")
    except Exception as e:
        print(f"文件格式不受支持或不是有效的栅格影像: {e}")
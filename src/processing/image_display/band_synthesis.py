# ===============================================
# 模块名称：band_synthesis.py
# 接口说明：波段合成模块
# 作者：YangQC
# 版本：v1.0.0
# 功能：支持真彩色、假彩色等多种波段合成方案
# 创建时间：2025-06-17
# 最后修改时间：2025-06-17
# ===============================================

import numpy as np
import sys
from pathlib import Path
if __package__ is None or __package__ == "":
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "src").is_dir():
            sys.path.insert(0, str(parent))
            break
from typing import List, Tuple
from osgeo import gdal

class BandSynthesis:
    """
    波段合成器，支持真彩色、假彩色等波段合成方案
    """
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.dataset = gdal.Open(filepath)
        if self.dataset is None:
            raise FileNotFoundError(f"无法打开文件: {filepath}")
        self.band_count = self.dataset.RasterCount

    def synthesize(self, band_indices: Tuple[int, int, int]) -> np.ndarray:
        """
        根据输入的波段索引合成三通道影像
        :param band_indices: (R, G, B) 波段编号(从1开始)
        :return: 合成后的3通道影像 (H, W, 3)
        """
        if any(b < 1 or b > self.band_count for b in band_indices):
            raise ValueError(f"波段编号超出范围(1-{self.band_count})")
        img = []
        for b in band_indices:
            band = self.dataset.GetRasterBand(b)
            arr = band.ReadAsArray()
            img.append(arr)
        img = np.stack(img, axis=-1)
        # 归一化到0-255 uint8
        img = self._normalize_to_uint8(img)
        return img

    def _normalize_to_uint8(self, arr: np.ndarray) -> np.ndarray:
        arr_min = arr.min(axis=(0,1), keepdims=True)
        arr_max = arr.max(axis=(0,1), keepdims=True)
        arr = (arr - arr_min) / (arr_max - arr_min + 1e-8) * 255
        return arr.astype(np.uint8)

    def close(self):
        self.dataset = None

# ===============================================
# 预留接口（供系统UI调用）
# ===============================================
def synthesize_band(filepath: str, rgb_bands: Tuple[int, int, int]) -> np.ndarray:
    """
    外部接口：波段合成
    :param filepath: 文件路径
    :param rgb_bands: 三元组，指定R,G,B波段编号
    :return: 合成后的3通道影像
    """
    synthesizer = BandSynthesis(filepath)
    try:
        img = synthesizer.synthesize(rgb_bands)
    finally:
        synthesizer.close()
    return img

# ===============================================
# 单元测试
# ===============================================
if __name__ == "__main__":
    test_file = "AA.tif"
    # 真彩色例：Landsat8为(4,3,2)，假彩色例：(5,4,3)
    test_rgb = (4, 3, 2)
    try:
        synth_img = synthesize_band(test_file, test_rgb)
        print(f"合成图像shape: {synth_img.shape}, dtype: {synth_img.dtype}")
    except Exception as e:
        print(f"测试失败: {e}")
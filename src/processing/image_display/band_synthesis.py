# ===============================================
# 模块名称：band_synthesis.py
# 接口说明：波段合成模块
# 作者：YangQC
# 版本：v1.0.0
# 功能：支持真彩色、假彩色等多种波段合成方案
# 创建时间：2025-06-17
# 最后修改时间：2025-06-17
# ===============================================
from __future__ import annotations
import sys
from pathlib import Path
from typing import Tuple
import numpy as np
import rasterio
if __package__ is None or __package__ == "":
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "src").is_dir():
            sys.path.insert(0, str(parent))
            break

class BandSynthesis:
    """
    波段合成器，支持真彩色、假彩色等波段合成方案
    """
    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self.dataset = rasterio.open(filepath)
        self.band_count = self.dataset.count

    def synthesize(self, band_indices: Tuple[int, int, int]) -> np.ndarray:
        """
        根据输入的波段索引合成三通道影像
        :param band_indices: (R, G, B) 波段编号(从1开始)
        :return: 合成后的3通道影像 (H, W, 3)
        """
        if any(b < 1 or b > self.band_count for b in band_indices):
            raise ValueError(f"波段编号超出范围(1-{self.band_count})")
        data = self.dataset.read(band_indices)
        data = self._normalize_to_uint8(data)
        return np.transpose(data, (1, 2, 0))

    @staticmethod
    def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
        arr_min = arr.min(axis=(1, 2), keepdims=True)
        arr_max = arr.max(axis=(1, 2), keepdims=True)
        arr = (arr - arr_min) / (arr_max - arr_min + 1e-8) * 255
        return arr.clip(0, 255).astype(np.uint8)
    
    def close(self) -> None:
        self.dataset.close()

# ===============================================
# 预留接口（供系统UI调用）
# ===============================================
def synthesize_band(
    filepath: str, rgb_bands: Tuple[int, int, int]
) -> np.ndarray:
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
    test_file = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("AA.tif")
    bands = (4, 3, 2)
    if len(sys.argv) >= 4:
        bands = tuple(map(int, sys.argv[2:5]))  # type: ignore[arg-type]
    try:
        result = synthesize_band(str(test_file), bands)
        print(f"合成图像shape: {result.shape}, dtype: {result.dtype}")
    except Exception as exc:  # pragma: no cover - manual usage
        print(f"测试失败: {exc}")
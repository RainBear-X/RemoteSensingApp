# 文件: src/processing/feature_extraction/indices.py
# 模块: src.processing.feature_extraction.indices
# 功能: 各类光谱指数计算
# 作者: 孟诣楠
# 版本: v1.0.1
# 最近更新: 2025-06-18
# 更新说明:
#   - 从 monolithic 脚本提取所有指数函数

import numpy as np

def calculate_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """归一化植被指数 NDVI"""
    denom = nir + red
    mask = denom > 1e-3
    ndvi = np.zeros_like(nir, dtype=np.float32)
    ndvi[mask] = (nir[mask] - red[mask]) / denom[mask]
    return np.clip(ndvi, -1.0, 1.0)

def calculate_evi(nir: np.ndarray, red: np.ndarray, blue: np.ndarray,
                  L: float=1, C1: float=6, C2: float=7.5, G: float=2.5) -> np.ndarray:
    """增强型植被指数 EVI"""
    denom = nir + C1*red - C2*blue + L
    mask = denom > 1e-3
    evi = np.zeros_like(nir, dtype=np.float32)
    evi[mask] = G*(nir[mask] - red[mask]) / denom[mask]
    return np.clip(evi, -1.0, 1.0)

def calculate_msavi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """修正土壤调整植被指数 MSAVI"""
    msavi = (2*nir + 1 - np.sqrt((2*nir+1)**2 - 8*(nir-red))) / 2
    return np.clip(msavi, -1.0, 1.0)

def calculate_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """归一化水体指数 NDWI"""
    denom = green + nir
    mask = denom > 1e-3
    ndwi = np.zeros_like(green, dtype=np.float32)
    ndwi[mask] = (green[mask] - nir[mask]) / denom[mask]
    return np.clip(ndwi, -1.0, 1.0)

def calculate_mndwi(green: np.ndarray, swir: np.ndarray) -> np.ndarray:
    """改进的归一化水体指数 MNDWI"""
    denom = green + swir
    mask = denom > 1e-3
    mndwi = np.zeros_like(green, dtype=np.float32)
    mndwi[mask] = (green[mask] - swir[mask]) / denom[mask]
    return np.clip(mndwi, -1.0, 1.0)

def calculate_ndbi(swir: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """归一化建筑指数 NDBI"""
    denom = swir + nir
    mask = denom > 1e-3
    ndbi = np.zeros_like(swir, dtype=np.float32)
    ndbi[mask] = (swir[mask] - nir[mask]) / denom[mask]
    return np.clip(ndbi, -1.0, 1.0)

def calculate_bsi(blue: np.ndarray, red: np.ndarray, nir: np.ndarray, swir: np.ndarray) -> np.ndarray:
    """裸土指数 BSI"""
    num = (swir + red) - (nir + blue)
    denom = (swir + red) + (nir + blue)
    mask = denom > 1e-3
    bsi = np.zeros_like(blue, dtype=np.float32)
    bsi[mask] = num[mask] / denom[mask]
    return np.clip(bsi, -1.0, 1.0)

# -*- coding: utf-8 -*-
"""
文件: image_utils.py
模块: src.utils.image_utils
功能: 自动识别 GeoTIFF / ENVI 格式，并统一读取为 numpy 数组 (H, W, C)，并保存影像和掩膜为 .npy
作者: 孟诣楠
版本: v1.5.0
最近更新: 2025-06-25
更新说明:
  - 去除 extract_labeled_samples 提取步骤，仅保存原始图像和掩膜
  - 图像 shape: (H, W, C)，掩膜 shape: (H, W)，输出为 image.npy 和 mask.npy
"""

import os
import numpy as np
import traceback
import rasterio

def parse_envi_header(hdr_path: str) -> dict:
    """读取 ENVI .hdr 文件并解析基本信息"""
    meta = {}
    with open(hdr_path, 'r') as f:
        for line in f:
            if '=' in line:
                key, val = line.strip().split('=', 1)
                meta[key.strip().lower()] = val.strip().strip('{}').strip()
    return meta

def load_envi_as_numpy(envi_path: str) -> np.ndarray:
    """
    读取 ENVI 格式影像并转为 (H, W, C) 格式的 numpy 数组
    """
    base = os.path.splitext(envi_path)[0]
    hdr_path = base + '.hdr'
    if not os.path.exists(hdr_path):
        raise FileNotFoundError(f"找不到头文件: {hdr_path}，无法解析 ENVI 格式")

    meta = parse_envi_header(hdr_path)
    width = int(meta['samples'])
    height = int(meta['lines'])
    bands = int(meta['bands'])
    dtype_code = int(meta['data type'])
    interleave = meta.get('interleave', 'bsq').lower()
    dtype_map = {1: 'uint8', 2: 'int16', 4: 'float32'}

    dtype = np.dtype(dtype_map.get(dtype_code))
    if dtype is None:
        raise ValueError(f"不支持的 ENVI 数据类型: {dtype_code}")

    with open(envi_path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=dtype)

    if interleave == 'bsq':
        data = data.reshape((bands, height, width))
        data = np.transpose(data, (1, 2, 0))
    elif interleave == 'bil':
        data = data.reshape((height, bands, width))
        data = np.transpose(data, (0, 2, 1))
    elif interleave == 'bip':
        data = data.reshape((height, width, bands))
    else:
        raise ValueError(f"不支持的 interleave 格式: {interleave}")

    return data

def load_geotiff_as_numpy(tif_path: str) -> np.ndarray:
    """从 GeoTIFF 读取为 numpy 数组 (H, W, C)"""
    with rasterio.open(tif_path) as src:
        data = src.read()
        return np.transpose(data, (1, 2, 0))

def unified_load_image_as_numpy(path: str) -> np.ndarray:
    """统一入口：根据路径判断 GeoTIFF 或 ENVI，返回 (H, W, C)"""
    try:
        return load_geotiff_as_numpy(path)
    except Exception:
        print(f"⚠️ 尝试 GeoTIFF 读取失败，改为 ENVI 格式解析: {path}")
        return load_envi_as_numpy(path)

def load_tif_as_numpy(tif_path: str, auto_convert: bool = True) -> np.ndarray:
    """
    ✅ 兼容旧接口: 从 GeoTIFF 或 ENVI 文件中读取 numpy 数组 (H, W, C)
    参数:
        tif_path: .tif 文件路径（可为 ENVI 文件）
        auto_convert: 是否自动回退 ENVI 格式解析（保留字段，统一使用 unified）
    """
    return unified_load_image_as_numpy(tif_path)

# === 测试 ===
if __name__ == '__main__':
    image_path = "AA.tif"  # 可为 ENVI 或 GeoTIFF
    mask_path = "mask.npy"

    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"❌ 找不到影像: {image_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"❌ 找不到掩膜: {mask_path}")

        image = load_tif_as_numpy(image_path)
        mask = np.load(mask_path)
        unique, counts = np.unique(mask, return_counts=True)
        print("标签类别分布:", dict(zip(unique, counts)))
        
        print(f"📷 图像 shape: {image.shape}")
        print(f"🎯 掩膜 shape: {mask.shape}")

        if mask.shape != image.shape[:2]:
            raise ValueError("❌ 掩膜尺寸与图像不一致")

        np.save("image.npy", image)
        np.save("mask.npy", mask)
        print("💾 原始图像和掩膜已保存为 image.npy 和 mask.npy")

    except Exception:
        print("🚨 出现错误:")
        traceback.print_exc()
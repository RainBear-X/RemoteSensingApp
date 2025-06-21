# -*- coding: utf-8 -*-
"""
Image Stretching Algorithms
---------------------------
• stretch_linear
• stretch_percent

Created : 2025-06-17
Author  : 张子涵
Version : 0.1.0
"""
from __future__ import annotations
import numpy as np


def _linear(data: np.ndarray, src_min: float, src_max: float,
            dst_min: float = 0.0, dst_max: float = 1.0) -> np.ndarray:
    scaled = (data - src_min) / (src_max - src_min + 1e-12)
    return np.clip(dst_min + scaled * (dst_max - dst_min), dst_min, dst_max)


def stretch_linear(img: np.ndarray,
                   src_min: float | None = None,
                   src_max: float | None = None) -> np.ndarray:
    """Linear stretch with optional source min/max."""
    src_min = np.nanmin(img) if src_min is None else src_min
    src_max = np.nanmax(img) if src_max is None else src_max
    return _linear(img, src_min, src_max)


def stretch_percent(img: np.ndarray,
                    low: float = 2.0, high: float = 98.0) -> np.ndarray:
    """Percent stretch using ``low`` and ``high`` percentiles."""
    out = np.empty_like(img, dtype=np.float32)
    if img.ndim == 3:
        for i in range(img.shape[0]):
            p_low, p_high = np.percentile(img[i], (low, high))
            out[i] = _linear(img[i], p_low, p_high)
    else:
        p_low, p_high = np.percentile(img, (low, high))
        out = _linear(img, p_low, p_high)
    return out

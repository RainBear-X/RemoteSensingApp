# -*- coding: utf-8 -*-
"""
Sharpening Filters
------------------

Created : 2025-06-17
Author  : 张子涵
Version : 0.1.0
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter, laplace


def sharpen_unsharp(img: np.ndarray,
                    radius: float = 1.0, amount: float = 1.0) -> np.ndarray:
    """Unsharp masking. Returns array with same range as input."""
    blurred = gaussian_filter(img, sigma=radius)
    mask = img - blurred
    out = img + amount * mask
    min_v, max_v = np.nanmin(img), np.nanmax(img)
    return np.clip(out, min_v, max_v)


def sharpen_laplacian(img: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Laplacian sharpening. Returns array with same range as input."""
    lap = laplace(img)
    out = img - alpha * lap
    min_v, max_v = np.nanmin(img), np.nanmax(img)
    return np.clip(out, min_v, max_v)

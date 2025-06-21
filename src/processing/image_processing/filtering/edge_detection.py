# -*- coding: utf-8 -*-
"""
Edge Detection
--------------

Created : 2025-06-17
Author  : 张子涵
Version : 0.1.0
"""
from __future__ import annotations
import numpy as np
from skimage import filters, feature


def edge_sobel(img: np.ndarray) -> np.ndarray:
    """Sobel edge detection."""
    if img.ndim == 3:
        img = img[0]
    return filters.sobel(img)


def edge_canny(img: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Canny edge detection."""
    if img.ndim == 3:
        img = img[0]
    return feature.canny(img, sigma=sigma).astype(np.float32)

def edge_roberts(img: np.ndarray) -> np.ndarray:
    """Roberts edge detection."""
    if img.ndim == 3:
        img = img[0]
    return filters.roberts(img)

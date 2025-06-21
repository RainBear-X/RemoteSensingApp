# -*- coding: utf-8 -*-
"""
Smoothing Filters
-----------------

Created : 2025-06-17
Author  : 张子涵
Version : 0.1.0
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import uniform_filter, gaussian_filter, median_filter


def smooth_mean(img: np.ndarray, size: int = 3) -> np.ndarray:
    """Mean (box) filter."""
    return uniform_filter(img, size=size, mode="reflect")


def smooth_gaussian(img: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Gaussian smoothing."""
    return gaussian_filter(img, sigma=sigma, mode="reflect")


def smooth_median(img: np.ndarray, size: int = 3) -> np.ndarray:
    """Median filter."""
    return median_filter(img, size=size, mode="reflect")

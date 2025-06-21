# -*- coding: utf-8 -*-
"""
Histogram Equalization
----------------------

Created : 2025-06-17
Author  : 张子涵
Version : 0.1.0
"""
from __future__ import annotations
import numpy as np
from skimage import exposure


def hist_equalize(img: np.ndarray) -> np.ndarray:
    """Histogram equalization for single or multi-band arrays."""
    if img.ndim == 3:
        return np.stack([exposure.equalize_hist(band) for band in img])
    return exposure.equalize_hist(img)

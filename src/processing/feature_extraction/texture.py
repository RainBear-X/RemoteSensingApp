# 文件: src/processing/feature_extraction/texture.py
# 模块: src.processing.feature_extraction.texture
# 功能: GLCM、LBP、Gabor 纹理特征完整实现
# 作者: 孟诣楠
# 版本: v1.0.2
# 最近更新: 2025-06-18
# 更新说明:
#   - 补充 calculate_glcm_features 和 calculate_gabor_features 的完整实现

import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from src.processing.feature_extraction.utils import (robust_normalize)

def calculate_glcm_features(
        band: np.ndarray,
        distances: list[int] = [1],
        angles: list[float] = [0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels: int = 32,
        window_size: int = 21,
        step_size: int = 21
) -> dict:
    """
    计算灰度共生矩阵(GLCM)纹理特征，并重采样到原始分辨率。

    返回: 字典，包含 'contrast','dissimilarity','homogeneity','energy','correlation'
    """
    # 1. 归一化并量化到 [0, levels)
    band_norm   = robust_normalize(band)
    band_scaled = (band_norm * (levels - 1)).astype(np.uint8)

    h, w = band.shape
    out_h = (h - window_size) // step_size + 1
    out_w = (w - window_size) // step_size + 1

    # 2. 初始化小尺寸特征图
    contrast_img     = np.zeros((out_h, out_w), dtype=np.float32)
    dissim_img       = np.zeros_like(contrast_img)
    homogeneity_img  = np.zeros_like(contrast_img)
    energy_img       = np.zeros_like(contrast_img)
    correlation_img  = np.zeros_like(contrast_img)

    # 3. 滑动窗口计算
    for i in range(0, h - window_size + 1, step_size):
        for j in range(0, w - window_size + 1, step_size):
            window = band_scaled[i:i+window_size, j:j+window_size]
            glcm = graycomatrix(
                window,
                distances=distances,
                angles=angles,
                levels=levels,
                symmetric=True,
                normed=True
            )
            # 各特征取平均
            contrast     = graycoprops(glcm, 'contrast').mean()
            dissimilarity= graycoprops(glcm, 'dissimilarity').mean()
            homogeneity  = graycoprops(glcm, 'homogeneity').mean()
            energy       = graycoprops(glcm, 'energy').mean()
            correlation  = graycoprops(glcm, 'correlation').mean()

            oi = i // step_size
            oj = j // step_size
            contrast_img[oi, oj]    = contrast
            dissim_img[oi, oj]      = dissimilarity
            homogeneity_img[oi, oj] = homogeneity
            energy_img[oi, oj]      = energy
            correlation_img[oi, oj] = correlation

    # 4. 重采样回原始大小
    resize = lambda img: cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    return {
        'contrast':    resize(contrast_img),
        'dissimilarity': resize(dissim_img),
        'homogeneity': resize(homogeneity_img),
        'energy':      resize(energy_img),
        'correlation': resize(correlation_img)
    }

def calculate_lbp_features(
        band: np.ndarray,
        radius: int = 3,
        n_points: int = 24
) -> np.ndarray:
    """
    计算局部二值模式(LBP)特征图，返回归一化后的 [0,1] 浮点数组。
    """
    band_norm = robust_normalize(band)
    img_uint8 = (band_norm * 255).astype(np.uint8)
    lbp = local_binary_pattern(img_uint8, n_points, radius, method='uniform')
    return lbp / lbp.max()

def calculate_gabor_features(
        band: np.ndarray,
        num_scales: int = 4,
        num_orientations: int = 6
) -> list[np.ndarray]:
    """
    生成一组 Gabor 滤波响应特征，返回每个(scale,orientation)组合的响应图列表。
    """
    band_norm  = robust_normalize(band)
    img_uint8  = (band_norm * 255).astype(np.uint8)

    # 参数空间
    scales       = np.logspace(-1, 0.5, num=num_scales)
    orientations = np.linspace(0, np.pi, num_orientations, endpoint=False)

    features = []
    for scale in scales:
        for theta in orientations:
            # 确定 kernel 大小
            ksize = max(5, int(round(5 * scale)) | 1)  # 保证奇数且>=5
            kernel = cv2.getGaborKernel(
                (ksize, ksize),
                sigma=scale,
                theta=theta,
                lambd=10*scale,
                gamma=0.5,
                psi=0,
                ktype=cv2.CV_32F
            )
            # 卷积
            resp = cv2.filter2D(img_uint8, cv2.CV_32F, kernel)
            # 归一化至 [0,1]
            resp = (resp - resp.min()) / (resp.max() - resp.min() + 1e-10)
            features.append(resp.astype(np.float32))
    return features

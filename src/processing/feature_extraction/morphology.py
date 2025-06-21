# 文件: src/processing/feature_extraction/morphology.py
# 模块: src.processing.feature_extraction.morphology
# 功能: 形态学与滤波器响应
# 作者: 孟诣楠
# 版本: v1.0.1
# 最近更新: 2025-06-18
#
# 更新说明:
#   - 从 monolithic 脚本提取形态学和滤波功能

import numpy as np
import cv2
from .utils import robust_normalize

def calculate_morphological_features(band: np.ndarray) -> dict:
    """腐蚀、膨胀、开闭运算、梯度等"""
    band = robust_normalize(band)
    img = (band*255).astype(np.uint8)
    features = {}
    for size in (3,5,7):
        kernel = np.ones((size,size),np.uint8)
        erosion = cv2.erode(img, kernel)
        dilation= cv2.dilate(img,kernel)
        opening = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
        closing = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)
        gradient= cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel)
        features.update({
            f'erosion_{size}': erosion/255.0,
            f'dilation_{size}': dilation/255.0,
            f'opening_{size}': opening/255.0,
            f'closing_{size}': closing/255.0,
            f'gradient_{size}': gradient/255.0
        })
    return features

def calculate_filter_responses(band: np.ndarray) -> dict:
    """高斯、DoG、拉普拉斯、Sobel"""
    band = robust_normalize(band)
    img = (band*255).astype(np.uint8)
    features = {}
    g5  = cv2.GaussianBlur(img,(5,5),0)/255.0
    g15 = cv2.GaussianBlur(img,(15,15),0)/255.0
    dog = (g5 - g15)
    features['dog'] = (dog - dog.min())/(dog.ptp()+1e-10)
    lap= cv2.Laplacian(img,cv2.CV_32F)/255.0
    features['laplacian'] = (lap-lap.min())/(lap.ptp()+1e-10)
    sx  = cv2.Sobel(img,cv2.CV_32F,1,0)/255.0
    sy  = cv2.Sobel(img,cv2.CV_32F,0,1)/255.0
    mag = np.sqrt(sx**2+sy**2)
    features['sobel_mag'] = mag/(mag.max()+1e-10)
    features['gaussian_5']  = g5
    features['gaussian_15'] = g15
    return features

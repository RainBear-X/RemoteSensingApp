# 文件: src/processing/feature_extraction/selection.py
# 模块: src.processing.feature_extraction.selection
# 功能: 方差筛选、多尺度特征
# 作者: 孟诣楠
# 版本: v1.0.1
# 最近更新: 2025-06-18
#
# 更新说明:
#   - 从 monolithic 脚本提取筛选与多尺度功能

import numpy as np
from .utils import robust_normalize
from skimage.filters.rank import entropy
from skimage.morphology import disk
import cv2

def feature_selection_by_variance(features: dict, threshold: float=0.01) -> dict:
    """基于方差的特征选择"""
    sel = {}
    for k,v in features.items():
        if isinstance(v,np.ndarray) and v.ndim==2 and np.var(v)>=threshold:
            sel[k]=v
        elif isinstance(v,dict):
            sub = {sk:sv for sk,sv in v.items() if isinstance(sv,np.ndarray) and np.var(sv)>=threshold}
            if sub: sel[k]=sub
    return sel

def calculate_multi_scale_features(band: np.ndarray, scales=[1,3,5,7]) -> dict:
    """多尺度均值、方差、标准差、局部熵"""
    b = robust_normalize(band)
    res={}
    h,w=b.shape
    for s in scales:
        mean= cv2.blur(b,(s,s)); var= cv2.blur(b*b,(s,s)) - mean*mean
        var[var<0]=0; std=np.sqrt(var)
        res[f'mean_scale_{s}']=mean
        res[f'variance_scale_{s}']=var
        res[f'std_dev_scale_{s}']=std
        if s<=5:
            img8=(b*255).astype(np.uint8)
            ent=entropy(img8,disk(s)); res[f'entropy_scale_{s}']=ent/ent.max()
    return res

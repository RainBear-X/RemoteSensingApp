# 文件: src/processing/feature_extraction/fusion.py
# 模块: src.processing.feature_extraction.fusion
# 功能: 特征融合、层次融合与空间上下文
# 作者: 孟诣楠
# 版本: v1.0.2
# 最近更新: 2025-06-18
# 更新说明:
#   - 补充 feature_fusion_for_segmentation、prepare_features_for_segmentation、hierarchical_feature_fusion 完整实现

import numpy as np
import cv2

def feature_fusion_for_segmentation(
        features: dict,
        selected: list[str] = None,
        method: str = 'weighted_sum'
) -> np.ndarray:
    """
    融合多幅 2D 特征图：
      - weighted_sum: 权重平均（默认等权）
      - concatenate: 通道堆叠
    """
    # 自动选择所有二维数组特征
    if selected is None:
        selected = [
            k for k, v in features.items()
            if isinstance(v, np.ndarray) and v.ndim == 2
        ]
    mats = []
    for name in selected:
        arr = features.get(name)
        if isinstance(arr, np.ndarray) and arr.ndim == 2:
            mats.append((arr - arr.min())/(arr.max()-arr.min()+1e-10))
    if not mats:
        raise ValueError("没有可融合的2D特征")
    if method == 'weighted_sum':
        w = np.ones(len(mats),dtype=np.float32)/len(mats)
        fused = sum(w[i] * mats[i] for i in range(len(mats)))
        return fused
    elif method == 'concatenate':
        return np.stack(mats, axis=-1)
    else:
        raise ValueError(f"未知融合方法: {method}")

def prepare_features_for_segmentation(
        features: dict,
        important: list[str] = None
) -> np.ndarray:
    """
    自动挑选重要特征并堆叠成 (H,W,C) 数组，
    支持带下划线的分量名（如 pca_0）。
    """
    if important is None:
        # 默认指数 + 前3个PCA
        important = [
            k for k in features.keys()
            if any(idx in k.lower() for idx in ['ndvi','evi','msavi','ndwi','mndwi','ndbi','bsi'])
        ]
        # PCA名如 'pca'
        if 'pca' in features and isinstance(features['pca'], list):
            for i in range(min(3,len(features['pca']))):
                important.append(f'pca_{i}')

    mats = []
    for name in important:
        if name in features and isinstance(features[name], np.ndarray) and features[name].ndim==2:
            mats.append(robust_norm := (features[name]-features[name].min())/(np.ptp(features[name])+1e-10))
        elif '_' in name:
            base, idx = name.rsplit('_',1)
            try:
                idx = int(idx)
                lst = features.get(base)
                if isinstance(lst, list) and 0 <= idx < len(lst):
                    mats.append((lst[idx]-lst[idx].min())/(np.ptp(lst[idx])+1e-10))
            except:
                continue
    if not mats:
        raise ValueError("没有找到适合分割的特征")
    return np.stack(mats, axis=-1)

def hierarchical_feature_fusion(
        features: dict
) -> dict:
    """
    分层次融合：
      - level_1: 主要类别（水、植被、建筑、裸土）
      - level_2: 细分类（河/湖、城市植被/山地植被等）
    """
    # L1
    L1 = []
    for key in ['ndwi','mndwi','ndvi','evi','ndbi','bsi']:
        if key in features:
            arr = features[key]
            if isinstance(arr, np.ndarray) and arr.ndim==2:
                L1.append((arr-arr.min())/(np.ptp(arr)+1e-10))
    L1 = np.stack(L1, axis=-1) if L1 else np.zeros((1,1,1),dtype=np.float32)
    # L2 (自定义示例)
    L2 = []
    if 'glcm' in features and isinstance(features['glcm'], dict):
        for sub in ['contrast','homogeneity']:
            if sub in features['glcm']:
                g = features['glcm'][sub]
                L2.append((g-g.min())/(np.ptp(g)+1e-10))
    if 'morphological' in features and isinstance(features['morphological'], dict):
        if 'gradient_5' in features['morphological']:
            g5 = features['morphological']['gradient_5']
            L2.append((g5-g5.min())/(np.ptp(g5)+1e-10))
    L2 = np.stack(L2, axis=-1) if L2 else np.zeros((1,1,1),dtype=np.float32)

    return {'level_1': L1, 'level_2': L2}   

def add_spatial_context(
        arr: np.ndarray,
        window_size: int = 7
) -> np.ndarray:
    """
    对 (H,W,C) 特征数组添加滑动平均上下文，返回 (H,W,2C)。
    """
    h,w,c = arr.shape
    ctx = np.zeros_like(arr)
    for i in range(c):
        ctx[:,:,i] = cv2.boxFilter(arr[:,:,i], -1, (window_size,window_size),
                                   normalize=True, borderType=cv2.BORDER_REFLECT)
    return np.concatenate([arr, ctx], axis=-1)

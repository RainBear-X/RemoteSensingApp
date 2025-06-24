# 文件: src/processing/feature_extraction/pca.py
# 模块: src.processing.feature_extraction.pca
# 功能: PCA 主成分提取
# 作者: 孟诣楠
# 版本: v1.0.1
# 最近更新: 2025-06-18
#
# 更新说明:
#   - 从 monolithic 脚本提取 perform_pca

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

def perform_pca(bands: list[np.ndarray], n_components: int=None,
                use_robust_scaling: bool=True) -> tuple:
    """
    返回 (components_list, explained_variance_ratio, PCA_object)
    """
    h, w = bands[0].shape
    data = np.stack(bands, axis=-1).reshape(-1, len(bands))
    if use_robust_scaling:
        data = RobustScaler().fit_transform(data)
    pca = PCA(n_components=n_components)
    comps = pca.fit_transform(data)
    comps = comps.reshape(h, w, -1).transpose(2,0,1)
    return list(comps), pca.explained_variance_ratio_, pca

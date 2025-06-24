# 文件: src/processing/feature_extraction/visualization.py
# 模块: src.processing.feature_extraction.visualization
# 功能: 特征可视化完整实现
# 作者: 孟诣楠
# 版本: v1.0.2
# 最近更新: 2025-06-18
# 更新说明:
#   - 补充 visualize_selected_features、visualize_hierarchical_features 完整实现

import matplotlib.pyplot as plt
import numpy as np

def visualize_selected_features(
        features: dict,
        max_features: int = 12,
        save_path: str = "selected_features_visualization.png"
):
    """
    扁平化特征字典，展示最多 max_features 个子图，并保存。
    """
    # 1. 扁平化
    flat = {}
    for k, v in features.items():
        if isinstance(v, np.ndarray) and v.ndim == 2:
            flat[k] = v
        elif isinstance(v, list) and all(isinstance(x, np.ndarray) for x in v):
            for idx, arr in enumerate(v):
                flat[f"{k}_{idx}"] = arr
        elif isinstance(v, dict):
            for subk, subv in v.items():
                if isinstance(subv, np.ndarray) and subv.ndim == 2:
                    flat[f"{k}_{subk}"] = subv

    # 2. 限制数量
    names = list(flat.keys())
    names = names[:max_features]

    # 3. 布局
    n = len(names)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    plt.figure(figsize=(4*cols, 3*rows))

    # 4. 绘制
    for i, name in enumerate(names):
        arr = flat[name]
        norm = (arr - arr.min())/(arr.max()-arr.min()+1e-10)
        # 选 cmap
        low = name.lower()
        if 'ndvi' in low: cmap='RdYlGn'
        elif 'ndwi' in low or 'water' in low: cmap='Blues'
        elif 'ndbi' in low or 'build' in low: cmap='hot'
        elif 'pca' in low: cmap='viridis'
        elif any(x in low for x in ['glcm','lbp','texture']): cmap='gray'
        else: cmap='viridis'

        plt.subplot(rows, cols, i+1)
        plt.imshow(norm, cmap=cmap)
        plt.title(name)
        plt.axis('off')
        plt.colorbar(fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def visualize_hierarchical_features(
        hier: dict,
        feats: dict,
        output_dir: str = "."
):
    """
    分层可视化：level_1、level_2 及组合示例。
    """
    # 一级
    L1 = hier.get('level_1')
    if isinstance(L1, np.ndarray):
        plt.figure(figsize=(15,10))
        plt.suptitle('第一级特征 - 主要类别区分', fontsize=16)
        n = min(6, L1.shape[2])
        for i in range(n):
            plt.subplot(2,3,i+1)
            plt.imshow(L1[:,:,i], cmap='viridis')
            plt.title(f'L1-{i+1}')
            plt.axis('off')
            plt.colorbar(fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/level_1_features.png", dpi=300, bbox_inches='tight')
        plt.close()

    # 二级
    L2 = hier.get('level_2')
    if isinstance(L2, np.ndarray) and L2.shape[2] > 1:
        plt.figure(figsize=(15,10))
        plt.suptitle('第二级特征 - 细分类', fontsize=16)
        m = min(6, L2.shape[2])
        for i in range(m):
            plt.subplot(2,3,i+1)
            plt.imshow(L2[:,:,i], cmap='plasma')
            plt.title(f'L2-{i+1}')
            plt.axis('off')
            plt.colorbar(fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/level_2_features.png", dpi=300, bbox_inches='tight')
        plt.close()

    # 示例组合
    if all(key in feats for key in ('ndwi','mndwi','ndvi','evi','ndbi','bsi')):
        plt.figure(figsize=(15,5))
        plt.suptitle('层次特征组合示例', fontsize=16)

        # 水体
        water = (feats['ndwi'] + feats['mndwi'])/2
        plt.subplot(131)
        plt.imshow(water, cmap='Blues')
        plt.title('水体组合'); plt.axis('off'); plt.colorbar(fraction=0.046, pad=0.04)

        # 植被
        veg = (feats['ndvi'] + feats['evi'])/2
        plt.subplot(132)
        plt.imshow(veg, cmap='Greens')
        plt.title('植被组合'); plt.axis('off'); plt.colorbar(fraction=0.046, pad=0.04)

        # 建裸
        urban = (feats['ndbi'] + feats['bsi'])/2
        plt.subplot(133)
        plt.imshow(urban, cmap='OrRd')
        plt.title('建筑/裸土组合'); plt.axis('off'); plt.colorbar(fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/combined_features.png", dpi=300, bbox_inches='tight')
        plt.close()

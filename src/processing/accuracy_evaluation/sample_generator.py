# 文件: src/processing/accuracy_evaluation/sample_generator.py
# 作者: 孟诣楠
# 版本: v1.0.3
# 最新更改时间: 2025-06-25

import numpy as np
import pandas as pd
import rasterio

def random_sample_points(class_map: np.ndarray, num_samples: int, random_state: int = 42):
    # ...同前面
    np.random.seed(random_state)
    H, W = class_map.shape
    total = H * W
    indices = np.random.choice(total, min(num_samples, total), replace=False)
    rows, cols = np.unravel_index(indices, (H, W))
    rastervalu = class_map[rows, cols]
    samples_df = pd.DataFrame({
        "OBJECTID": np.arange(1, len(rows)+1),
        "row": rows,
        "col": cols,
        "RASTERVALU": rastervalu,      # 分类结果
        "CLASSIFIED": [-1]*len(rows)  # -1待人工标注真值
    })
    return samples_df

def save_samples(samples_df: pd.DataFrame, path: str):
    # ...同前面
    out = samples_df[["OBJECTID", "CLASSIFIED", "RASTERVALU", "row", "col"]]
    if path.endswith('.csv'):
        out.to_csv(path, index=False)
    elif path.endswith('.xlsx'):
        out.to_excel(path, index=False)
    else:
        raise ValueError("只支持csv或xlsx后缀！")

if __name__ == "__main__":
    tif_path = "AA.tif"         # 原始.tif路径
    npy_path = "class_map.npy"        # 分类结果npy路径
    num_samples = 30

    # 1. 自动读取原图像的高宽
    with rasterio.open(tif_path) as src:
        H, W = src.height, src.width
        print(f"原图尺寸: 高={H} 宽={W}")

    # 2. 加载分类结果，自动reshape
    arr = np.load(npy_path)
    if arr.ndim == 1:
        if arr.size != H*W:
            raise ValueError(f".npy数组长度({arr.size})与原图像高宽({H}*{W})不符！")
        arr = arr.reshape(H, W)
    print("分类图像shape:", arr.shape)

    # 3. 随机采样并保存
    df = random_sample_points(arr, num_samples=num_samples)
    save_samples(df, "random_samples.xlsx")
    print("采样点已保存到 random_samples.xlsx")

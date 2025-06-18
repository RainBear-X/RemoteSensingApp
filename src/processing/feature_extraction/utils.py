# 文件: src/processing/feature_extraction/utils.py
# 模块: src.processing.feature_extraction.utils
# 功能: 通用辅助函数（如稳健归一化）
# 作者: 孟诣楠
# 版本: v1.0.1
# 最近更新: 2025-06-18
#
# 更新说明:
#   - 从 monolithic 脚本提取 robust_normalize 函数

import numpy as np

def robust_normalize(band: np.ndarray, lower_percentile: float = 2, upper_percentile: float = 98) -> np.ndarray:
    """
    使用稳健归一化方法处理波段数据，避免异常值影响
    """
    min_val = np.percentile(band, lower_percentile)
    max_val = np.percentile(band, upper_percentile)
    clipped = np.clip(band, min_val, max_val)
    eps = 1e-10
    return (clipped - min_val) / (max_val - min_val + eps)

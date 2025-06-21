# ===============================================
# 模块名称：file_loader.py
# 接口说明：遥感影像文件加载器
# 作者：9(冒浩溶)
# 版本：v1.2.0
# 功能：支持多种遥感影像格式的加载
# 创建时间：2025-06-17
# 最后修改时间：2025-06-17
# ===============================================

import os
from typing import Tuple, Optional, Dict, Any

try:
    import rasterio
except ImportError:
    rasterio = None

try:
    from PIL import Image
except ImportError:
    Image = None

import numpy as np

class ImageLoaderError(Exception):
    """自定义异常：影像加载失败"""
    pass

def _load_with_rasterio(file_path: str):
    """使用rasterio加载遥感影像(如tif/img等)"""
    if rasterio is None:
        raise ImageLoaderError("rasterio 未安装，请先安装 rasterio 库。")
    try:
        with rasterio.open(file_path) as src:
            array = src.read()
            meta = src.meta
        return array, meta
    except Exception as e:
        raise ImageLoaderError(f"rasterio 影像加载失败: {str(e)}")

def _load_with_pil(file_path: str):
    """使用Pillow加载常规图片(如jpg/png/bmp等)"""
    if Image is None:
        raise ImageLoaderError("Pillow 未安装，请先安装 pillow 库。")
    try:
        with Image.open(file_path) as img:
            array = np.array(img)
            meta = {
                'mode': img.mode,
                'size': img.size,
                'format': img.format,
            }
        # 补充成C,H,W结构以和rasterio风格一致
        if array.ndim == 2:
            array = array[np.newaxis, :, :]
        elif array.ndim == 3:
            array = array.transpose(2, 0, 1)
        return array, meta
    except Exception as e:
        raise ImageLoaderError(f"Pillow 影像加载失败: {str(e)}")

def load_image(file_path: str, options: Optional[Dict[str, Any]] = None) -> Tuple[Any, Dict]:
    """
    影像加载主接口，自动检测格式，rasterio优先，失败用Pillow兜底

    Args:
        file_path (str): 影像文件路径。
        options (dict, 可选): 额外参数（预留）。

    Returns:
        Tuple: (影像数组, 影像元数据)

    Raises:
        ImageLoaderError: 如果文件无法加载。
    """
    if not os.path.isfile(file_path):
        raise ImageLoaderError(f"文件不存在: {file_path}")

    # 先尝试用rasterio加载，如果不支持或出错再用Pillow兜底
    try:
        return _load_with_rasterio(file_path)
    except Exception as err_rio:
        try:
            return _load_with_pil(file_path)
        except Exception as err_pil:
            raise ImageLoaderError(f"无法加载文件: {file_path}\n"
                                   f"rasterio错误: {err_rio}\n"
                                   f"Pillow错误: {err_pil}")


# ========== 预留接口==========
def open_image_file(file_path: str, options: Optional[Dict[str, Any]] = None):
    """对外接口，加载遥感影像并返回数据与元信息"""
    return load_image(file_path, options)

# ========== 独立测试 ==========
if __name__ == "__main__":
    test_paths = [
        "南海之滨.bmp   "
    ]
    for test_path in test_paths:
        print(f"测试文件: {test_path}")
        try:
            image_array, image_meta = open_image_file(test_path)
            print("影像加载成功！形状：", image_array.shape)
            print("影像元数据：", image_meta)
        except Exception as e:
            print("测试失败:", e)
        print("-" * 40)
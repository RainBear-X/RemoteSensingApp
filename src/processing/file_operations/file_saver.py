# ===============================================
# 模块名称：file_saver.py
# 接口说明：遥感影像文件保存器
# 作者：9(冒浩溶)
# 版本：v1.2.0
# 功能：多种格式图像的保存
# 创建时间：2025-06-17
# 最后修改时间：2025-06-17
# ===============================================

import os
from typing import Any, Dict, Optional

try:
    import rasterio
except ImportError:
    rasterio = None

try:
    from PIL import Image
except ImportError:
    Image = None

import numpy as np

SUPPORTED_SAVE_FORMATS = ('.tif', '.tiff', '.img', '.jpg', '.jpeg', '.png', '.bmp')

class ImageSaverError(Exception):
    pass

def save_image(array: Any, meta: Dict, save_path: str, options: Optional[Dict] = None) -> None:
    ext = os.path.splitext(save_path)[1].lower()
    if ext not in SUPPORTED_SAVE_FORMATS:
        raise ImageSaverError(f"不支持的保存格式: {ext}")

    # 统一确保array为np.ndarray
    if not isinstance(array, np.ndarray):
        # 有些极端case可能传入PIL.Image对象
        array = np.array(array)

    if ext in ('.tif', '.tiff', '.img'):
        if rasterio is None:
            raise ImageSaverError("rasterio 未安装，请先安装 rasterio 库。")
        meta = meta.copy()
        meta.update({'driver': 'GTiff' if ext in ('.tif', '.tiff') else 'HFA'})
        # 自动补齐meta信息
        meta['dtype'] = str(array.dtype)
        meta['count'] = array.shape[0]
        meta['height'] = array.shape[1]
        meta['width'] = array.shape[2]
        # crs/transform可选，保留meta里的内容
        try:
            with rasterio.open(save_path, 'w', **meta) as dst:
                dst.write(array)
        except Exception as e:
            raise ImageSaverError(f"rasterio影像保存失败: {str(e)}")
    elif ext in ('.jpg', '.jpeg', '.png', '.bmp'):
        if Image is None:
            raise ImageSaverError("Pillow 未安装，请先安装 pillow 库。")
        # numpy array (C,H,W) -> (H,W,C) or (H,W)
        if array.ndim == 3:
            # 如果是单通道 (1,H,W)，去掉通道维度
            if array.shape[0] == 1:
                arr = array[0]
            else:
                arr = array.transpose(1, 2, 0)
        elif array.ndim == 2:
            arr = array
        else:
            raise ImageSaverError("输入影像数组维度不支持")
        # 类型兼容（Pillow要求uint8/rgb等）
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        try:
            img = Image.fromarray(arr)
            img.save(save_path)
        except Exception as e:
            raise ImageSaverError(f"Pillow影像保存失败: {str(e)}")
    else:
        raise ImageSaverError(f"暂不支持此格式保存: {ext}")

# ========== 预留接口 ==========
def save_image_file_as(array: Any, meta: Dict, save_path: str, options: Optional[Dict] = None):
    return save_image(array, meta, save_path, options)

# ========== 独立测试 ==========
if __name__ == "__main__":
    # 假设你用file_loader已经加载了一张影像
    from file_loader import open_image_file
    test_path = "南海之滨.bmp"  # 任意能被file_loader加载的文件
    try:
        image_array, image_meta = open_image_file(test_path)
        ext ='.tif'# 想保存成什么格式就写什么后缀
        save_path = f"南海之滨{ext}"
        save_image_file_as(image_array, image_meta, save_path)
        print(f"影像保存成功: {save_path}")
            
    except Exception as e:
        print("测试失败:", e)
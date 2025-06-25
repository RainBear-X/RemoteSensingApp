# ===============================================
# 模块名称：metadata_viewer.py
# 接口说明：遥感影像与矢量数据元数据查看模块
# 作者：YangQC
# 版本：v2.0.0
# 功能：支持查看遥感影像（GDAL）空间参考、波段信息、采集时间等元数据，
#      以及矢量数据（GeoPandas）基本元数据
# 创建时间：2025-06-17
# 最后修改时间：2025-06-17
# ===============================================

from osgeo import gdal, osr
import geopandas as gpd
from typing import Dict, Any
import os

gdal.UseExceptions()
osr.UseExceptions()

class MetadataViewer:
    """
    元数据查看器，读取并解析遥感影像或矢量数据元数据
    """
    def __init__(self, filepath: str):
        self.filepath = os.path.abspath(filepath)
        self.dataset = None
        self.is_vector = None

        # 1. 尝试用GeoPandas读取（矢量数据，如shp等）
        try:
            gdf = gpd.read_file(self.filepath)
            if not gdf.empty:
                self.dataset = gdf
                self.is_vector = True
                return
        except Exception:
            pass

        # 2. 再尝试用GDAL读取（栅格数据，如tif等）
        try:
            ds = gdal.Open(self.filepath)
            if ds is not None:
                self.dataset = ds
                self.is_vector = False
                return
        except Exception:
            pass

        # 都打不开，报错
        raise FileNotFoundError(f"无法打开文件: {filepath}")

    def get_metadata(self) -> Dict[str, Any]:
        if self.is_vector:
            # 矢量数据元数据（GeoPandas方式）
            gdf = self.dataset
            meta = {
                "Type": "Vector",
                "Driver": "GeoPandas",
                "CRS": str(gdf.crs),
                "FeatureCount": len(gdf),
                "GeometryType": str(gdf.geom_type.unique().tolist()),
                "Fields": list(gdf.columns),
                "First5Records": gdf.head().to_dict(orient="records"),
            }
            return meta
        else:
            # 栅格数据元数据
            ds = self.dataset
            meta = {
                "Type": "Raster",
                "Driver": ds.GetDriver().LongName,
                "Size": (ds.RasterXSize, ds.RasterYSize),
                "Bands": ds.RasterCount,
                "Projection": ds.GetProjection(),
                "GeoTransform": ds.GetGeoTransform(),
                "Metadata": ds.GetMetadata(),
                "BandInfo": []
            }
            for i in range(1, ds.RasterCount + 1):
                band = ds.GetRasterBand(i)
                band_meta = {
                    "Band": i,
                    "Description": band.GetDescription(),
                    "DataType": gdal.GetDataTypeName(band.DataType),
                    "NoDataValue": band.GetNoDataValue(),
                    "Min": band.GetMinimum(),
                    "Max": band.GetMaximum(),
                }
                meta["BandInfo"].append(band_meta)
            return meta

    def close(self):
        self.dataset = None

# ===============================================
# 预留接口（供系统UI调用）
# ===============================================
def view_metadata(filepath: str) -> Dict[str, Any]:
    viewer = MetadataViewer(filepath)
    try:
        result = viewer.get_metadata()
    finally:
        viewer.close()
    return result

# ===============================================
# 单元测试
# ===============================================
if __name__ == "__main__":
    test_file = "AA"
    try:
        meta = view_metadata(test_file)
        for k, v in meta.items():
            print(f"{k}: {v}")
    except Exception as e:
        print(f"测试失败: {e}")
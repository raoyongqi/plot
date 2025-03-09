import rasterio
import numpy as np
from rasterio.warp import reproject, Resampling
import os

# 读取目标 TIFF 获取尺寸
target_tiff = "data/cropped/soil_tiff/TMIN.tif"
if not os.path.exists(target_tiff):
    raise FileNotFoundError(f"Target TIFF {target_tiff} not found!")

with rasterio.open(target_tiff) as ref:
    target_transform = ref.transform
    target_width = ref.width
    target_height = ref.height
    target_crs = ref.crs

print(f"Target width: {target_width}, Target height: {target_height}")

# 读取源 TIFF
src_tiff = "data/cropped/geo/tiff/cropped_hand_500m_china_03_08.tif"
with rasterio.open(src_tiff) as src:
    data = src.read(1)  # 读取数据

    # 直接使用目标尺寸
    transform = src.transform
    width = target_width
    height = target_height

    # 创建新的数据数组
    resampled_data = np.empty((height, width), dtype=src.dtypes[0])

    # 执行重采样
    reproject(
        source=data,
        destination=resampled_data,
        src_transform=src.transform,
        dst_transform=target_transform,
        src_crs=src.crs,
        dst_crs=target_crs,
        resampling=Resampling.bilinear
    )

    # 更新元数据
    new_meta = src.meta.copy()
    new_meta.update({
        "height": height,
        "width": width,
        "transform": target_transform
    })

    # 保存结果
    output_tiff = "data/cropped_result/tiff/cropped_hand_500m_china_03_08_resampled.tif"
    with rasterio.open(output_tiff, "w", **new_meta) as dst:
        dst.write(resampled_data, 1)

print(f"Resampling completed! Saved to {output_tiff}")

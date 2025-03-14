import os
import rasterio
from rasterio.mask import mask
import numpy as np
import geopandas as gpd
import platform
if platform.system() == "Windows":

    base_path = r'C:\Users\r\Desktop\work12'
else:

    base_path = '/home/r/Desktop/work12'

geojson_file = os.path.join(base_path,'中华人民共和国.json')

grasslands_gdf = gpd.read_file(geojson_file)

if platform.system() == "Windows":
    tiff_folders = [
        r'C:\Users\r\Desktop\work12\data\climate\wc2.1_5m',
    ]
else:
    tiff_folders = [
        '/home/r/Desktop/work12/data/result/',
    ]

geojson_output_folder = os.path.join(base_path,'data', 'cropped', 'geojson')
tiff_output_folder = os.path.join(base_path,'data', 'cropped', 'tiff')



os.makedirs(geojson_output_folder, exist_ok=True)
os.makedirs(tiff_output_folder, exist_ok=True)

for tiff_folder in tiff_folders:
    if not os.path.isdir(tiff_folder):
        print(f"Folder does not exist: {tiff_folder}")
        continue

    for tiff_file in os.listdir(tiff_folder):
        if tiff_file.endswith('.tif'):
            tiff_path = os.path.join(tiff_folder, tiff_file)
            
            tiff_output_path = os.path.join(tiff_output_folder, f'cropped_{tiff_file}')
            geojson_output_path = os.path.join(geojson_output_folder, f'cropped_{os.path.splitext(tiff_file)[0]}.geojson')

            with rasterio.open(tiff_path) as src:

                image_data = src.read(1).astype(np.float32)

                with rasterio.MemoryFile() as memfile:
                    with memfile.open(
                        driver="GTiff",
                        height=image_data.shape[0],
                        width=image_data.shape[1],
                        count=1,
                        dtype="float32",
                        crs=src.crs,
                        transform=src.transform,
                        nodata=np.nan,
                    ) as dataset:
                        dataset.write(image_data, 1)

                        out_image, out_transform = mask(dataset, grasslands_gdf.geometry, crop=True, nodata=np.nan)

                        out_meta = dataset.meta.copy()
                        out_meta.update({
                            "driver": "GTiff",
                            "height": out_image.shape[1],
                            "width": out_image.shape[2],
                            "transform": out_transform,
                            "dtype": "float32",  # 保持数据类型为浮点型
                            "nodata": 0  # 设置 nodata 为 0
                        })

                        out_image = np.where(np.isnan(out_image), 0, out_image)  # 这里将 NaN 替换为 0

                        with rasterio.open(tiff_output_path, "w", **out_meta) as dest:
                            # Write the first band only, as we're dealing with single-band images
                            dest.write(out_image[0], 1)  # out_image[0] is a 2D array

            with rasterio.open(tiff_output_path) as cropped_src:
                for idx, row in grasslands_gdf.iterrows():
                    geom = row['geometry']
                    if geom.is_empty:
                        continue

                    if geom.geom_type == 'Polygon':

                        coords = np.array(geom.exterior.coords)
                    elif geom.geom_type == 'MultiPolygon':

                        coords = []
                        for poly in geom.geoms:  # 使用 .geoms 获取所有 Polygon
                            coords.extend(np.array(poly.exterior.coords))  # 添加每个多边形的外部坐标
                    else:
                        continue  # 如果是其他类型，跳过

                    pixel_coords = [cropped_src.index(x, y) for x, y in coords]

                    pixel_values = []
                    for row, col in pixel_coords:
                        if 0 <= row < cropped_src.height and 0 <= col < cropped_src.width:
                            pixel_value = cropped_src.read(1)[row, col]
                            if not np.isnan(pixel_value):
                                pixel_values.append(pixel_value)

                    if pixel_values:
                        avg_value = np.nanmean(pixel_values)  # 使用平均值来代表区域
                        grasslands_gdf.at[idx, 'value'] = avg_value
                    else:
                        grasslands_gdf.at[idx, 'value'] = np.nan

            # 保存更新后的 GeoJSON 文件
            grasslands_gdf.to_file(geojson_output_path, driver="GeoJSON")

                        
            print(f"Clipped TIFF image saved to {tiff_output_path}")
            print(f"Corresponding GeoJSON saved to {geojson_output_path}")

import os
import rasterio
from rasterio.mask import mask
import numpy as np
import geopandas as gpd
import platform
if platform.system() == "Windows":
    # Windows 环境的文件路径
    base_path = r'C:\Users\r\Desktop\work12'
else:
    # Linux 环境的文件路径
    base_path = '/home/r/Desktop/r-climate'
geojson_file_path = '中华人民共和国分市.json'  # 请确保这个文件路径正确
gdf_geojson = gpd.read_file(geojson_file_path)

import pandas as pd

# Read the Excel file
df = pd.read_csv('data/climate_soil_loc.csv')  # Replace with the path to your Excel file

import pandas as pd




cities_data = df[['City','Province','District']]

# Get the unique cities from the "City" column
unique_cities = df['City'].unique()

filtered_gdf = gdf_geojson[gdf_geojson['name'].isin(unique_cities)]


import geopandas as gpd
import matplotlib.pyplot as plt

filtered_gdf = filtered_gdf[filtered_gdf.is_valid]


merged_df = pd.merge(cities_data, filtered_gdf, left_on='City', right_on='name', how='inner')


provinces_to_include = ['西藏自治区', '新疆维吾尔自治区', '甘肃省', '青海省', '四川省', '内蒙古自治区']

# 过滤 merged_df，保留 Province 列在 provinces_to_include 列表中的数据
filtered_merged_df = merged_df[merged_df['Province'].isin(provinces_to_include)]


province_gdf =  gpd.read_file('中华人民共和国.json')

excluded_provinces = ['西藏自治区', '新疆维吾尔自治区', '甘肃省', '青海省', '四川省', '内蒙古自治区']

# 使用 `.loc` 和 `.isin()` 来筛选出不在 excluded_provinces 列表中的省份
filtered_province_gdf = province_gdf[~province_gdf['name'].isin(excluded_provinces)]

merged_gdf = gpd.GeoDataFrame( pd.concat([filtered_merged_df, filtered_province_gdf], ignore_index=True))


if platform.system() == "Windows":
    tiff_folders = [
        r'C:\Users\r\Desktop\work12\data\result',
    ]
else:
    tiff_folders = [
        '/home/r/Desktop/r-climate/data/result/',
    ]

geojson_output_folder = os.path.join(base_path, 'data', 'cropped_result', 'geojson')
tiff_output_folder = os.path.join(base_path, 'data', 'cropped_result', 'tiff')

os.makedirs(geojson_output_folder, exist_ok=True)
os.makedirs(tiff_output_folder, exist_ok=True)

import os
import rasterio
import numpy as np
from rasterio.mask import mask
from shapely.geometry import Polygon, MultiPolygon

for tiff_folder in tiff_folders:
    if not os.path.isdir(tiff_folder):
        print(f"Folder does not exist: {tiff_folder}")
        continue

    for tiff_file in os.listdir(tiff_folder):
        if tiff_file.endswith('.tif'):
            tiff_path = os.path.join(tiff_folder, tiff_file)
            
            # 构建输出路径
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

                        out_image, out_transform = mask(dataset, merged_gdf.geometry, crop=True, nodata=np.nan)

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
                            dest.write(out_image[0], 1)  # out_image[0] 是二维数组

            with rasterio.open(tiff_output_path) as cropped_src:
                for idx, row in merged_gdf.iterrows():
                    geom = row['geometry']
                    if geom.is_empty:
                        continue

                    polygons = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms)

                    all_pixel_values = []

                    for poly in polygons:

                        coords = np.array(poly.exterior.coords)

                        pixel_coords = [cropped_src.index(x, y) for x, y in coords]

                        for row, col in pixel_coords:
                            if 0 <= row < cropped_src.height and 0 <= col < cropped_src.width:
                                pixel_value = cropped_src.read(1)[row, col]
                                if not np.isnan(pixel_value):
                                    all_pixel_values.append(pixel_value)

                    if all_pixel_values:
                        avg_value = np.nanmean(all_pixel_values)  # 使用平均值来代表区域
                        merged_gdf.at[idx, 'value'] = avg_value
                    else:
                        merged_gdf.at[idx, 'value'] = np.nan  # 如果没有有效值，则设置为 NaN

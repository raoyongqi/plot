import os
import rasterio
from rasterio.mask import mask
import numpy as np
import geopandas as gpd
import platform

# 1. 设置基础路径
if platform.system() == "Windows":
    base_path = r'C:\Users\r\Desktop\work12'
else:
    base_path = '/home/r/Desktop/work12'

# 2. 加载 GeoJSON 文件
geojson_file = os.path.join(base_path, '中华人民共和国.json')
print(f"Loading GeoJSON file: {geojson_file}")

try:
    grasslands_gdf = gpd.read_file(geojson_file)
    print(f"GeoJSON loaded successfully. Total features: {len(grasslands_gdf)}")
except Exception as e:
    print(f"Error loading GeoJSON: {e}")
    exit(1)

# 3. 设定 TIFF 文件路径
if platform.system() == "Windows":
    tiff_folders = [r'C:\Users\r\Desktop\work12\data\geo']
else:
    tiff_folders = ['/home/r/Desktop/work12/data/result/']

# 4. 设置输出路径
geojson_output_folder = os.path.join(base_path, 'data', 'cropped', 'geo', 'geojson')
tiff_output_folder = os.path.join(base_path, 'data', 'cropped', 'geo', 'tiff')

os.makedirs(geojson_output_folder, exist_ok=True)
os.makedirs(tiff_output_folder, exist_ok=True)

# 5. 遍历 TIFF 文件
for tiff_folder in tiff_folders:
    if not os.path.isdir(tiff_folder):
        print(f"Error: Folder does not exist -> {tiff_folder}")
        continue

    for tiff_file in os.listdir(tiff_folder):
        if not tiff_file.endswith('.tif'):
            continue
        
        tiff_path = os.path.join(tiff_folder, tiff_file)
        tiff_output_path = os.path.join(tiff_output_folder, f'cropped_{tiff_file}')
        geojson_output_path = os.path.join(geojson_output_folder, f'cropped_{os.path.splitext(tiff_file)[0]}.geojson')

        print(f"\nProcessing TIFF: {tiff_path}")

        with rasterio.open(tiff_path) as src:
            print(f"TIFF opened successfully. CRS: {src.crs}, Size: {src.width}x{src.height}")
            
            if grasslands_gdf.crs != src.crs:
                print("Reprojecting GeoJSON to match TIFF CRS...")
                grasslands_gdf = grasslands_gdf.to_crs(src.crs)

            image_data = src.read(1).astype(np.float32)

            print(f"src...{src}")
            out_image, out_transform = mask(src, grasslands_gdf.geometry, crop=True, nodata=np.nan)
            
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "dtype": "float32",
                "nodata": 0
            })

            out_image = np.where(np.isnan(out_image), 0, out_image)


        try:
            with rasterio.open(tiff_output_path, "w", **out_meta) as dest:
                dest.write(out_image[0], 1)
            print(f"Clipped TIFF saved: {tiff_output_path}")
        except Exception as e:
            print(f"Error saving clipped TIFF: {e}")
            continue

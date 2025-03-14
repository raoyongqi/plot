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
        r'C:\Users\r\Desktop\work12\data\HWSD_1247\tif',
    ]
else:
    tiff_folders = [
        '/home/r/Desktop/work12/data/result/',
    ]



geojson_output_folder = os.path.join(base_path,'data', 'cropped', 'geojson')
tiff_output_folder = os.path.join(base_path,'data', 'cropped', 'soil_tiff')



os.makedirs(geojson_output_folder, exist_ok=True)
os.makedirs(tiff_output_folder, exist_ok=True)

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

                        out_image, out_transform = mask(dataset, grasslands_gdf.geometry, crop=True, nodata=np.nan)

                        # 更新元数据
                        out_meta = dataset.meta.copy()
                        out_meta.update({
                            "driver": "GTiff",
                            "height": out_image.shape[1],
                            "width": out_image.shape[2],
                            "transform": out_transform,
                            "dtype": "float32",
                            "nodata": 0
                        })

                        out_image = np.where(np.isnan(out_image), 0, out_image)

                        with rasterio.open(tiff_output_path, "w", **out_meta) as dest:

                            dest.write(out_image[0], 1)


            with rasterio.open(tiff_output_path) as cropped_src:
                for idx, row in grasslands_gdf.iterrows():
                    geom = row['geometry']
                    if geom.is_empty:
                        continue

                    if geom.geom_type == 'Polygon':

                        coords = np.array(geom.exterior.coords)
                    elif geom.geom_type == 'MultiPolygon':
                        
                        coords = []
                        for poly in geom.geoms:

                            coords.extend(np.array(poly.exterior.coords))
                    else:
                        continue

                    pixel_coords = [cropped_src.index(x, y) for x, y in coords]

                    pixel_values = []
                    for row, col in pixel_coords:
                        if 0 <= row < cropped_src.height and 0 <= col < cropped_src.width:
                            pixel_value = cropped_src.read(1)[row, col]
                            if not np.isnan(pixel_value):
                                pixel_values.append(pixel_value)

                    if pixel_values:
                        avg_value = np.nanmean(pixel_values)
                        grasslands_gdf.at[idx, 'value'] = avg_value
                    else:
                        grasslands_gdf.at[idx, 'value'] = np.nan

            grasslands_gdf.to_file(geojson_output_path, driver="GeoJSON")

                        
            print(f"Clipped TIFF image saved to {tiff_output_path}")
            print(f"Corresponding GeoJSON saved to {geojson_output_path}")

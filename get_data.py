import pandas as pd
import numpy as np
import os
import rasterio
from rasterio.transform import rowcol
import platform
import os

# 判断操作系统并设置路径
if platform.system() == "Windows":
    base_path = r'C:\Users\r\Desktop\work12\data'
else:  # Linux 或其他 Unix 系统
    base_path = '/home/r/Desktop/work12/data'

# 动态设置文件路径
excel_file = os.path.join(base_path, 'climate_data.xlsx')
tif_folder = os.path.join(base_path, 'HWSD_1247', 'tif')
output_excel = os.path.join(base_path, 'climate_soil_tif.xlsx')



# 读取 Excel 文件中的经纬度数据
df = pd.read_excel(excel_file)
df.columns = [col.upper() for col in df.columns]

# 确保 DataFrame 中存在 LAT 和 LON 列
if 'LAT' not in df.columns or 'LON' not in df.columns:
    raise ValueError("Excel file must contain 'LAT' and 'LON' columns")

def get_band_data(tif_file, lat_lon_points):
    with rasterio.open(tif_file) as src:
        band_data = src.read(1)  # 读取第一个波段
        transform = src.transform

        # 计算行列索引
        lat_lon_points_array = np.array([lat_lon_points['LON'], lat_lon_points['LAT']]).T
        row_col_indices = [rowcol(transform, lon, lat) for lon, lat in lat_lon_points_array]
        row_col_indices = np.array(row_col_indices).astype(int)

        # 提取栅格值
        pixel_values = []
        # pixel_values = [band_data[row, col] for row, col in row_col_indices]
        for indice in row_col_indices:

            pixel_values.append(band_data[*indice])

        # pixel_values = np.where(np.isnan(pixel_values), -9999, pixel_values)
        
        return pixel_values

# # 获取文件夹中的所有 TIFF 文件
tif_files = [f for f in os.listdir(tif_folder) if f.endswith('.tif')]
print(tif_files)

def process_file(tif_file):
    tif_file_path = os.path.join(tif_folder, tif_file)
    
    # 提取 TIFF 文件的波段数据
    band_data = get_band_data(tif_file_path, df)
    
    # 提取 TIFF 文件的基本文件名（不包含扩展名）作为列名
    band_name = os.path.splitext(os.path.basename(tif_file))[0]
    
    # 将数据添加到 DataFrame 中
    df[band_name] = band_data

# 对每个 TIFF 文件进行处理
for tif_file in tif_files:
    process_file(tif_file)

# 将结果输出为新的 Excel 文件
df.to_excel(output_excel, index=False)
print(f'Results have been saved to {output_excel}')

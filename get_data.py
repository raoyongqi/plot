import pandas as pd
import numpy as np
import os
import rasterio
from rasterio.transform import rowcol
import platform
import os

if platform.system() == "Windows":
    base_path = r'C:\Users\r\Desktop\work12\data'
else:
    base_path = '/home/r/Desktop/work12/data'

excel_file = os.path.join(base_path, 'climate_data.xlsx')
tif_folder = os.path.join(base_path, 'HWSD_1247', 'tif')
output_excel = os.path.join(base_path, 'climate_soil_tif.xlsx')



df = pd.read_excel(excel_file)
df.columns = [col.upper() for col in df.columns]

if 'LAT' not in df.columns or 'LON' not in df.columns:
    raise ValueError("Excel file must contain 'LAT' and 'LON' columns")

def get_band_data(tif_file, lat_lon_points):
    with rasterio.open(tif_file) as src:
        band_data = src.read(1)  # 读取第一个波段
        transform = src.transform

        lat_lon_points_array = np.array([lat_lon_points['LON'], lat_lon_points['LAT']]).T
        row_col_indices = [rowcol(transform, lon, lat) for lon, lat in lat_lon_points_array]
        row_col_indices = np.array(row_col_indices).astype(int)

        pixel_values = []

        for indice in row_col_indices:

            pixel_values.append(band_data[*indice])

        
        return pixel_values

tif_files = [f for f in os.listdir(tif_folder) if f.endswith('.tif')]
print(tif_files)

def process_file(tif_file):
    tif_file_path = os.path.join(tif_folder, tif_file)
    
    band_data = get_band_data(tif_file_path, df)
    
    band_name = os.path.splitext(os.path.basename(tif_file))[0]
    
    df[band_name] = band_data

for tif_file in tif_files:
    process_file(tif_file)

df.to_excel(output_excel, index=False)
print(f'Results have been saved to {output_excel}')

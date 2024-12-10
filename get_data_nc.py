import pandas as pd
import numpy as np
import os
import xarray as xr

# 输入文件路径和文件夹路径
excel_file = 'data/climate_data.xlsx'
nc_folder = 'data/HWSD_1247/data'
output_excel = 'data/climate_soil.xlsx'

# 读取 Excel 文件中的经纬度数据
df = pd.read_excel(excel_file)

# 将列名转换为大写
df.columns = [col.upper() for col in df.columns]

# 确保 DataFrame 中存在 LAT 和 LON 列
if 'LAT' not in df.columns or 'LON' not in df.columns:
    raise ValueError("Excel file must contain 'LAT' and 'LON' columns")

# 定义一个函数来从 NetCDF 文件中获取波段数据
def get_band_data(nc_file, lat_lon_points):
    with xr.open_dataset(nc_file) as ds:
        latitudes = ds['lat'].values
        longitudes = ds['lon'].values

        band_data = {}  # 创建一个字典来存储每个波段的数据
        
        # 遍历数据中的每个波段
        for band_name in ds.data_vars:
            if band_name in ['lat', 'lon']:
                continue

            # 获取当前波段的数据
            band_values = ds[band_name].values

            closest_band_data_list = []  # 存储当前波段每个经纬度点的值
            if len(ds[band_name].values.shape) == 3:
                print(f"Skipping {band_name} as it is 3D.")
                if band_name == 'AWT_SOC':
                    continue  # Skip to the next band in this case
                if band_name == 'BULK_DEN':
                    continue  # Skip to the next band in this case

                if band_name == 'DOM_MU':
                    continue  # Skip to the next band in this case
                if band_name == 'DOM_SOC':
                    continue  # Skip to the next band in this case

                if band_name == 'PCT_CLAY':
                    continue  # Skip to the next band in this case
                if band_name == 'PCT_SAND':
                    continue  # Skip to the next band in this case
                if band_name == 'PH':
                    continue  # Skip to the next band in this case
                if band_name == 'REF_BULK':
                    continue  # Skip to the next band in this case
            for lat_lon in lat_lon_points:
                lat_idx = np.abs(latitudes - lat_lon[1]).argmin()  # 计算最接近的纬度索引
                lon_idx = np.abs(longitudes - lat_lon[0]).argmin()  # 计算最接近的经度索引

                closest_band_data = band_values[lat_idx, lon_idx]
                closest_band_data_list.append(closest_band_data)
            
            # 将该波段的所有数据存储到字典中
            band_data[band_name] = np.where(np.isnan(closest_band_data_list), -9999, closest_band_data_list)

        return band_data  # 返回波段数据字典

# 获取文件夹中的所有 NetCDF 文件
nc_files = [f for f in os.listdir(nc_folder) if f.endswith('.nc') or f.endswith('.nc4')]

lat_lon_points = df[['LON', 'LAT']].values

# 顺序处理 NetCDF 文件
for nc_file in nc_files:
    # 获取每个 NetCDF 文件中的波段数据
    band_data = get_band_data(os.path.join(nc_folder, nc_file), lat_lon_points)

    # 将每个波段的数据添加到 DataFrame 中
    for band_name, values in band_data.items():
        df[band_name] = values

# 将结果输出为新的 Excel 文件
df.to_excel(output_excel, index=False)
print(f'Results have been saved to {output_excel}')

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

if 'LAT' not in df.columns or 'LON' not in df.columns:
    raise ValueError("Excel file must contain 'LAT' and 'LON' columns")

def get_band_data(nc_file, lat_lon_points):
    with xr.open_dataset(nc_file) as ds:
        latitudes = ds['lat'].values
        longitudes = ds['lon'].values

        band_data = {}
        
        for band_name in ds.data_vars:
            if band_name in ['lat', 'lon']:
                continue

            band_values = ds[band_name].values

            closest_band_data_list = []
            if len(ds[band_name].values.shape) == 3:
                print(f"Skipping {band_name} as it is 3D.")
                if band_name == 'AWT_SOC':
                    continue  
                if band_name == 'BULK_DEN':
                    continue  

                if band_name == 'DOM_MU':
                    continue 
                if band_name == 'DOM_SOC':
                    continue

                if band_name == 'PCT_CLAY':
                    continue
                if band_name == 'PCT_SAND':
                    continue
                if band_name == 'PH':
                    continue
                if band_name == 'REF_BULK':
                    continue
            for lat_lon in lat_lon_points:

                lat_idx = np.abs(latitudes - lat_lon[1]).argmin()
                lon_idx = np.abs(longitudes - lat_lon[0]).argmin()

                closest_band_data = band_values[lat_idx, lon_idx]
                closest_band_data_list.append(closest_band_data)
            
            band_data[band_name] = np.where(np.isnan(closest_band_data_list), -9999, closest_band_data_list)

        return band_data
    
nc_files = [f for f in os.listdir(nc_folder) if f.endswith('.nc') or f.endswith('.nc4')]

lat_lon_points = df[['LON', 'LAT']].values

for nc_file in nc_files:

    band_data = get_band_data(os.path.join(nc_folder, nc_file), lat_lon_points)

    for band_name, values in band_data.items():
        df[band_name] = values

df.to_excel(output_excel, index=False)
print(f'Results have been saved to {output_excel}')

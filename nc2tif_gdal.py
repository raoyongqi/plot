import os
import subprocess
from netCDF4 import Dataset

# 输入和输出文件夹路径
input_folder = r'C:\Users\r\Desktop\work12\data\HWSD_1247\data'
output_folder = r'C:\Users\r\Desktop\work12\data\HWSD_1247\tif'


os.makedirs(output_folder, exist_ok=True)

def get_nodata_value(nc_file):

    with Dataset(nc_file, 'r') as ds:
        # 假设数据在第一个变量中，检查它的缺失值属性
        var = ds.variables[list(ds.variables.keys())[0]]
        return var._FillValue if '_FillValue' in var.ncattrs() else None

def get_variable_attributes(nc_file):

    with Dataset(nc_file, 'r') as ds:
        variables = {}
        for var_name in ds.variables:
            var = ds.variables[var_name]
            # 通过检查维度数量来忽略坐标变量
            if len(var.dimensions) > 1:
                attributes = {}
                if 'long_name' in var.ncattrs():
                    attributes['long_name'] = var.long_name
                if 'variable' in var.ncattrs():
                    attributes['variable'] = var.variable
                variables[var_name] = attributes
        return variables

def get_crs_info(nc_file):

    with Dataset(nc_file, 'r') as ds:

        if 'crs' in ds.ncattrs():
            crs_info = ds.getncattr('crs')
        else:

            crs_info = None

        if crs_info is None:
            crs_info = 'EPSG:4326'
        return crs_info

def convert_and_resample(input_file, variable_attributes, nodata_value, crs_info):

    for var_name, attrs in variable_attributes.items():
        long_name = attrs.get('long_name', var_name)
        variable = attrs.get('variable', var_name)
        
        temp_tif = os.path.join(output_folder, f'{os.path.splitext(os.path.basename(input_file))[0]}_{variable}_temp.tif')

        final_output_file = os.path.join(output_folder, f'{os.path.splitext(os.path.basename(input_file))[0]}_{variable}_resampled.tif')
        
        translate_command = [
            'gdal_translate',
            '-of', 'GTiff',   
            f'-a_nodata {nodata_value}' if nodata_value is not None else '', 
            '-ot', 'Float32',
            '-b', str(list(variable_attributes.keys()).index(var_name) + 1),
            input_file,
            temp_tif
        ]
        
        if crs_info is not None:
            translate_command.extend(['-a_srs', crs_info])
        
        translate_command = [arg for arg in translate_command if arg]
        
        try:
            subprocess.run(translate_command, check=True, text=True, capture_output=True)
            print(f'Successfully converted {input_file} variable {variable} to {temp_tif}')
            
            warp_command = [
                'gdalwarp',
                '-tr', '0.083333333333333', '0.083333333333333',
                '-r', 'bilinear',
                temp_tif,
                final_output_file
            ]
            
            subprocess.run(warp_command, check=True, text=True, capture_output=True)
            print(f'Successfully resampled {temp_tif} to {final_output_file}')
        
        except subprocess.CalledProcessError as e:
            print(f'Error processing {input_file} variable {variable}')
            print(f'Standard Output:\n{e.stdout}')
            print(f'Standard Error:\n{e.stderr}')
        finally:
            if os.path.exists(temp_tif):
                os.remove(temp_tif)

for file_name in os.listdir(input_folder):
    if file_name.endswith('.nc4'):

        input_file = os.path.join(input_folder, file_name)
        nodata_value = get_nodata_value(input_file)
        
        variable_attributes = get_variable_attributes(input_file)
        
        crs_info = get_crs_info(input_file)
        
        convert_and_resample(input_file, variable_attributes, nodata_value, crs_info)

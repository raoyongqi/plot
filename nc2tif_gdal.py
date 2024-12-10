import os
import subprocess
from netCDF4 import Dataset

# 输入和输出文件夹路径
input_folder = r'C:\Users\r\Desktop\work12\data\HWSD_1247\data'
output_folder = r'C:\Users\r\Desktop\work12\data\HWSD_1247\tif'


# 创建输出文件夹（如果不存在）
os.makedirs(output_folder, exist_ok=True)

def get_nodata_value(nc_file):
    """
    获取 NetCDF 文件中的 NoData 值
    """
    with Dataset(nc_file, 'r') as ds:
        # 假设数据在第一个变量中，检查它的缺失值属性
        var = ds.variables[list(ds.variables.keys())[0]]
        return var._FillValue if '_FillValue' in var.ncattrs() else None

def get_variable_attributes(nc_file):
    """
    获取 NetCDF 文件中的所有数据变量及其属性，忽略坐标变量
    """
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
    """
    从 NetCDF 文件中获取 CRS 信息。如果找不到，则返回 WGS84 坐标系。
    """
    with Dataset(nc_file, 'r') as ds:
        # 尝试获取 CRS 信息（根据 NetCDF 文件的具体情况）
        if 'crs' in ds.ncattrs():
            crs_info = ds.getncattr('crs')
        else:
            # 如果 CRS 信息不在 'crs' 中，尝试其他常见的属性名
            crs_info = None
        # 如果 CRS 信息不可用，返回 WGS84
        if crs_info is None:
            crs_info = 'EPSG:4326'
        return crs_info

def convert_and_resample(input_file, variable_attributes, nodata_value, crs_info):
    """
    将 NetCDF 文件转换为 TIFF 并重采样到指定分辨率
    """
    for var_name, attrs in variable_attributes.items():
        long_name = attrs.get('long_name', var_name)
        variable = attrs.get('variable', var_name)
        
        # 临时未重采样的 TIFF 文件路径
        temp_tif = os.path.join(output_folder, f'{os.path.splitext(os.path.basename(input_file))[0]}_{variable}_temp.tif')
        # 重采样后的 TIFF 文件路径
        final_output_file = os.path.join(output_folder, f'{os.path.splitext(os.path.basename(input_file))[0]}_{variable}_resampled.tif')
        
        # 构建 gdal_translate 命令
        translate_command = [
            'gdal_translate',
            '-of', 'GTiff',   # 指定输出格式为 GTiff
            f'-a_nodata {nodata_value}' if nodata_value is not None else '',  # 指定 NoData 值，如果不存在则忽略
            '-ot', 'Float32',  # 指定输出数据类型为 Float32
            '-b', str(list(variable_attributes.keys()).index(var_name) + 1),  # 指定要转换的波段
            input_file,  # 输入文件
            temp_tif  # 临时 TIFF 文件
        ]
        
        # 添加 CRS 信息到 gdal_translate 命令
        if crs_info is not None:
            translate_command.extend(['-a_srs', crs_info])
        
        # 移除空字符串元素
        translate_command = [arg for arg in translate_command if arg]
        
        try:
            subprocess.run(translate_command, check=True, text=True, capture_output=True)
            print(f'Successfully converted {input_file} variable {variable} to {temp_tif}')
            
            # 构建 gdalwarp 命令进行重采样
            warp_command = [
                'gdalwarp',
                '-tr', '0.083333333333333', '0.083333333333333',  # 目标分辨率
                '-r', 'bilinear',  # 重采样方法：双线性插值
                temp_tif,  # 输入临时 TIFF 文件
                final_output_file  # 输出最终 TIFF 文件
            ]
            
            subprocess.run(warp_command, check=True, text=True, capture_output=True)
            print(f'Successfully resampled {temp_tif} to {final_output_file}')
        
        except subprocess.CalledProcessError as e:
            print(f'Error processing {input_file} variable {variable}')
            print(f'Standard Output:\n{e.stdout}')
            print(f'Standard Error:\n{e.stderr}')
        finally:
            # 删除临时 TIFF 文件
            if os.path.exists(temp_tif):
                os.remove(temp_tif)

# 遍历输入文件夹中的 NetCDF 文件
for file_name in os.listdir(input_folder):
    if file_name.endswith('.nc4'):
        # 构建完整的输入文件路径
        input_file = os.path.join(input_folder, file_name)
        nodata_value = get_nodata_value(input_file)
        
        # 获取变量属性
        variable_attributes = get_variable_attributes(input_file)
        
        # 获取 CRS 信息
        crs_info = get_crs_info(input_file)
        
        # 转换并重采样
        convert_and_resample(input_file, variable_attributes, nodata_value, crs_info)

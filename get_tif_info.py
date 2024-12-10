import rasterio
from rasterio.transform import rowcol

# 经纬度
lon, lat = 104.87, 31.79

# 打开 TIFF 文件
with rasterio.open('data/HWSD_1247/tif/AWC_CLASS.tif') as src:
    # 获取坐标参考系统（CRS）并打印
    crs = src.crs
    print(f"CRS: {crs}")

    # 获取仿射变换矩阵
    transform = src.transform

    # 使用仿射变换将经纬度转换为行列索引
    row, col = rowcol(transform, lon, lat)

    # 打印行列索引
    print(f"Row: {row}, Column: {col}")

    # 读取栅格数据并提取指定位置的值
    band_data = src.read(1)  # 假设使用第一个波段
    value = band_data[row, col]  # 获取对应行列的值

    print(f"Value at ({lon}, {lat}): {value}")

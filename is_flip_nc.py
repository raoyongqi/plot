import xarray as xr
import matplotlib.pyplot as plt

# 输入 NetCDF 文件路径
input_file = 'data/HWSD_1247/data/AWC_CLASS.nc4'

# 打开 NetCDF 文件
ds = xr.open_dataset(input_file)


print("Variables in the dataset:")
print(ds.data_vars)

# 选择要绘制的变量（假设选择第一个变量）
variable = list(ds.data_vars.keys())[0]
data = ds[variable]

# 打印数据的维度和坐标信息
print(f"Data dimensions: {data.dims}")
print(f"Data coordinates: {data.coords}")

# 绘制数据
# 假设数据是二维的
if len(data.dims) == 2:
    plt.figure(figsize=(10, 6))
    data.plot()  # 使用 xarray 的内置绘图方法
    plt.title(f'{variable} - 2D Plot')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

# 如果数据是三维的（例如时间、经纬度）
elif len(data.dims) == 3:
    # 选择第一个时间步长
    data_2d = data.isel(time=0)
    plt.figure(figsize=(10, 6))
    data_2d.plot()
    plt.title(f'{variable} - Time Step 0')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

# 如果数据是四维的（例如时间、层、经纬度）
elif len(data.dims) == 4:
    # 选择第一个时间步长和第一个层次
    data_2d = data.isel(time=0, level=0)
    plt.figure(figsize=(10, 6))
    data_2d.plot()
    plt.title(f'{variable} - Time Step 0, Level 0')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

else:
    print("Unsupported data dimensions for plotting.")

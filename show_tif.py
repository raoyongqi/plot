import rasterio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap

# 打开 TIFF 文件
tif_file = "C:/Users/r/Desktop/work12/data/HWSD_1247/tif/AWT_SOC.tif"

with rasterio.open(tif_file) as src:
    # 读取栅格数据
    data = src.read(1)
    
    # 获取无效值
    nodata = src.nodata
    
    # 将数据转换为一维数组，并移除无效值（如 NaN 或无效值）
    data_flat = data.flatten()
    data_flat = data_flat[~np.isnan(data_flat)]
    if nodata is not None:
        data_flat = data_flat[data_flat != nodata]
    
    # 计算数据的最小值和最大值，用于颜色映射
    vmin, vmax = np.min(data_flat), np.max(data_flat)
    
    # 创建颜色映射和归一化对象
    cmap = get_cmap('viridis')  # 选择颜色映射
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    # 绘制栅格数据
    fig, ax = plt.subplots(figsize=(10, 6))
    img = ax.imshow(data, cmap=cmap, norm=norm)
    
    # 添加颜色条
    cbar = plt.colorbar(img, ax=ax, orientation='horizontal', pad=0.2, label='Value')
    cbar.set_label('Value')
    
    # 显示栅格数据和坐标系
    crs = src.crs
    print(crs)
    plt.title('Raster Data Visualization with Colorbar')
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.show()

    # 生成箱线图
    plt.figure(figsize=(10, 6))
    plt.boxplot(data_flat, vert=False)
    plt.title('Boxplot of TIFF Data')
    plt.xlabel('Values')
    plt.show()

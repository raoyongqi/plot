import rasterio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap

# 打开 TIFF 文件
tif_file = "C:/Users/r/Desktop/work12/data/HWSD_1247/tif/AWT_SOC.tif"

with rasterio.open(tif_file) as src:

    data = src.read(1)
    
    nodata = src.nodata
    
    data_flat = data.flatten()
    data_flat = data_flat[~np.isnan(data_flat)]
    if nodata is not None:
        data_flat = data_flat[data_flat != nodata]
    
    vmin, vmax = np.min(data_flat), np.max(data_flat)
    
    cmap = get_cmap('viridis')  # 选择颜色映射
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    img = ax.imshow(data, cmap=cmap, norm=norm)
    
    cbar = plt.colorbar(img, ax=ax, orientation='horizontal', pad=0.2, label='Value')
    cbar.set_label('Value')
    
    crs = src.crs
    print(crs)
    plt.title('Raster Data Visualization with Colorbar')
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.boxplot(data_flat, vert=False)
    plt.title('Boxplot of TIFF Data')
    plt.xlabel('Values')
    plt.show()

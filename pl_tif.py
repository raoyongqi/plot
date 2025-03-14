import matplotlib.pyplot as plt
import numpy as np
import rasterio
import geopandas as gpd
import cartopy.crs as ccrs
import os

albers_proj = ccrs.AlbersEqualArea(
    central_longitude=105,
    central_latitude=35,
    standard_parallels=(25, 47)
)

fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={'projection': albers_proj})

geojson_file_path = '中华人民共和国.json'
gdf_geojson = gpd.read_file(geojson_file_path)

if gdf_geojson.crs != albers_proj:

    gdf_geojson = gdf_geojson.to_crs(albers_proj)

gdf_geojson.plot(ax=ax, edgecolor='black', facecolor='white', alpha=0.5, label='GeoJSON Data')

tif_file = 'data/cropped_result/tiff/cropped_predicted_rf.tif'

file_name = os.path.basename(tif_file)
title = os.path.splitext(file_name)[0]

with rasterio.open(tif_file) as src:
    data = src.read(1)
    no_data_value = src.nodata
    
    if no_data_value is not None:
        data = np.where(data == no_data_value, np.nan, data)
    
    transform = src.transform
    bounds = [transform * (0, 0), transform * (src.width, src.height)]
    extent = [bounds[0][0], bounds[1][0], bounds[1][1], bounds[0][1]]
    
    vmin, vmax = np.nanmin(data), np.nanmax(data)
    print(f"TIFF 文件 {tif_file} 的最小值: {vmin}")
    print(f"TIFF 文件 {tif_file} 的最大值: {vmax}")
    
    cmap = plt.get_cmap('viridis').reversed()
    
    im = ax.imshow(data, cmap=cmap, interpolation='none', extent=extent, transform=ccrs.PlateCarree(), alpha=1)
    
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.05)
    cbar.set_label('Pixel Value')

plt.title(f'{title}')

ax.set_xlabel('Easting (meters)')
ax.set_ylabel('Northing (meters)')

gridlines = ax.gridlines(draw_labels=True, color='gray', linestyle='--', alpha=0.5)
gridlines.xlabel_style = {'size': 10}
gridlines.ylabel_style = {'size': 10}

gridlines.top_labels = False
gridlines.right_labels = False

output_file_path = f'data/{title}.png'
plt.savefig(output_file_path, dpi=300, bbox_inches='tight')

plt.show()

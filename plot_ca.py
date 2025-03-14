import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import pandas as pd  # 导入 pandas

geojson_file_path = 'china.json'
gdf_geojson = gpd.read_file(geojson_file_path)


file_path = 'data/climate_soil_tif.xlsx'

points_df = pd.read_excel(file_path)

points_count = points_df.groupby(['LON', 'LAT']).size().reset_index(name='Count')

print(points_count.head())

print(f"共有不同经纬度点: {len(points_count)}")

points_gdf = gpd.GeoDataFrame(
    points_count,
    geometry=gpd.points_from_xy(points_count['LON'], points_count['LAT']), 
    crs='EPSG:4326'
)

print(f"去重后样点数量: {len(points_gdf)}")

albers_proj = ccrs.AlbersEqualArea(
    central_longitude=105,
    central_latitude=35,
    standard_parallels=(25, 47)
)

fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={'projection': albers_proj})

if gdf_geojson.crs != albers_proj:
    gdf_geojson = gdf_geojson.to_crs(albers_proj)

points_gdf = points_gdf.to_crs(albers_proj)

num_points = len(points_gdf)

print(f"样点数量: {num_points}")

gdf_geojson.plot(ax=ax, edgecolor='black', facecolor='white', label='GeoJSON Data')


points_gdf.plot(ax=ax, color='red', marker='o', label='Sample Points', markersize=20)

plt.title('Sample Points')

legend_patches = [

    mpatches.Patch(color='red', label='Sample Points'),
]
plt.legend(handles=legend_patches)

gridlines = ax.gridlines(draw_labels=True, color='gray', linestyle='--', alpha=0.5)
gridlines.xlabel_style = {'size': 10}
gridlines.ylabel_style = {'size': 10}

gridlines.top_labels = False
gridlines.right_labels = False

output_file_path = 'data/sample.png'
plt.savefig(output_file_path, dpi=300, bbox_inches='tight')

plt.show()
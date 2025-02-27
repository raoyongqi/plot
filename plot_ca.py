import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import pandas as pd  # 导入 pandas

# 读取 GeoJSON 数据
geojson_file_path = 'china.json'
gdf_geojson = gpd.read_file(geojson_file_path)

# # 读取 Shapefile 数据
# shp_file_path = 'clipped_data.geojson'
# gdf_shp = gpd.read_file(shp_file_path)

# # 筛选出值等于 10 的 Grasslands (草地)
# if 'value' in gdf_shp.columns:
#     grasslands_gdf = gdf_shp[gdf_shp['value'] == 10]
# else:
#     raise KeyError("The column 'value' does not exist in the shapefile data.")

# 读取 Excel 文件
file_path = 'data/climate_soil_tif.xlsx'  # 替换为您的文件路径

points_df = pd.read_excel(file_path)  # 第一个工作表


# 统计每个经纬度的数量
points_count = points_df.groupby(['LON', 'LAT']).size().reset_index(name='Count')

# 输出统计信息
print(points_count.head())  # 查看统计后的前几行数据
print(f"共有不同经纬度点: {len(points_count)}")

# 只保留经纬度唯一的点用于绘图
points_gdf = gpd.GeoDataFrame(
    points_count,  # 使用统计后的数据
    geometry=gpd.points_from_xy(points_count['LON'], points_count['LAT']), 
    crs='EPSG:4326'  # WGS84 坐标系
)

# 输出样点数量（去重后）
print(f"去重后样点数量: {len(points_gdf)}")


# 定义 Albers 投影坐标系
albers_proj = ccrs.AlbersEqualArea(
    central_longitude=105,
    central_latitude=35,
    standard_parallels=(25, 47)
)

# 创建绘图对象
fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={'projection': albers_proj})

# 转换 GeoJSON 数据的坐标系到自定义投影坐标系
if gdf_geojson.crs != albers_proj:
    gdf_geojson = gdf_geojson.to_crs(albers_proj)

# # 转换 Shapefile 数据的坐标系到自定义投影坐标系
# if grasslands_gdf.crs != albers_proj:
#     grasslands_gdf = grasslands_gdf.to_crs(albers_proj)

# 转换样点的坐标系到自定义投影坐标系
points_gdf = points_gdf.to_crs(albers_proj)
# 统计样点数量
num_points = len(points_gdf)
print(f"样点数量: {num_points}")

# 绘制转换后的 GeoJSON 数据
gdf_geojson.plot(ax=ax, edgecolor='black', facecolor='white', label='GeoJSON Data')

# 绘制筛选后的 Grasslands 数据
# grasslands_gdf.plot(ax=ax, edgecolor='none', facecolor='green', linewidth=2, alpha=0.5, label='Grasslands (value=10)')

# 绘制样点
points_gdf.plot(ax=ax, color='red', marker='o', label='Sample Points', markersize=20)

# 添加标题
plt.title('Sample Points')
# 统计样点数量



legend_patches = [

    mpatches.Patch(color='red', label='Sample Points'),
]
plt.legend(handles=legend_patches)

# 添加经纬度网格线
gridlines = ax.gridlines(draw_labels=True, color='gray', linestyle='--', alpha=0.5)
gridlines.xlabel_style = {'size': 10}
gridlines.ylabel_style = {'size': 10}
# 隐藏右边和上边的网格线标签
gridlines.top_labels = False
gridlines.right_labels = False

# 保存图形到文件
output_file_path = 'data/sample.png'
plt.savefig(output_file_path, dpi=300, bbox_inches='tight')

# 显示图形
plt.show()
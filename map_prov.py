import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from sklearn.preprocessing import MinMaxScaler

# Step 1: 读取Excel文件
df = pd.read_excel('data/climate_soil.xlsx')  # 请确认文件路径和文件名

# Step 2: 检查数据是否加载正确
print("数据的前几行：")
print(df.head())  # 打印前几行数据查看
print("\n数据列名：")
print(df.columns)  # 打印所有列名，确保包含 'LON', 'LAT', 和 'RATIO'

# 确保列名没有多余的空格
df.columns = df.columns.str.strip()  # 去除列名的前后空格

# Step 3: 按经纬度分组，计算病害值的平均值
average_disease = df.groupby(['LON', 'LAT'])['RATIO'].mean().reset_index()

# Step 4: 读取GeoJSON数据
geojson_file_path = '中华人民共和国.json'  # 请确保这个文件路径正确
gdf_geojson = gpd.read_file(geojson_file_path)

# Step 5: 查看GeoJSON数据的类型和属性
print("\nGeoJSON 数据类型:")
print(gdf_geojson.geom_type)  # 查看几何类型
print("\nGeoJSON 数据的前几行:")
print(gdf_geojson.head())  # 打印前几行，查看其包含的属性

# Step 6: 提取每个多边形的信息
# 计算每个多边形的面积
gdf_geojson['area'] = gdf_geojson.geometry.area

# 计算每个多边形的周长
gdf_geojson['perimeter'] = gdf_geojson.geometry.length

# Step 7: 筛选出特定的GeoJSON数据
# 例如：根据某个属性值筛选，假设筛选面积大于10000的多边形
filtered_gdf = gdf_geojson[~gdf_geojson['name'].isin(['西藏自治区','青海省'])]

# Step 8: 绘制筛选后的地图
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

# 绘制筛选后的GeoJSON地图
filtered_gdf.plot(ax=ax, color='lightblue', edgecolor='black')

# 添加每个多边形的信息作为标签
for idx, row in filtered_gdf.iterrows():
    # 获取多边形的几何信息
    polygon = row['geometry']
    # 获取面积和周长
    area = row['area']
    perimeter = row['perimeter']
    
    # 获取多边形的中心点坐标，用于放置标签
    x, y = polygon.centroid.coords[0]
    
    # 添加标签到地图
    ax.text(x, y, f"Area: {area:.2f}\nPerimeter: {perimeter:.2f}", fontsize=8, ha='center', color='black')

# Step 9: 添加标题和其他设置
ax.set_title('筛选后的GeoJSON 多边形信息', fontdict={'fontsize': '15', 'fontweight' : '3'})
ax.set_axis_off()

# 显示地图
plt.show()

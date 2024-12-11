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

# Step 5: 归一化植物病害数据（RATIO）
scaler = MinMaxScaler(feature_range=(0, 1))
average_disease['normalized'] = scaler.fit_transform(average_disease[['RATIO']])

# Step 6: 创建GeoDataFrame用于绘制病害点
geometry = [Point(lon, lat) for lon, lat in zip(average_disease['LON'], average_disease['LAT'])]
gdf_disease = gpd.GeoDataFrame(average_disease, geometry=geometry)

# Step 7: 绘制地图
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

# 绘制GeoJSON地图
gdf_geojson.plot(ax=ax, color='lightgray', edgecolor='black')

# 根据植物病害数据绘制病害点，并根据归一化后的病害值着色
gdf_disease.plot(column='normalized', cmap='YlOrRd', ax=ax, markersize=50, legend=True)

# 添加标题和其他设置
ax.set_title('植物病害分布图', fontdict={'fontsize': '5', 'fontweight' : '3'})
ax.set_axis_off()

# 显示地图
plt.show()

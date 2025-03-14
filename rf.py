import os
import rasterio
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 读取tif文件并提取数据和元数据，包括经纬度信息
def read_tif_with_coords(file_path):
    with rasterio.open(file_path) as src:
        data = src.read(1)  # 读取第一波段的数据
        profile = src.profile
        transform = src.transform  # 获取仿射变换信息
        width = src.width
        height = src.height

        # 生成所有像素的行列号
        rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

        # 将行列号转换为地理坐标（经纬度）
        xs, ys = rasterio.transform.xy(transform, rows, cols)


    return data, profile, np.array(xs), np.array(ys)

# 保存预测结果为tif文件

# 获取特征名称
def get_feature_name(file_name):
    base_name = os.path.basename(file_name)
    feature_name = base_name.replace('cropped_', '').replace('.tif', '')
    return feature_name

# 1. 读取Excel文件
file_path = 'data/climate_soil_tif.xlsx'  # 替换为你的文件路径
data = pd.read_excel(file_path)
data.columns = data.columns.str.lower()
# 2. 筛选特征列：以 '_resampled' 结尾，'wc' 开头（不区分大小写），以及 'LON' 和 'LAT' 列
feature_columns = [col for col in data.columns if col !='ratio']


# 3. 分离特征变量和目标变量
X = data[feature_columns]
y = data['ratio']  # 目标变量

# 4. 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. 初始化并训练随机森林回归模型
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 6. 预测并评估模型
y_pred = rf.predict(X_test)

# 7. 评估模型性能
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 输出结果
print(f"均方误差 (MSE): {mse:.4f}")
print(f"R² 得分: {r2:.4f}")

tif_folder1 = 'data/cropped/tiff'  # 替换为实际tif文件夹路径
tif_folder2 = 'data/cropped/soil_tiff'  # 替换为第二个tif文件夹路径

# 获取两个文件夹中的所有 .tif 文件
tif_files = []

tif_files += [os.path.join(tif_folder1, f) for f in os.listdir(tif_folder1) if f.endswith('.tif')]

tif_files += [os.path.join(tif_folder2, f) for f in os.listdir(tif_folder2) if f.endswith('.tif')]

output_folder = 'data/result'  # 替换为实际输出文件夹路径
data_list = []
profiles = []

for i, file in enumerate(tif_files):
    data, profile, xs, ys = read_tif_with_coords(file)
    data_list.append(data)
    profiles.append(profile)
    if "elev" in file:  # 根据文件名判断
        print(f"Elev data is from file: {file}")
    if i == 0:  # 只保存第一个tif的经纬度信息
        lons, lats = xs, ys

data_stack = np.stack(data_list, axis=-1)
rows, cols, bands = data_stack.shape
data_2d = data_stack.reshape((rows * cols, bands))

coords_2d = np.stack((lons.flatten(), lats.flatten()), axis=1)
data_with_coords = np.hstack((coords_2d, data_2d))

feature_names = ['LON', 'LAT'] + [get_feature_name(f) for f in tif_files]
df = pd.DataFrame(data_with_coords, columns=feature_names)

model_feature_names = feature_columns


print(model_feature_names)

df.columns = df.columns.str.lower()

model_feature_names = [col.lower() for col in model_feature_names]


df = df[model_feature_names]

y_pred = rf.predict(df)

y_pred_2d = y_pred.reshape((rows, cols))

os.makedirs(output_folder, exist_ok=True)
def save_tif(file_path, data, profile):
    with rasterio.open(file_path, 'w', **profile) as dst:
        dst.write(data, 1)

model_name = 'data/result'

output_file = os.path.join(output_folder, f'predicted_rf.tif')
save_tif(output_file, y_pred_2d, profiles[0])

print(f"预测结果已保存到 {output_file}")

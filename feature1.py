# 安装 featurewiz（如果尚未安装）
# pip install featurewiz
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from quantile_forest import RandomForestQuantileRegressor
import joblib  # Used for saving the model
import os  # Used for file and directory handling
import re
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib  # 用于保存模型
import os  # 用于处理文件和目录
from boruta import BorutaPy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


file_path = 'data/climate_soil_tif.xlsx'  # 替换为你的文件路径
data = pd.read_excel(file_path)
# data.drop(columns=['Province', 'City', 'District'], inplace=True)

data.columns = data.columns.str.lower()

# 找出所有列名中包含下划线的列，并检查前后部分是否相同
data.columns = [col.replace('_resampled', '') if '_resampled' in col else col for col in data.columns]
data.columns = [col.replace('wc2.1_5m_', '') if col.startswith('wc2.1_5m_') else col for col in data.columns]
new_columns = []
for col in data.columns:
    if '_' in col:  # 如果列名中有下划线
        parts = col.split('_')  # 用下划线拆分列名
        if len(parts) > 1 and parts[0] == parts[-1]:  # 如果前后部分相同
            # 将拆分后的第一部分和最后一部分合并
            new_columns.append('_'.join(parts[:1]))  # 取前两个部分作为列名
        elif len(parts) > 2 and parts[1] == parts[-1]:  # 如果前后部分相同
            # 将拆分后的第一部分和最后一部分合并
            new_columns.append('_'.join(parts[:2]))  # 取前两个部分作为列名
        elif len(parts) > 3 and parts[2] == parts[-1]:  # 如果前后部分相同
            # 将拆分后的第一部分和最后一部分合并
            new_columns.append('_'.join(parts[:2]))  # 取前两个部分作为列名
        else:
            new_columns.append(col)  # 否则保留原列名
    else:
        new_columns.append(col)  # 如果没有下划线，直接保留原列名

# 更新 DataFrame 的列名
data.columns = new_columns
# 2. 筛选特征列

# 将所有 'prec_*' 列加总为 MAP
data['MAP'] = data.filter(like='prec_').sum(axis=1)
data['WIND'] = data.filter(like='wind_').mean(axis=1)
data['MAX_MAT'] = data.filter(like='tmax_').mean(axis=1)
data['MIN_MAT'] = data.filter(like='tmin_').mean(axis=1)
data['AVG_MAT'] = data.filter(like='tavg_').mean(axis=1)

data['SRAD'] = data.filter(like='srad_').mean(axis=1)
data['VAPR'] = data.filter(like='vapr_').mean(axis=1)
data['TSEA'] = data['bio_4']
data['PSEA'] =data['bio_15']

# 删除 'prec_*' 列
data = data.drop(columns=data.filter(like='prec_').columns)
data = data.drop(columns=data.filter(like='srad_').columns)
data = data.drop(columns=data.filter(like='tmax_').columns)
data = data.drop(columns=data.filter(like='tmin_').columns)
data = data.drop(columns=data.filter(like='tavg_').columns)
data = data.drop(columns=data.filter(like='vapr_').columns)

data = data.drop(columns=data.filter(like='wind_').columns)
data = data.drop(columns=data.filter(like='bio_').columns)
data.columns = data.columns.str.upper()
columns_to_drop = ['MU_GLOBAL', 'REF_DEPTH', 'LANDMASK', 'ROOTS', 'ISSOIL']
existing_columns_to_drop = [col for col in columns_to_drop if col in data.columns]

# 仅删除存在的列
data = data.drop(columns=existing_columns_to_drop)
feature_columns = [col for col in data.columns]


df = data[feature_columns]
from featurewiz import featurewiz
import pandas as pd

# 1. 读取数据

# 2. 设定目标变量
target_column = 'RATIO'  # 替换为你的目标列名称

# 3. 运行 featurewiz 进行特征选择
selected_features, transformed_train = featurewiz(
    df, 
    target=target_column, 
    corr_limit=0.70,  # 相关性阈值
    verbose=2,
          feature_engg='target', # 不进行额外的特征工程
    category_encoders=None # 不进行类别变量编码
)

# 4. 输出选出的重要特征
print("Selected Features:", selected_features)

# 5. 只保留选出的特征和目标列
df_selected = df[selected_features + [target_column]]

# 6. 保存处理后的数据
df_selected.to_csv("filtered_data.csv", index=False)
print("Filtered data saved as filtered_data.csv")

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


file_path = 'data/climate_soil_tif.xlsx'
data = pd.read_excel(file_path)

data.columns = data.columns.str.lower()

data.columns = [col.replace('_resampled', '') if '_resampled' in col else col for col in data.columns]
data.columns = [col.replace('wc2.1_5m_', '') if col.startswith('wc2.1_5m_') else col for col in data.columns]
new_columns = []
for col in data.columns:
    if '_' in col: 
        parts = col.split('_')
        if len(parts) > 1 and parts[0] == parts[-1]:
            new_columns.append('_'.join(parts[:1]))
        elif len(parts) > 2 and parts[1] == parts[-1]:
            new_columns.append('_'.join(parts[:2]))
        elif len(parts) > 3 and parts[2] == parts[-1]:
            new_columns.append('_'.join(parts[:2]))
        else:
            new_columns.append(col)
    else:
        new_columns.append(col)

data.columns = new_columns

data['MAP'] = data.filter(like='prec_').sum(axis=1)
data['WIND'] = data.filter(like='wind_').mean(axis=1)
data['MAX_MAT'] = data.filter(like='tmax_').mean(axis=1)
data['MIN_MAT'] = data.filter(like='tmin_').mean(axis=1)
data['AVG_MAT'] = data.filter(like='tavg_').mean(axis=1)

data['SRAD'] = data.filter(like='srad_').mean(axis=1)
data['VAPR'] = data.filter(like='vapr_').mean(axis=1)
data['TSEA'] = data['bio_4']
data['PSEA'] =data['bio_15']

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

data = data.drop(columns=existing_columns_to_drop)
feature_columns = [col for col in data.columns]


df = data[feature_columns]
from featurewiz import featurewiz
import pandas as pd


target_column = 'RATIO'

selected_features, transformed_train = featurewiz(
    df, 
    target=target_column, 
    corr_limit=0.70,
    verbose=2,
    feature_engg='target',
    category_encoders=None
)

print("Selected Features:", selected_features)

df_selected = df[selected_features + [target_column]]

df_selected.to_csv("filtered_data.csv", index=False)
print("Filtered data saved as filtered_data.csv")

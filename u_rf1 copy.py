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
            new_columns.append(col)  # 否则保留原列名
    else:
        new_columns.append(col)  # 如果没有下划线，直接保留原列名

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

data = data.drop(columns=existing_columns_to_drop)
feature_columns = [col for col in data.columns if col != 'RATIO']


X = data[feature_columns]
y = data['RATIO']  # 目标变量

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf = RandomForestRegressor(n_estimators=100, random_state=42)

feat_selector = BorutaPy(rf, n_estimators='auto',max_iter=10, alpha=0.05, random_state=42, verbose=2)


feat_selector.fit(X_train.values, y_train.values)


sorted_features = [feature for _, feature in sorted(zip(feat_selector.ranking_, feature_columns))]
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

results_sorted = pd.DataFrame(columns=['number', 'Model', 'MSE', 'R2'])

for number in range(17, len(sorted_features) + 1):

    X = data[sorted_features[:number]]
    y = data['RATIO']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_model = RandomForestRegressor(random_state=42)
    gb_model = GradientBoostingRegressor(random_state=42)
    qr_rf_model = RandomForestQuantileRegressor(random_state=42)


    rf_param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, None]}
    gb_param_grid = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7]}
    qr_rf_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None]
    }


    rf_grid_search = GridSearchCV(rf_model, rf_param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    gb_grid_search = GridSearchCV(gb_model, gb_param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    qr_rf_grid_search = GridSearchCV(qr_rf_model, qr_rf_param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

    rf_grid_search.fit(X_train, y_train)
    gb_grid_search.fit(X_train, y_train)
    qr_rf_grid_search.fit(X_train, y_train)

    rf_model.fit(X_train, y_train)
    gb_model.fit(X_train, y_train)
    qr_rf_model.fit(X_train, y_train)

    rf_pred = rf_model.predict(X_test)
    gb_pred = gb_model.predict(X_test)
    qr_rf_pred = qr_rf_model.predict(X_test)

    rf_pred_grid = rf_grid_search.best_estimator_.predict(X_test)
    gb_pred_grid = gb_grid_search.best_estimator_.predict(X_test)
    qr_rf_pred_grid = qr_rf_grid_search.best_estimator_.predict(X_test)

    def evaluate_model(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return round(mse, 4), round(r2,4)

    rf_mse, rf_r2 = evaluate_model(y_test, rf_pred)
    gb_mse, gb_r2 = evaluate_model(y_test, gb_pred)
    qr_rf_mse, qr_rf_r2 = evaluate_model(y_test, qr_rf_pred)

    rf_mse_grid, rf_r2_grid = evaluate_model(y_test, rf_pred_grid)
    gb_mse_grid, gb_r2_grid = evaluate_model(y_test, gb_pred_grid)
    qr_rf_mse_grid, qr_rf_r2_grid = evaluate_model(y_test, qr_rf_pred_grid)

    results = pd.DataFrame({
        'number': [number] * 6,
        'Model': [
            'RandomForest (No GridSearch)', 'RandomForest (With GridSearch)',
            'GradientBoosting (No GridSearch)', 'GradientBoosting (With GridSearch)',
            'QuantileRegressor (No GridSearch)', 'QuantileRegressor (With GridSearch)'
        ],
        'MSE': [rf_mse, rf_mse_grid, gb_mse, gb_mse_grid, qr_rf_mse, qr_rf_mse_grid],
        'R2': [rf_r2, rf_r2_grid, gb_r2, gb_r2_grid, qr_rf_r2, qr_rf_r2_grid]
    })
    results = results.sort_values(by='R2', ascending=False)

    results_sorted = pd.concat([results_sorted, results], ignore_index=True)
    print(results)
results_sorted.to_csv('model_evaluation_results.csv', index=False)

latex_table = results_sorted.to_latex(index=False)
with open('table_results_sorted.tex', 'w') as f:
    f.write(latex_table)

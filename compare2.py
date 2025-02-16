import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from boruta import BorutaPy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import time
import xgboost as xgb
import lightgbm as lgb  
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from matplotlib import rcParams

file_path = 'data/climate_soil.xlsx'  # 替换为你的文件路径
data = pd.read_excel(file_path)

data.columns = data.columns.str.lower()

data.columns = [col.replace('_resampled', '') if '_resampled' in col else col for col in data.columns]
data.columns = [col.replace('wc2.1_5m_', '') if col.startswith('wc2.1_5m_') else col for col in data.columns]
new_columns = []
for col in data.columns:
    if '_' in col:  # 如果列名中有下划线
        parts = col.split('_')  # 用下划线拆分列名
        if len(parts) > 1 and parts[0] == parts[-1]:  
            new_columns.append('_'.join(parts[:1]))  
        elif len(parts) > 2 and parts[1] == parts[-1]: 
            # 将拆分后的第一部分和最后一部分合并
            new_columns.append('_'.join(parts[:2]))  
        elif len(parts) > 3 and parts[2] == parts[-1]:  
            # 将拆分后的第一部分和最后一部分合并
            new_columns.append('_'.join(parts[:2]))  
        else:
            new_columns.append(col)  
    else:
        new_columns.append(col)  

data.columns = new_columns

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
data = data.drop(columns=['MU_GLOBAL','REF_DEPTH', 'LANDMASK', 'ROOTS', 'ISSOIL'])
feature_columns = [col for col in data.columns if col != 'RATIO']


X = data[feature_columns]
y = data['RATIO']  # 目标变量


# 4. 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 5. 初始化并训练随机森林回归模型
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# 6. 使用 Boruta 进行特征选择
feat_selector = BorutaPy(rf, n_estimators='auto',max_iter=10, alpha=0.05, random_state=42, verbose=2)
# # 7. 获取重要特征并选择前17个

feat_selector.fit(X_train.values, y_train.values)
# 按照ranking_的顺序排序特征名
sorted_features = [feature for _, feature in sorted(zip(feat_selector.ranking_, feature_columns))]


X = data[[*sorted_features[:17]]]
y =  data['RATIO']  # 目标变量  # 替换为你的目标变量


# 4. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


start_time = time.time()
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse')
xgb_model.fit(X_train, y_train)
xgb_time = time.time() - start_time

y_pred_xgb = xgb_model.predict(X_test)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

print(f"XGBoost Mean Squared Error (MSE): {mse_xgb:.4f}")
print(f"XGBoost R² Score: {r2_xgb:.4f}")
print(f"XGBoost Training Time: {xgb_time:.2f} seconds")

# 11. Train and evaluate the Random Forest model
start_time = time.time()
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_time = time.time() - start_time

y_pred_rf = rf_model.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Random Forest Mean Squared Error (MSE): {mse_rf:.4f}")
print(f"Random Forest R² Score: {r2_rf:.4f}")
print(f"Random Forest Training Time: {rf_time:.2f} seconds")

# 12. Train and evaluate the LightGBM model
start_time = time.time()
lgb_model = lgb.LGBMRegressor()
lgb_model.fit(X_train, y_train)
lgb_time = time.time() - start_time

y_pred_lgb = lgb_model.predict(X_test)
mse_lgb = mean_squared_error(y_test, y_pred_lgb)
r2_lgb = r2_score(y_test, y_pred_lgb)

print(f"LightGBM Mean Squared Error (MSE): {mse_lgb:.4f}")
print(f"LightGBM R² Score: {r2_lgb:.4f}")
print(f"LightGBM Training Time: {lgb_time:.2f} seconds")

# 13. Prepare data for plotting
models = [ 'XGBoost', 'Random Forest', 'LightGBM']
r2_scores = [ r2_xgb, r2_rf, r2_lgb]
times = [ xgb_time, rf_time, lgb_time]

# Create DataFrame for sorting
results_df = pd.DataFrame({
    'Model': models,
    'R2 Score': r2_scores,
    'Training Time': times
})

# Sort by R² score from high to low
results_df_sorted_r2 = results_df.sort_values(by='R2 Score', ascending=False)

# Sort by Training Time from low to high
results_df_sorted_time = results_df.sort_values(by='Training Time', ascending=True)

# Plot performance-time graph
plt.figure(figsize=(14, 7))

# Plot R² Scores sorted from high to low
plt.subplot(1, 2, 1)
plt.bar(results_df_sorted_r2['Model'], results_df_sorted_r2['R2 Score'], color=['blue', 'green', 'red', 'purple'])
plt.xlabel('Model')
plt.ylabel('R² Score')
plt.title('R² Score Comparison (High to Low)')

# Plot Training Time sorted from low to high
plt.subplot(1, 2, 2)
plt.bar(results_df_sorted_time['Model'], results_df_sorted_time['Training Time'], color=['blue', 'green', 'red', 'purple'])
plt.xlabel('Model')
plt.ylabel('Training Time (seconds)')
plt.title('Training Time Comparison (Low to High)')

plt.tight_layout()
plt.savefig('data/model_performance_comparison.png')  # Save plot to file
plt.show()


print("Model Performance Comparison:")

# Print R² scores sorted from high to low
print("\nR² Scores (High to Low):")
print(results_df_sorted_r2[['Model', 'R2 Score']])

# Print training times sorted from low to high
results_df_combined = results_df.sort_values(by='R2 Score', ascending=False)

# Save the combined table to LaTeX
combined_tex_path = "data/model_performance.tex"
results_df_combined.to_latex(
    combined_tex_path,
    index=False,
    caption="Model Performance Comparison (R² Score and Training Time)",
    label="tab:model_performance",
    float_format="%.4f"
)

# Output path for reference
print(f"Combined Model Performance table saved to LaTeX: {combined_tex_path}")
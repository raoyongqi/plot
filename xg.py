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
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib
import os
import shap
import numpy as np


file_path = 'data/climate_soil.xlsx'  # 替换为你的文件路径
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
data['WIND'] = data.filter(like='wind_').sum(axis=1)
data['MAX MAT'] = data.filter(like='tmax_').mean(axis=1)
data['MIN MAT'] = data.filter(like='tmin_').mean(axis=1)
data['AVG MAT'] = data.filter(like='tavg_').mean(axis=1)

data['SARD'] = data.filter(like='srad_').mean(axis=1)
data['VAPR'] = data.filter(like='vapr_').sum(axis=1)

data = data.drop(columns=data.filter(like='prec_').columns)
data = data.drop(columns=data.filter(like='srad_').columns)
data = data.drop(columns=data.filter(like='tmax_').columns)
data = data.drop(columns=data.filter(like='tmin_').columns)
data = data.drop(columns=data.filter(like='tavg_').columns)
data = data.drop(columns=data.filter(like='vapr_').columns)

data = data.drop(columns=data.filter(like='wind_').columns)
data = data.drop(columns=data.filter(like='bio_').columns)
data.columns = data.columns.str.upper()
data = data.drop(columns=['REF_DEPTH', 'LANDMASK', 'ROOTS', 'ISSOIL'])
data.columns = data.columns.str.replace('_', ' ')

feature_columns = [col for col in data.columns if col != 'RATIO']

X = data[feature_columns]

y = data['RATIO']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf = RandomForestRegressor(n_estimators=100, random_state=42)

feat_selector = BorutaPy(rf, n_estimators='auto',max_iter=10, alpha=0.05, random_state=42, verbose=2)

feat_selector.fit(X_train.values, y_train.values)

sorted_features = [feature for _, feature in sorted(zip(feat_selector.ranking_, feature_columns))]


X = data[[*sorted_features[:17]]]
y =  data['RATIO']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgb_model = XGBRegressor(n_estimators=100, random_state=42, learning_rate=0.1, max_depth=6)

xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)

os.makedirs('data/model', exist_ok=True)

joblib.dump(xgb_model, 'data/model/xgboost_model.pkl')

explainer = shap.Explainer(xgb_model, X_train)

shap_values = explainer(X_test)

shap_values_data = shap_values.values

shap_importance = pd.DataFrame(shap_values_data, columns=X_train.columns) 

mean_abs_shap = shap_importance.abs().mean(axis=0)

feature_importance = pd.DataFrame(list(zip(mean_abs_shap.index, mean_abs_shap.values)), columns=["Feature", "SHAP Value"])

feature_importance = feature_importance.sort_values(by="SHAP Value", ascending=False)


X = data[feature_columns]
y = data['RATIO']  # 目标变量

# 4. 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. 初始化并训练XGBoost回归模型
xgb_model = XGBRegressor(n_estimators=100, random_state=42, learning_rate=0.1, max_depth=6)
xgb_model.fit(X_train, y_train)

# 6. 预测并评估模型
y_pred = xgb_model.predict(X_test)

# 7. 评估模型性能
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 输出结果
print(f"均方误差 (MSE): {mse:.4f}")
print(f"R² 得分: {r2:.4f}")

# 8. 确保保存路径存在
os.makedirs('data/model', exist_ok=True)

# 9. 保存模型
joblib.dump(xgb_model, 'data/model/xgboost_model.pkl')

# 12. 计算 SHAP 值并绘制 SHAP 图
explainer = shap.Explainer(xgb_model, X_train)
shap_values = explainer(X_test)

# 提取 SHAP 值
shap_values_data = shap_values.values  # 获取 SHAP 值数组
shap_importance = pd.DataFrame(shap_values_data, columns=X_train.columns)  # 创建 DataFrame

# 计算每个特征的平均绝对 SHAP 值
mean_abs_shap = shap_importance.abs().mean(axis=0)

# 计算每个特征的标准误差 (stderr)
stderr_shap = shap_importance.std(axis=0) / np.sqrt(shap_importance.shape[0])

# 计算每个特征的 95% 置信区间 (CI)
ci_lower = shap_importance.quantile(0.025, axis=0)  # 2.5% 分位数
ci_upper = shap_importance.quantile(0.975, axis=0)  # 97.5% 分位数

# 创建一个 DataFrame，包含特征、平均 SHAP 值、标准误差和 95% CI
feature_importance = pd.DataFrame({
    "Feature": mean_abs_shap.index,
    "SHAP Value": mean_abs_shap.values,
    "Standard Error (stderr)": stderr_shap.values,
    "CI Lower (95%)": ci_lower.values,
    "CI Upper (95%)": ci_upper.values
})

# 排序并输出影响最大的变量
feature_importance = feature_importance.sort_values(by="SHAP Value", ascending=False)
print("对模型影响最大的变量:")

# 绘制 SHAP 图
shap.summary_plot(shap_values, X_test, show=False, cmap='PiYG')

# 保存 SHAP 图
plt.savefig('data/model/shap_summary_plot.png', bbox_inches='tight')  # 使用 bbox_inches='tight' 确保内容完整

# 关闭图形
plt.close()
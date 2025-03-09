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

# 1. 读取Excel文件
# 1. 读取Excel文件
file_path = 'data/hand_data.xlsx'  # 替换为你的文件路径
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
data = data.drop(columns=existing_columns_to_drop)
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


X = data[[*sorted_features[:16]]]
y =  data['RATIO']  # 目标变量  # 替换为你的目标变量

rf = RandomForestRegressor(random_state=42)

rf.fit(X, y)

feature_importances = rf.feature_importances_

features = X.columns

# 创建一个 DataFrame 用于排序
importance_df = pd.DataFrame({
    "Response":"Plant Disease",
    "Feature": features,
    "Importance": feature_importances
}).sort_values(by="Importance", ascending=False)

importance_df["Feature"] = importance_df["Feature"].apply(lambda x: x.replace('_', ' '))

category_dict = {
    'SRAD': 'Climate',
    'LON': 'Geography',
    'S SAND': 'Soil',
    'LAT': 'Geography',
    'VAPR': 'Climate',
    'WIND': 'Climate',
    'ELEV': 'Geography',
    'MAX MAT': 'Climate',
    'TSEA': 'Climate',
    'PSEA': 'Climate',
    'HAND': 'Geography',

    'MAP': 'Climate',
    'AVG MAT': 'Climate',
    'MIN MAT': 'Climate',
    'MU GLOBAL': 'Soil',
    'S CLAY': 'Soil',
    'T SAND': 'Soil',
    'T REF BULK': 'Soil',
    'S REF BULK': 'Soil',

    'T BULK DEN': 'Soil',
    'T GRAVEL': 'Soil'
}

importance_df['Category'] = importance_df['Feature'].map(category_dict)
print(importance_df['Category'])
importance_df = importance_df[importance_df["Importance"] >= 0.03]
print(importance_df["Importance"])
importance_df = importance_df[["Response",'Feature', 'Category', "Importance"]]

latex_output = importance_df.to_latex(index=False, float_format="%.2f", caption="Feature Importance Table", label="tab:feature_importance")

output_file = "feature_importance1.tex"

with open(output_file, "w") as f:

    f.write(latex_output)


output_file = "feature_importance1.xlsx"
importance_df.to_excel(output_file, float_format="%.2f",index=False)
# # 8. 使用选择的特征重新训练模型
# X_train_filtered = X_train[top_17_features]
# X_test_filtered = X_test[top_17_features]

# rf.fit(X_train_filtered, y_train)
# y_pred = rf.predict(X_test_filtered)

# # 9. 评估模型
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# print(f"均方误差 (MSE): {mse:.4f}")
# print(f"R² 得分: {r2:.4f}")

# # 10. 保存模型
# os.makedirs('data/model', exist_ok=True)
# joblib.dump(rf, 'data/model/random_forest_model_top_17.pkl')

# # 11. 保存变量重要性
# feature_importances = rf.feature_importances_
# data = []

# for feature_name, importance_value in zip(top_17_features, feature_importances):
#     # 判断 category 的类别
#     if feature_name.lower() in ["lon", "lat", "elev"]:
#         category = "geo"
#     elif feature_name in ['MAP', 'WIND', 'MAXMAT', 'AVGMAT', 'SARD', 'VAPR']:
#         category = "clim"
#     else:
#         category = "soil"

#     data.append({
#         "Feature": feature_name,
#         "Importance": importance_value,
#         "Category": category
#     })

# # 创建 DataFrame
# importance_df = pd.DataFrame(data)

# importance_df_sorted = importance_df.sort_values(by='Importance', ascending=False)

# # 转换为 LaTeX 三线表
# latex_table = importance_df_sorted.to_latex(index=False, header=True, 
#                                             caption="Top 17 Feature Importances", 
#                                             label="tab:top_17_feature_importances",
#                                             column_format="|l|r|")  # l 表示左对齐，r 表示右对齐

# # 输出 LaTeX 代码
# with open('data/model/top_17_feature_importances.tex', 'w') as f:
#     f.write(latex_table)

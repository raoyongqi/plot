import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import  GradientBoostingRegressor

from sklearn.metrics import mean_squared_error, r2_score
import joblib  # 用于保存模型
import os  # 用于处理文件和目录
import re
import matplotlib.pyplot as plt
import seaborn as sns

file_path = 'data/hand_data.xlsx'

data = pd.read_excel(file_path)

feature_columns = [col for col in data.columns if col !='RATIO']

X = data[feature_columns]
y = data['RATIO']  # 目标变量

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf =  GradientBoostingRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"均方误差 (MSE): {mse:.4f}")
print(f"R² 得分: {r2:.4f}")

os.makedirs('data/model', exist_ok=True)

joblib.dump(rf, 'data/model/random_forest_model.pkl')

feature_importances = rf.feature_importances_
data = []

for feature_name, importance_value in zip(feature_columns, feature_importances):
    
    feature_name = re.sub('_resampled', '', feature_name)
    
    if feature_name.lower() in ["lon", "lat","hand"]:
        category = "geo"
    elif feature_name.startswith('WC'):
        category = "clim"
    else:
        category = "soil"

    data.append({
        "Feature": feature_name,
        "Importance": importance_value,
        "Category": category
    })

importance_df = pd.DataFrame(data)

importance_df.sort_values(by='Importance', ascending=False).to_csv('data/model/feature_importances.csv', index=False)

predictions_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})

top_n = 10
top_importance_df = importance_df.sort_values(by='Importance', ascending=False).head(top_n)


# # 设置颜色映射
# # 设置颜色映射，使用更柔和的颜色
# category_colors = {
#     "geo": "#4C8BF9",  # 柔和的蓝色
#     "clim": "#6BCB4A",  # 柔和的绿色
#     "soil": "#FFA500"   # 柔和的橙色
# }

# # 将颜色应用于 DataFrame
# top_importance_df['color'] = top_importance_df['Category'].map(category_colors)

# # 绘制前10个最重要的变量重要性图
# plt.figure(figsize=(10, 6))

# # 直接使用颜色列表作为 palette
# bar_plot = sns.barplot(x='Importance', y='Feature', data=top_importance_df, 
#             palette=top_importance_df['color'].tolist())

# # 添加标题和标签
# plt.title(f'Top {top_n} Important Features', fontsize=20)  # 增加字体大小
# plt.xlabel('Importance', fontsize=20)  # 增加字体大小
# plt.ylabel('Features', fontsize=20)  # 增加字体大小

# # 设置坐标轴刻度字体大小
# plt.tick_params(axis='both', which='major', labelsize=14)

# # 添加图例
# handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in category_colors.values()]
# plt.legend(handles, category_colors.keys(), title="Category", fontsize=12)

# # 显示图形
# plt.tight_layout()

# plt.savefig('data/model/rf_summary_plot.png', bbox_inches='tight')  # 使用 bbox_inches='tight' 确保内容完整

# plt.show()
# plt.close()

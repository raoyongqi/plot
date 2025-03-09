import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import  GradientBoostingRegressor

from sklearn.metrics import mean_squared_error, r2_score
import joblib  # 用于保存模型
import os  # 用于处理文件和目录
import re
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 读取Excel文件
file_path = 'data/hand_data.xlsx'  # 替换为你的文件路径
data = pd.read_excel(file_path)

# 2. 筛选特征列
feature_columns = [col for col in data.columns if col !='RATIO']

# 3. 分离特征变量和目标变量
X = data[feature_columns]
y = data['RATIO']  # 目标变量

# 4. 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. 初始化并训练随机森林回归模型
rf =  GradientBoostingRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 6. 预测并评估模型
y_pred = rf.predict(X_test)

# 7. 评估模型性能
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 输出结果
print(f"均方误差 (MSE): {mse:.4f}")
print(f"R² 得分: {r2:.4f}")

# 8. 确保保存路径存在
os.makedirs('data/model', exist_ok=True)

# 9. 保存模型
joblib.dump(rf, 'data/model/random_forest_model.pkl')

# 10. 保存变量重要性
feature_importances = rf.feature_importances_
data = []

# 遍历特征列和特征重要性
for feature_name, importance_value in zip(feature_columns, feature_importances):
    feature_name = re.sub('_resampled', '', feature_name)  # 移除 '_resampled'
    
    # 判断 category 的类别
    if feature_name.lower() in ["lon", "lat","hand"]:
        category = "geo"
    elif feature_name.startswith('WC'):
        category = "clim"
    else:
        category = "soil"

    # 将每一行的字典添加到列表中
    data.append({
        "Feature": feature_name,
        "Importance": importance_value,
        "Category": category
    })

# 创建 DataFrame
importance_df = pd.DataFrame(data)

# 按 Importance 降序排列并保存为 CSV 文件
importance_df.sort_values(by='Importance', ascending=False).to_csv('data/model/feature_importances.csv', index=False)

# 11. 保存预测结果
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

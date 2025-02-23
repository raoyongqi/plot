import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# 文件列表
files = [
    'denseNet_actual_vs_predicted.csv',
    'residual_dilated10_actual_vs_predicted.csv',
    'residual_dilated20_actual_vs_predicted.csv',
    'wavenet1_actual_vs_predicted.csv',
    'wavenet2_actual_vs_predicted.csv',
    'wavenet3_actual_vs_predicted.csv'
]

# 设置颜色和标签
colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown']
labels = ['DenseNet', 'Residual Dilated 10', 'Residual Dilated 20', 'Wavenet1', 'Wavenet2', 'Wavenet3']

# 绘制实际值与不同模型的预测值
plt.figure(figsize=(10, 6))

# 绘制实际值
# 读取 wavenet3 的数据，用于动态计算拟合线的范围
df_wavenet3 = pd.read_csv('wavenet3_actual_vs_predicted.csv')
min_actual = df_wavenet3["Actual"].min()
max_actual = df_wavenet3["Actual"].max()

# 根据 wavenet3 数据的实际值范围，动态生成拟合线的 x 轴
x_vals = np.linspace(min_actual, max_actual, 100)  # 创建与实际数据范围相同的 x 值
plt.plot(x_vals, x_vals, color='black', linestyle='-', label='Actual Line')  # 绘制实际值拟合线

# 创建一个字典来保存每个模型的R²值
r_squared_values = {}

# 读取并绘制每个文件的预测值与实际值
for i, file in enumerate(files):
    df = pd.read_csv(file)
    plt.scatter(df["Actual"], df["Predicted"], color=colors[i], label=labels[i], alpha=0.6)

    # 拟合线
    reg = LinearRegression().fit(df[["Actual"]], df["Predicted"])


    # 计算 R² 值
    r_squared = reg.score(df[["Actual"]], df["Predicted"])
    r_squared_values[labels[i]] = r_squared

# 设置图标
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted for Different Models")

# 添加 R² 值到图例旁边
legend_labels = [f"{label} (R² = {r_squared:.2f})" for label, r_squared in r_squared_values.items()]
plt.legend(labels=['Actual Line'] + legend_labels, loc='upper left')

plt.show()

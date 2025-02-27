import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# 文件列表
files = [
    'history/denseNet_training_history.csv',
    'history/wavenet1_training_history.csv',
    'history/wavenet2_training_history.csv',
    'history/wavenet5_training_history.csv'
]

# 设置颜色和线型
line_styles = {
    'denseNet': ('-', 'blue'),
    'wavenet1': ('--', 'orange'),
    'wavenet2': ('-.', 'green'),
    'wavenet5': (':', 'red')
}
import matplotlib.gridspec as gridspec
# 创建一个图像和两个子图
fig = plt.figure(figsize=(14, 18))
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 2])  # 上面高度比 1:2，2 行 2 列
# 创建子图
ax1 = fig.add_subplot(gs[0, 0])  # 左侧第一个子图（训练损失）
ax2 = fig.add_subplot(gs[0, 1])  # 右侧第二个子图（验证损失）
ax3 = fig.add_subplot(gs[1, :])  # 下面第三个子图（实际 vs 预测，占满第二行）
# 绘制训练损失和验证损失

# 为每个 CSV 文件绘制训练损失（loss）和验证损失（val_loss）曲线
for file in files:
    # 读取当前 CSV 文件
    df = pd.read_csv(file)
    
    # 提取 'loss' 和 'val_loss' 列
    loss = df['loss'].values
    val_loss = df['val_loss'].values
    
    # 从文件名中提取模型名称
    model_name = file.split("/")[-1].split("_")[0]
    
    # 获取该模型的线型和颜色
    linestyle, color = line_styles[model_name]
    
    # 在左侧子图（ax1）上绘制训练损失曲线
    ax1.plot(loss, label=f'{model_name} Train Loss', linestyle=linestyle, color=color)
    
    # 在右侧子图（ax2）上绘制验证损失曲线
    ax2.plot(val_loss, label=f'{model_name} Val Loss', linestyle=linestyle, color=color)

# 设置左侧子图的标题和标签
ax1.set_title('Training Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss Value')

# 设置右侧子图的标题和标签
ax2.set_title('Validation Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss Value')

# 设置 Y 轴范围
ax1.set_ylim([10, 50])  # 左侧 Y 轴范围
ax2.set_ylim([10, 50])  # 右侧 Y 轴范围

# 添加图例
ax1.legend(loc='upper left')
ax2.legend(loc='upper left')


# 绘制实际值与不同模型的预测值
files_actual = [
    'actual/denseNet_actual_vs_predicted.csv',
    'actual/wavenet1_actual_vs_predicted.csv',
    'actual/wavenet2_actual_vs_predicted.csv',
    'actual/wavenet5_actual_vs_predicted.csv'
]

# 设置颜色和标签
colors = ['blue', 'green', 'red', 'orange']
labels = ['DenseNet', 'Wavenet1', 'Wavenet2', 'Wavenet5']

# 读取 wavenet3 的数据，用于动态计算拟合线的范围
df_wavenet3 = pd.read_csv('actual/wavenet5_actual_vs_predicted.csv')
min_actual = df_wavenet3["Actual"].min()
max_actual = df_wavenet3["Actual"].max()


# 为每个 CSV 文件绘制训练损失（loss）和验证损失（val_loss）曲线
for file in files:
    # 读取当前 CSV 文件
    df = pd.read_csv(file)
    
    # 提取 'loss' 和 'val_loss' 列
    loss = df['loss'].values
    val_loss = df['val_loss'].values
    
    # 从文件名中提取模型名称
    model_name = file.split("/")[-1].split("_")[0]
    
    # 获取该模型的线型和颜色
    linestyle, color = line_styles[model_name]
    
    # 在左侧子图（ax1）上绘制训练损失曲线
    ax1.plot(loss, label=f'{model_name} Train Loss', linestyle=linestyle, color=color)
    
    # 在右侧子图（ax2）上绘制验证损失曲线
    ax2.plot(val_loss, label=f'{model_name} Val Loss', linestyle=linestyle, color=color)

# 设置左侧子图的标题和标签
ax1.set_title('Training Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss Value')

# 设置右侧子图的标题和标签
ax2.set_title('Validation Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss Value')

# 设置 Y 轴范围
ax1.set_ylim([10, 50])  # 左侧 Y 轴范围
ax2.set_ylim([10, 50])  # 右侧 Y 轴范围


#df_wavenet3 = pd.read_csv('actual/wavenet5_actual_vs_predicted.csv')
min_actual = df_wavenet3["Actual"].min()
max_actual = df_wavenet3["Actual"].max()
files_actual = [
    'actual/denseNet_actual_vs_predicted.csv',
    'actual/wavenet1_actual_vs_predicted.csv',
    'actual/wavenet2_actual_vs_predicted.csv',
    'actual/wavenet5_actual_vs_predicted.csv'
]
# 根据 wavenet3 数据的实际值范围，动态生成拟合线的 x 轴
x_vals = np.linspace(min_actual, max_actual, 100)  # 创建与实际数据范围相同的 x 值
ax3.plot(x_vals, x_vals, color='black', linestyle='-', label='Actual Line')  # 绘制实际值拟合线

# 创建一个字典来保存每个模型的R²值
r_squared_values = {}

# 读取并绘制每个文件的预测值与实际值
for i, file in enumerate(files_actual):
    df = pd.read_csv(file)
    ax3.scatter(df["Actual"], df["Predicted"], color=colors[i], label=labels[i], alpha=0.6)

    # 拟合线
    reg = LinearRegression().fit(df[["Actual"]], df["Predicted"])
    y_pred = reg.predict(df[["Actual"]])
    ax3.plot(df["Actual"], y_pred, color=colors[i], linestyle='--', label="_nolegend_")  # 不显示拟合线的图例

    # 计算 R² 值
    r_squared = reg.score(df[["Actual"]], df["Predicted"])
    r_squared_values[labels[i]] = r_squared

# 设置图标
# 设置图标
ax3.set_xlabel("Actual")  # 正确设置 x 轴标签
ax3.set_ylabel("Predicted")  # 正确设置 y 轴标签
ax3.set_title("Actual vs Predicted for Different Models")  # 正确设置标题


# 添加 R² 值到图例旁边
legend_labels = [f"{label} (R² = {r_squared:.2f})" for label, r_squared in r_squared_values.items()]
ax3.legend(labels=['Actual Line'] + legend_labels, loc='upper left')
plt.tight_layout()

plt.show()

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


fig = plt.figure(figsize=(14, 18))
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 2])  # 上面高度比 1:2，2 行 2 列

ax1 = fig.add_subplot(gs[0, 0])  # 左侧第一个子图（训练损失）
ax2 = fig.add_subplot(gs[0, 1])  # 右侧第二个子图（验证损失）
ax3 = fig.add_subplot(gs[1, :])  # 下面第三个子图（实际 vs 预测，占满第二行）

for file in files:

    df = pd.read_csv(file)
    
    loss = df['loss'].values
    val_loss = df['val_loss'].values
    
    model_name = file.split("/")[-1].split("_")[0]
    
    linestyle, color = line_styles[model_name]
    
    ax1.plot(loss, label=f'{model_name} Train Loss', linestyle=linestyle, color=color)
    
    ax2.plot(val_loss, label=f'{model_name} Val Loss', linestyle=linestyle, color=color)

ax1.set_title('Training Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss Value')

ax2.set_title('Validation Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss Value')

ax1.set_ylim([10, 50])  # 左侧 Y 轴范围
ax2.set_ylim([10, 50])  # 右侧 Y 轴范围

ax1.legend(loc='upper left')
ax2.legend(loc='upper left')

files_actual = [
    'history/denseNet_actual_vs_predicted.csv',
    'history/wavenet1_actual_vs_predicted.csv',
    'history/wavenet2_actual_vs_predicted.csv',
    'history/wavenet5_actual_vs_predicted.csv'
]

colors = ['blue', 'green', 'red', 'orange']
labels = ['DenseNet', 'Wavenet1', 'Wavenet2', 'Wavenet5']

df_wavenet3 = pd.read_csv('history/wavenet5_actual_vs_predicted.csv')
min_actual = df_wavenet3["Actual"].min()
max_actual = df_wavenet3["Actual"].max()

for file in files:

    df = pd.read_csv(file)
    
    loss = df['loss'].values
    val_loss = df['val_loss'].values
    
    model_name = file.split("/")[-1].split("_")[0]
    
    linestyle, color = line_styles[model_name]
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import savgol_filter

    loss = np.array(loss)  # 转换为 NumPy 数组
    val_loss = np.array(val_loss)

    window_size = min(11, len(loss))  # 确保窗口大小不超过数据长度
    if window_size % 2 == 0:  # 窗口大小必须是奇数
        window_size += 1

    smooth_loss = savgol_filter(loss, window_size, 3)  # 3阶多项式拟合
    smooth_val_loss = savgol_filter(val_loss, window_size, 3)

    loss_std = np.std(loss) * 0.1
    val_loss_std = np.std(val_loss) * 0.1

    ax1.plot(smooth_loss, label=f'{model_name} Train Loss', linestyle=linestyle, color=color)
    ax1.fill_between(range(len(smooth_loss)), smooth_loss - loss_std, smooth_loss + loss_std, 
                    color=color, alpha=0.2)  # 绘制阴影

    ax2.plot(smooth_val_loss, label=f'{model_name} Val Loss', linestyle=linestyle, color=color)
    ax2.fill_between(range(len(smooth_val_loss)), smooth_val_loss - val_loss_std, smooth_val_loss + val_loss_std, 
                    color=color, alpha=0.2)  # 绘制阴影

ax1.set_title('Training Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss Value')


ax2.set_title('Validation Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss Value')

ax1.set_ylim([10, 50])  # 左侧 Y 轴范围
ax2.set_ylim([10, 50])  # 右侧 Y 轴范围


min_actual = df_wavenet3["Actual"].min()
max_actual = df_wavenet3["Actual"].max()
files_actual = [
    'history/denseNet_actual_vs_predicted.csv',
    'history/wavenet1_actual_vs_predicted.csv',
    'history/wavenet2_actual_vs_predicted.csv',
    'history/wavenet5_actual_vs_predicted.csv'
]

x_vals = np.linspace(min_actual, max_actual, 100)  # 创建与实际数据范围相同的 x 值
ax3.plot(x_vals, x_vals, color='black', linestyle='-', label='Actual Line')  # 绘制实际值拟合线

r_squared_values = {}

for i, file in enumerate(files_actual):
    df = pd.read_csv(file)
    ax3.scatter(df["Actual"], df["Predicted"], color=colors[i], label=labels[i], alpha=0.6)

    reg = LinearRegression().fit(df[["Actual"]], df["Predicted"])
    y_pred = reg.predict(df[["Actual"]])
    ax3.plot(df["Actual"], y_pred, color=colors[i], linestyle='--', label="_nolegend_")  # 不显示拟合线的图例

    r_squared = reg.score(df[["Actual"]], df["Predicted"])
    r_squared_values[labels[i]] = r_squared

ax3.set_xlabel("Actual")
ax3.set_ylabel("Predicted")
ax3.set_title("Actual vs Predicted for Different Models")

legend_labels = [f"{label} (R² = {r_squared:.2f})" for label, r_squared in r_squared_values.items()]
ax3.legend(labels=['Actual Line'] + legend_labels, loc='upper left')
plt.tight_layout()

plt.show()

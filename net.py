import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# 设置随机种子以确保可重复性
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
tf.random.set_seed(42)

# 读取 Excel 文件
file_path = 'data/climate_soil_tif.xlsx'  # 替换为你的文件路径
data = pd.read_excel(file_path)

# 筛选特征列
feature_columns = [col for col in data.columns if col.endswith('_resampled') or col.lower().startswith('wc') or col in ['LON', 'LAT']]

# 分离特征和目标变量
X = data[feature_columns]
y = data['RATIO']

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------
# PyTorch模型部分
# ------------------------
# 转换为PyTorch张量
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# 构建神经网络模型
class NNModel(nn.Module):
    def __init__(self):
        super(NNModel, self).__init__()
        self.layer1 = nn.Linear(X_train_scaled.shape[1], 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.leaky_relu(self.layer1(x))
        x = self.dropout(x)
        x = self.leaky_relu(self.layer2(x))
        x = self.dropout(x)
        x = self.leaky_relu(self.layer3(x))
        x = self.output(x)
        return x

# 初始化模型
model = NNModel()

# 设置优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# 训练模型
epochs = 150
batch_size = 16
train_losses_pytorch = []
val_losses_pytorch = []
# 8. Early stopping setup
patience = 40  # Number of epochs to wait for improvement
best_val_loss = float('inf')
patience_counter = 0
train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    permutation = torch.randperm(X_train_tensor.size(0))
    for i in range(0, X_train_tensor.size(0), batch_size):
        indices = permutation[i:i+batch_size]
        batch_x, batch_y = X_train_tensor[indices], y_train_tensor[indices]

        optimizer.zero_grad()
        y_pred = model(batch_x)
        loss = loss_fn(y_pred, batch_y)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        y_train_pred = model(X_train_tensor)
        train_loss = loss_fn(y_train_pred, y_train_tensor).item()
        y_val_pred = model(X_test_tensor)
        val_loss = loss_fn(y_val_pred, y_test_tensor).item()

    train_losses_pytorch.append(train_loss)
    val_losses_pytorch.append(val_loss)

    # Check if validation loss improved
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0  # Reset patience counter
    else:
        patience_counter += 1

    # Stop if no improvement for 'patience' epochs
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1} due to no improvement in validation loss")
        break

    if epoch % 10 == 0:
        print(f'Epoch {epoch}/{epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

# 计算PyTorch预测
y_test_pred_pytorch = model(X_test_tensor).detach().numpy()

# ------------------------
# TensorFlow模型部分
# ------------------------

# 构建神经网络模型
nn_model = Sequential([
    Dense(128, input_dim=X_train_scaled.shape[1]),
    LeakyReLU(alpha=0.1),
    Dropout(0.1),
    Dense(64),
    LeakyReLU(alpha=0.1),
    Dropout(0.1),
    Dense(32),
    LeakyReLU(alpha=0.1),
    Dense(1, activation='linear')
])

# 设置优化器和回调函数
optimizer = Adam(learning_rate=0.001)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-5)
early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

# 编译模型
nn_model.compile(loss='mean_squared_error', optimizer=optimizer)

# 训练模型并保存训练历史
history = nn_model.fit(
    X_train_scaled, y_train,
    validation_data=(X_test_scaled, y_test),
    epochs=60, batch_size=16,
    verbose=1, callbacks=[reduce_lr, early_stopping]
)

# 计算TensorFlow预测
y_test_pred_tensorflow = nn_model.predict(X_test_scaled)

# ------------------------
# 绘制损失曲线（左右并排图）
# ------------------------

# ------------------------
# 绘制训练和验证损失（左右并排图）
# ------------------------

# ------------------------
# 绘制训练和验证损失（左右并排图）
# ------------------------


fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# 设置字体大小
fontsize = 16

# 绘制训练损失（左图）
axs[0].plot(train_losses_pytorch, label='Training Loss (PyTorch)', color='b')
axs[0].plot(history.history['loss'], label='Training Loss (TensorFlow)', color='r')
axs[0].set_title('Training Loss Comparison (PyTorch vs TensorFlow)', fontsize=fontsize)
axs[0].set_xlabel('Epochs', fontsize=fontsize)
axs[0].set_ylabel('Loss', fontsize=fontsize)
axs[0].legend(fontsize=fontsize)

# 绘制验证损失（右图）
axs[1].plot(val_losses_pytorch, label='Validation Loss (PyTorch)', color='b', linestyle='dashed')
axs[1].plot(history.history['val_loss'], label='Validation Loss (TensorFlow)', color='r', linestyle='dashed')
axs[1].set_title('Validation Loss Comparison (PyTorch vs TensorFlow)', fontsize=fontsize)
axs[1].set_xlabel('Epochs', fontsize=fontsize)
axs[1].set_ylabel('Loss', fontsize=fontsize)
axs[1].legend(fontsize=fontsize)

# 保存图像
plt.tight_layout()
plt.savefig("loss_comparison.png", dpi=300, bbox_inches='tight')
plt.show()


# ------------------------
# 计算PyTorch预测的MSE和R²
# ------------------------
mse_pytorch = mean_squared_error(y_test, y_test_pred_pytorch.flatten())
r2_pytorch = r2_score(y_test, y_test_pred_pytorch.flatten())

# ------------------------
# 计算TensorFlow预测的MSE和R²
# ------------------------
mse_tensorflow = mean_squared_error(y_test, y_test_pred_tensorflow.flatten())
r2_tensorflow = r2_score(y_test, y_test_pred_tensorflow.flatten())

# ------------------------
# 创建MSE和R²对比表格
# ------------------------
comparison_metrics_df = pd.DataFrame({
    'Model': ['PyTorch', 'TensorFlow'],
    'MSE': [mse_pytorch, mse_tensorflow],
    'R²': [r2_pytorch, r2_tensorflow]
})

# 打印对比表格
print(comparison_metrics_df)

# 可选择将表格保存为CSV文件
comparison_metrics_df.to_csv('model_comparison_metrics.csv', index=False)

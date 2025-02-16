import numpy as np
import matplotlib.pyplot as plt
from quantile_forest import RandomForestQuantileRegressor

# 模拟数据 (替换为你的实际数据)
np.random.seed(0)
n = 100
X = np.linspace(0, 10, n).reshape(-1, 1)  # 注意需要二维数组作为输入
y = 3 * X.flatten() + np.random.normal(scale=2, size=n)

# 初始化并训练随机森林分位数回归模型
quantile_regressor = RandomForestQuantileRegressor(random_state=0)
quantile_regressor.fit(X, y)

# 设置分位数，获取条件中位数（q=0.5）和95%预测区间
quantiles = [0.05, 0.5, 0.95]

# 获取预测值和预测区间
predictions = quantile_regressor.predict(X, quantiles=quantiles)
lower_bound = predictions[0]
pred_median = predictions[1]
upper_bound = predictions[2]

# 确保X和pred_median是相同维度的
X_flattened = X.flatten()  # 将X转化为一维数组

# 绘制结果
plt.figure(figsize=(8, 6))

# 绘制真实观察值
plt.scatter(X_flattened, y, color='yellow', label='True Observations', s=100)

# 绘制预测的条件中位数
plt.plot(X_flattened, pred_median, color='red', label='Predicted Conditional Median (q=0.5)', lw=2)

# 绘制95%预测区间
plt.fill_between(X_flattened, lower_bound, upper_bound, color='blue', alpha=0.3, label='95% Prediction Interval')

# 添加标签和标题
plt.xlabel('True Observations')
plt.ylabel('Predicted Conditional Median')
plt.title('True Observations vs Predicted Conditional Median')

# 显示图例
plt.legend()

# 显示网格
plt.grid(True)
plt.show()

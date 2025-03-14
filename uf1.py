import numpy as np
import matplotlib.pyplot as plt
from quantile_forest import RandomForestQuantileRegressor

np.random.seed(0)
n = 100
X = np.linspace(0, 10, n).reshape(-1, 1)

y = 3 * X.flatten() + np.random.normal(scale=2, size=n)

quantile_regressor = RandomForestQuantileRegressor(random_state=0)

quantile_regressor.fit(X, y)

quantiles = [0.05, 0.5, 0.95]

predictions = quantile_regressor.predict(X, quantiles=quantiles)
lower_bound = predictions[0]
pred_median = predictions[1]
upper_bound = predictions[2]

X_flattened = X.flatten()  


plt.figure(figsize=(8, 6))

plt.scatter(X_flattened, y, color='yellow', label='True Observations', s=100)

plt.plot(X_flattened, pred_median, color='red', label='Predicted Conditional Median (q=0.5)', lw=2)

plt.fill_between(X_flattened, lower_bound, upper_bound, color='blue', alpha=0.3, label='95% Prediction Interval')

plt.xlabel('True Observations')
plt.ylabel('Predicted Conditional Median')
plt.title('True Observations vs Predicted Conditional Median')

plt.legend()

plt.grid(True)
plt.show()

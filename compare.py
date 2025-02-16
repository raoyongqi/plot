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
import numpy as np
import time
import xgboost as xgb
import lightgbm as lgb  # Import LightGBM
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib import rcParams
# 1. 读取Excel文件
# 1. 读取Excel文件
file_path = "data/selection.csv"  # 替换为你的文件路径
selection = pd.read_csv(file_path)

# 对自变量进行标准化，保持因变量不变
X = selection.drop(columns='RATIO')
y = selection['RATIO']


# 4. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 5. Standardize the data
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# 6. Build the neural network model
# nn_model = Sequential()
# nn_model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))  # Input layer and hidden layer
# nn_model.add(Dense(32, activation='relu'))  # Second hidden layer
# nn_model.add(Dense(1))  # Output layer

# # 7. Compile the model
# nn_model.compile(loss='mean_squared_error', optimizer='adam')

# # 8. Train the model
# start_time = time.time()
# nn_model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=0)  # verbose=0 to turn off training progress bar
# nn_time = time.time() - start_time

# 9. Predict and evaluate the neural network model
# 10. Train and evaluate the XGBoost model
start_time = time.time()
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse')
xgb_model.fit(X_train, y_train)
xgb_time = time.time() - start_time

y_pred_xgb = xgb_model.predict(X_test)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

print(f"XGBoost Mean Squared Error (MSE): {mse_xgb:.4f}")
print(f"XGBoost R² Score: {r2_xgb:.4f}")
print(f"XGBoost Training Time: {xgb_time:.2f} seconds")

# 11. Train and evaluate the Random Forest model
start_time = time.time()
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_time = time.time() - start_time

y_pred_rf = rf_model.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Random Forest Mean Squared Error (MSE): {mse_rf:.4f}")
print(f"Random Forest R² Score: {r2_rf:.4f}")
print(f"Random Forest Training Time: {rf_time:.2f} seconds")

# 12. Train and evaluate the LightGBM model
start_time = time.time()
lgb_model = lgb.LGBMRegressor()
lgb_model.fit(X_train, y_train)
lgb_time = time.time() - start_time

y_pred_lgb = lgb_model.predict(X_test)
mse_lgb = mean_squared_error(y_test, y_pred_lgb)
r2_lgb = r2_score(y_test, y_pred_lgb)

print(f"LightGBM Mean Squared Error (MSE): {mse_lgb:.4f}")
print(f"LightGBM R² Score: {r2_lgb:.4f}")
print(f"LightGBM Training Time: {lgb_time:.2f} seconds")

# 13. Prepare data for plotting
models = [ 'XGBoost', 'Random Forest', 'LightGBM']
r2_scores = [ r2_xgb, r2_rf, r2_lgb]
times = [ xgb_time, rf_time, lgb_time]

# Create DataFrame for sorting
results_df = pd.DataFrame({
    'Model': models,
    'R2 Score': r2_scores,
    'Training Time': times
})

# Sort by R² score from high to low
results_df_sorted_r2 = results_df.sort_values(by='R2 Score', ascending=False)

# Sort by Training Time from low to high
results_df_sorted_time = results_df.sort_values(by='Training Time', ascending=True)

# Plot performance-time graph
plt.figure(figsize=(14, 7))

# Plot R² Scores sorted from high to low
plt.subplot(1, 2, 1)
plt.bar(results_df_sorted_r2['Model'], results_df_sorted_r2['R2 Score'], color=['blue', 'green', 'red', 'purple'])
plt.xlabel('Model')
plt.ylabel('R² Score')
plt.title('R² Score Comparison (High to Low)')

# Plot Training Time sorted from low to high
plt.subplot(1, 2, 2)
plt.bar(results_df_sorted_time['Model'], results_df_sorted_time['Training Time'], color=['blue', 'green', 'red', 'purple'])
plt.xlabel('Model')
plt.ylabel('Training Time (seconds)')
plt.title('Training Time Comparison (Low to High)')

plt.tight_layout()
plt.savefig('data/model_performance_comparison.png')  # Save plot to file
plt.show()


print("Model Performance Comparison:")

# Print R² scores sorted from high to low
print("\nR² Scores (High to Low):")
print(results_df_sorted_r2[['Model', 'R2 Score']])

# Print training times sorted from low to high
results_df_combined = results_df.sort_values(by='R2 Score', ascending=False)

# Save the combined table to LaTeX
combined_tex_path = "data/model_performance.tex"
results_df_combined.to_latex(
    combined_tex_path,
    index=False,
    caption="Model Performance Comparison (R² Score and Training Time)",
    label="tab:model_performance",
    float_format="%.4f"
)

# Output path for reference
print(f"Combined Model Performance table saved to LaTeX: {combined_tex_path}")
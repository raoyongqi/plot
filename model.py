import pandas as pd
import numpy as np
import os  # Import os to handle directory operations
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Set font for displaying Chinese characters
rcParams['font.sans-serif'] = ['/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc']
rcParams['axes.unicode_minus'] = False  # Fix issue with minus sign '-' showing as a block

# Load the Excel file
file_path = 'data/climate_soil.xlsx'
data = pd.read_excel(file_path)

# Select feature columns
feature_columns = [col for col in data.columns if col !='RATIO']
X = data[feature_columns]
y = data['RATIO']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the neural network model
nn_model = Sequential()
nn_model.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))
nn_model.add(Dense(32, activation='relu'))
nn_model.add(Dense(1))

# Compile the model
nn_model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
nn_model.fit(X_train_scaled, y_train, epochs=50, batch_size=10, verbose=0)

# Predict and evaluate the neural network model
y_pred_nn = nn_model.predict(X_test_scaled)

# Train and evaluate the XGBoost model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse')
xgb_model.fit(X_train_scaled, y_train)
y_pred_xgb = xgb_model.predict(X_test_scaled)

# Train and evaluate the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)

# Train and evaluate the LightGBM model
lgb_model = lgb.LGBMRegressor()
lgb_model.fit(X_train_scaled, y_train)
y_pred_lgb = lgb_model.predict(X_test_scaled)

# Prepare data for plotting
models = ['Neural Network', 'XGBoost', 'Random Forest', 'LightGBM']
predictions = [y_pred_nn, y_pred_xgb, y_pred_rf, y_pred_lgb]

# Create the directory if it doesn't exist
output_dir = 'data'
os.makedirs(output_dir, exist_ok=True)

# Create the plots
for model_name, pred_k in zip(models, predictions):
    plt.figure(figsize=(10, 10))  # Adjust the figure size as needed
    
    # Ensure y_test and pred_k are 1-dimensional arrays
    y_test_flat = y_test.values.flatten()  # Convert y_test to 1D array
    pred_k_flat = pred_k.flatten()  # Convert predictions to 1D array

    # 绘制散点图和密度图
    sns.kdeplot(x=y_test_flat, y=pred_k_flat, fill=True, cmap="Blues", thresh=0.05)  # Use soft blue color
    plt.scatter(y_test_flat, pred_k_flat, alpha=0.5, color="purple")

    # 拟合曲线
    linear_model = LinearRegression()
    linear_model.fit(y_test_flat.reshape(-1, 1), pred_k_flat)
    pred_line = linear_model.predict(y_test_flat.reshape(-1, 1))
    plt.plot(y_test_flat, pred_line, color='orange', label="Fitted Line")  # Use soft orange color

    # 理想的 y=x 参考线
    plt.plot([min(y_test_flat), max(y_test_flat)], [min(y_test_flat), max(y_test_flat)], 'k--', label="Ideal Line")

    # 设置标签和标题
    plt.xlabel('True Values', fontsize=14, fontweight='bold')  # 横轴标题加粗
    plt.ylabel('Predicted Values', fontsize=14, fontweight='bold')  # 纵轴标题加粗
    plt.title(model_name, fontsize=16, fontweight='bold')  # 标题加粗

    plt.xlim(min(y_test_flat), max(y_test_flat))
    plt.ylim(min(y_test_flat), max(y_test_flat))

    # 显示图例
    plt.legend(fontsize=16, loc='best', title_fontsize='16', frameon=True)

    # 保存每个图像
    plt.savefig(f'data/{model_name.replace(" ", "_")}_plot.png', dpi=300)

    # 调整布局以避免重叠
    plt.tight_layout()
    plt.show()  # Optional: Show each plot


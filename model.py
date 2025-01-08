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
import time  # For measuring execution time

# Set font for displaying Chinese characters
rcParams['font.sans-serif'] = ['/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc']
rcParams['axes.unicode_minus'] = False  # Fix issue with minus sign '-' showing as a block

# Load the Excel file
file_path = 'data/climate_soil_tif.xlsx'
data = pd.read_excel(file_path)

# Select feature columns
feature_columns = [col for col in data.columns if col != 'RATIO']
X = data[feature_columns]
y = data['RATIO']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Function to evaluate model performance
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    start_time = time.time()  # Measure the time before training
    if model_name == "Neural Network":
        model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=1)  # Train the NN
    else:
        model.fit(X_train, y_train)  # Train other models without epochs and batch_size

    end_time = time.time()  # Measure the time after training

    # Predict and evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    execution_time = end_time - start_time  # Time taken for training

    # Return the results
    return model_name, execution_time, mse, r2, y_pred

# Initialize list to store performance results and predictions
performance_results = []
predictions = []

# Create models
models = [
    ("Neural Network", Sequential([
        Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)
    ])),
    ("XGBoost", xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse', random_state=42)),
    ("Random Forest", RandomForestRegressor(n_estimators=100, random_state=42)),
    ("LightGBM", lgb.LGBMRegressor(random_state=42))
]

# Train and evaluate each model
for model_name, model in models:
    if model_name == "Neural Network":
        model.compile(loss='mean_squared_error', optimizer='adam')  # Compile for NN
    model_results = evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test, model_name)
    performance_results.append(model_results[:-1])  # Exclude predictions from this list for CSV
    predictions.append(model_results[-1])  # Save predictions for plotting

# Convert the performance results to a DataFrame
performance_df = pd.DataFrame(performance_results, columns=["Model", "Time (s)", "MSE", "R2"])

# Save the results to a CSV file
output_dir = 'data'
os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists
performance_df.to_csv(f'{output_dir}/model_performance.csv', index=False)

# Display the performance table
print(performance_df)

# # Create plots for each model
# for model_name, pred_k in zip([model[0] for model in models], predictions):
#     plt.figure(figsize=(10, 10))  # Set the figure size
    
#     # Ensure y_test and pred_k are 1-dimensional arrays
#     y_test_flat = y_test.values.flatten()  # Convert y_test to 1D array
#     pred_k_flat = pred_k.flatten()  # Convert predictions to 1D array

#     # Plot scatter and density plots
#     sns.kdeplot(x=y_test_flat, y=pred_k_flat, fill=True, cmap="Blues", thresh=0.05, alpha=0.7)  # Density plot
#     plt.scatter(y_test_flat, pred_k_flat, alpha=0.5, color="purple")  # Scatter plot

#     # Fit and plot the regression line
#     linear_model = LinearRegression()
#     linear_model.fit(y_test_flat.reshape(-1, 1), pred_k_flat.reshape(-1, 1))
#     pred_line = linear_model.predict(y_test_flat.reshape(-1, 1))
#     plt.plot(y_test_flat, pred_line, color='orange', label="Fitted Line")  # Regression line

#     # Plot the ideal reference line (y=x)
#     plt.plot([min(y_test_flat), max(y_test_flat)], [min(y_test_flat), max(y_test_flat)], 'k--', label="Ideal Line")

#     # Set axis labels and title
#     plt.xlabel('True Values', fontsize=14, fontweight='bold')
#     plt.ylabel('Predicted Values', fontsize=14, fontweight='bold')
#     plt.title(model_name, fontsize=16, fontweight='bold')

#     # Set axis limits
#     plt.xlim(min(y_test_flat), max(y_test_flat))
#     plt.ylim(min(y_test_flat), max(y_test_flat))

#     # Show legend
#     plt.legend(fontsize=12, loc='best', title_fontsize='12', frameon=True)

#     # Save each plot
#     plot_path = f'{output_dir}/{model_name.replace(" ", "_")}_plot.png'
#     plt.savefig(plot_path, dpi=300)

#     # Adjust layout and display the plot
#     plt.tight_layout()
#     plt.show()

#     print(f"Plot saved for {model_name} at {plot_path}")

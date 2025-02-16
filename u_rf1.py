import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from quantile_forest import RandomForestQuantileRegressor
import joblib  # Used for saving the model
import os  # Used for file and directory handling
import re
import matplotlib.pyplot as plt
import seaborn as sns

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
# 设置工作目录并加载数据
file_path = 'data/climate_soil.xlsx'  # 替换为你的文件路径
selection = pd.read_excel(file_path)

# 对自变量进行标准化，保持因变量不变
X = selection.drop(columns='RATIO')
y = selection['RATIO']
# 4. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Initialize the models
rf_model = RandomForestRegressor(random_state=42)
gb_model = GradientBoostingRegressor(random_state=42)
qr_rf_model = RandomForestQuantileRegressor(random_state=42)

# 6. Define parameter grids for GridSearchCV
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None]
}

gb_param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

qr_rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None]
}

# 7. GridSearchCV with RandomForestRegressor
rf_grid_search = GridSearchCV(rf_model, rf_param_grid, cv=5, scoring='neg_mean_squared_error')
rf_grid_search.fit(X_train, y_train)

# 8. GridSearchCV with GradientBoostingRegressor
gb_grid_search = GridSearchCV(gb_model, gb_param_grid, cv=5, scoring='neg_mean_squared_error')
gb_grid_search.fit(X_train, y_train)

# 9. GridSearchCV with QuantileRandomForestRegressor
qr_rf_grid_search = GridSearchCV(qr_rf_model, qr_rf_param_grid, cv=5, scoring='neg_mean_squared_error')
qr_rf_grid_search.fit(X_train, y_train)

# 10. Train the models without GridSearchCV
rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)
qr_rf_model.fit(X_train, y_train)

# 11. Predict on the test set
rf_pred = rf_model.predict(X_test)
gb_pred = gb_model.predict(X_test)
qr_rf_pred = qr_rf_model.predict(X_test)

rf_pred_grid = rf_grid_search.best_estimator_.predict(X_test)
gb_pred_grid = gb_grid_search.best_estimator_.predict(X_test)
qr_rf_pred_grid = qr_rf_grid_search.best_estimator_.predict(X_test)

# 12. Evaluate models
def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, r2

# 13. Evaluate the models
rf_mse, rf_r2 = evaluate_model(y_test, rf_pred)
gb_mse, gb_r2 = evaluate_model(y_test, gb_pred)
qr_rf_mse, qr_rf_r2 = evaluate_model(y_test, qr_rf_pred)

rf_mse_grid, rf_r2_grid = evaluate_model(y_test, rf_pred_grid)
gb_mse_grid, gb_r2_grid = evaluate_model(y_test, gb_pred_grid)
qr_rf_mse_grid, qr_rf_r2_grid = evaluate_model(y_test, qr_rf_pred_grid)

# 14. Display results in a table
results = pd.DataFrame({
    'Model': ['RandomForest (No GridSearch)', 'RandomForest (With GridSearch)', 
              'GradientBoosting (No GridSearch)', 'GradientBoosting (With GridSearch)', 
              'QuantileRandomForest (No GridSearch)', 'QuantileRandomForest (With GridSearch)'],
    'MSE': [round(rf_mse, 2), round(rf_mse_grid, 2), round(gb_mse, 2), round(gb_mse_grid, 2), round(qr_rf_mse, 2), round(qr_rf_mse_grid, 2)],
    'R2': [round(rf_r2, 2), round(rf_r2_grid, 2), round(gb_r2, 2), round(gb_r2_grid, 2), round(qr_rf_r2, 2), round(qr_rf_r2_grid, 2)]
})

results_sorted = results.sort_values(by='R2', ascending=False)

# Save the results to a CSV file
results_sorted.to_csv('model_evaluation_results.csv', index=False)
latex_table = results_sorted.to_latex(index=False)
with open('table_results_sorted.tex', 'w') as f:
    f.write(latex_table)

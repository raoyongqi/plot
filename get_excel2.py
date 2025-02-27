# Creating the dataframe based on the provided data
import pandas as pd
model_data = {
    'Model': ['GradientBoostingRegressor', 'RandomForestRegressor', 'HistGradientBoostingRegressor', 
              'LGBMRegressor', 'ExtraTreeRegressor', 'ExtraTreesRegressor', 'BaggingRegressor', 'DecisionTreeRegressor'],
    'Adjusted R-Squared': [0.22, 0.20, 0.18, 0.18, 0.17, 0.17, 0.17, 0.17],
    'R-Squared': [0.31, 0.29, 0.27, 0.27, 0.26, 0.26, 0.26, 0.26],
    'RMSE': [4.99, 5.07, 5.11, 5.13, 5.14, 5.15, 5.15, 5.15],
    'Time Taken': [0.18, 0.43, 0.19, 0.06, 0.01, 0.19, 0.06, 0.01]
}

# Create DataFrame
df_model = pd.DataFrame(model_data)

# Saving the dataframe to an Excel file
model_file_path = 'model_performance.xlsx'
df_model.to_excel(model_file_path, index=False)

model_file_path
import pandas as pd

# Data to be converted into a DataFrame
data = {
    "Model": [
        "QuantileRandomForest (With GridSearch)", 
        "GradientBoosting (No GridSearch)", 
        "GradientBoosting (With GridSearch)", 
        "RandomForest (No GridSearch)", 
        "RandomForest (With GridSearch)", 
        "QuantileRandomForest (No GridSearch)"
    ],
    "MSE": [24.86, 25.24, 25.65, 25.71, 25.75, 27.07],
    "R2": [0.31, 0.30, 0.29, 0.28, 0.28, 0.25]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save to an Excel file
file_path = 'model_comparison.xlsx'
df.to_excel(file_path, index=False)
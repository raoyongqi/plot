import pandas as pd

# Create a dictionary with the given data
data = {
    'Model': ['PyTorch', 'TensorFlow'],
    'MSE': [24.8267, 24.8613],
    'R 方': [0.3091, 0.3081]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to an Excel file
file_path = 'model_comparison.xlsx'
df.to_excel(file_path, index=False)

file_path

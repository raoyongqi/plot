import pandas as pd

data = {
    'Model': ['PyTorch', 'TensorFlow'],
    'MSE': [24.8267, 24.8613],
    'R 方': [0.3091, 0.3081]
}

df = pd.DataFrame(data)

file_path = 'model_comparison.xlsx'
df.to_excel(file_path, index=False)

file_path

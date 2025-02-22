import pandas as pd
import tensorflow as tf

import numpy as np

file_path = 'data/climate_soil_tif.xlsx'  # 替换为你的文件路径
data = pd.read_excel(file_path)
import matplotlib.pyplot as plt

# 处理列名
data.columns = data.columns.str.lower()
data.columns = [col.replace('_resampled', '') if '_resampled' in col else col for col in data.columns]
data.columns = [col.replace('wc2.1_5m_', '') if col.startswith('wc2.1_5m_') else col for col in data.columns]
new_columns = []
for col in data.columns:
    if '_' in col:
        parts = col.split('_')
        if len(parts) > 1 and parts[0] == parts[-1]:
            new_columns.append('_'.join(parts[:1]))
        elif len(parts) > 2 and parts[1] == parts[-1]:
            new_columns.append('_'.join(parts[:2]))
        elif len(parts) > 3 and parts[2] == parts[-1]:
            new_columns.append('_'.join(parts[:2]))
        else:
            new_columns.append(col)
    else:
        new_columns.append(col)

data.columns = new_columns

# 特征和目标变量
feature_columns = [col for col in data.columns]
#Load dataset
dataset = data[feature_columns]
feature_columns = [col for col in data.columns if col != 'ratio']

from sklearn.model_selection import train_test_split

X = data[feature_columns]
data = data.rename(columns={'ratio': 'Pathogen Load'})


y = data['Pathogen Load']  # 目标变量

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=1/9, random_state=42)
print(len(X_train), len(X_valid), len(X_test))
# Plot the outputs and see how they are
ncols=2


import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score


def evaluate(y_true, y_pred):
    mse = np.array([mean_squared_error(y_true[:, idx], y_pred[:,idx]) for idx in range(len(y_true[0]))])
    r2 = np.array([r2_score(y_true[: ,idx], y_pred[:, idx]) for idx in range(len(y_true[0]))])
    rpd = y_true.std()/np.sqrt(mse)
    return mse, r2, rpd

def evaluate_model(model, X, y):
    pred = model.predict(X)
    if(tf.is_tensor(pred)):
        pred = pred.numpy()
        
    return pred, evaluate(y.squeeze(), pred.squeeze())

from sklearn.cross_decomposition import PLSRegression

# define a function to evaluate pls
def pls_evaluate_num_comp(X_train, y_train, X_valid, y_valid, num_comp):
    pls = PLSRegression(n_components=num_comp)
    pls.fit(X_train, y_train)
    y_valid_pred = pls.predict(X_valid)
    mse = mean_squared_error(y_valid_pred, y_valid)
    r2 = r2_score(y_valid_pred, y_valid)
    rpd = y_valid.std()/np.sqrt(mse)
    return (y_valid_pred, mse, r2, rpd)


# Try optimize the number of components (without variable selection) => we will use X1
def pls_evaluate_num_comps(X_train, y_train, X_valid, y_valid, num_comps):
    mses = []
    r2s = []
    rpds = []
    for num_comp in num_comps:
        _, mse, r2, rpd = pls_evaluate_num_comp(X_train, y_train, X_valid, y_valid, num_comp)
        mses.append(mse)
        r2s.append(r2)
        rpds.append(rpd)
    return (mses, r2s, rpds)

def plot_metric(scores, objective, yLabel):
    with plt.style.context('seaborn-v0_8-poster'):
        plt.plot(num_comps, scores, '-o', color='gray', alpha=0.5)
        idx = np.argmin(scores) if objective == 'min' else np.argmax(scores)
        plt.plot(num_comps[idx], scores[idx], 'P', color='red', ms=10)
        plt.xlabel("Number of components")
        plt.ylabel(yLabel)
    plt.show()
    return (num_comps[idx], scores[idx])

def pls_evaluate_plot_num_comps(X_train, y_train, X_valid, y_valid, num_comps):
    mses, r2s, rpds = pls_evaluate_num_comps(X_train, y_train, X_valid, y_valid, num_comps)
    # Plot mses
    num_comp, mse = plot_metric(mses, 'min', 'MSE')
    print(f'The best mse is {mse} with {num_comp} PLS components')

num_comps = np.arange(1, 133)
pls_evaluate_plot_num_comps(X_train, y_train, X_valid, y_valid, num_comps)


import itertools
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

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
def holdout_grid_search(clf, X_train, y_train, X_valid, y_valid, hyperparams, fixed_hyperparams={}):

    all_mses = []
    best_estimator = None
    best_hyperparams = {}
    
    # hold best running score
    best_score = 1000 # set to a very big value

    # get list of param values
    lists = hyperparams.values()
    
    # get all param combinations
    param_combinations = list(itertools.product(*lists))
    total_param_combinations = len(param_combinations)

    # iterate through param combinations
    for i, params in enumerate(param_combinations, 1):
        # fill param dict with params
        param_dict = {}
        for param_index, param_name in enumerate(hyperparams):
            param_dict[param_name] = params[param_index]
            
        # create estimator with specified params
        estimator = clf(**param_dict, **fixed_hyperparams)

        # fit estimator
        estimator.fit(X_train, y_train)
        
        # get predictions on validation set
        preds = estimator.predict(X_valid)
        
        # compute cindex for predictions
        estimator_score = mean_squared_error(y_valid, preds)
        all_mses.append(estimator_score)

        print(f'[{i}/{total_param_combinations}] {param_dict}')
        print(f'Val MSE: {estimator_score}\n')

        # if new low score, update low score, best estimator
        # and best params 
        if estimator_score < best_score:
            best_score = estimator_score
            best_estimator = estimator
            best_hyperparams = param_dict

    # add fixed hyperparamters to best combination of variable hyperparameters
    best_hyperparams.update(fixed_hyperparams)
    
    
    return all_mses, best_estimator, best_hyperparams


hyperparams = {
    'max_depth': [10, 20, 40, 80, 150, 200, 400, 500, 600, 700, 1000, 2000, 3000],
    'n_estimators': [50, 100, 200, 300, 1000]
}
fixed_hyperparams = {
    'random_state': 42
}
def random_forest_grid_search(X_train, y_train, X_valid, y_valid, hyperparams, fixed_hyperparams={}):
    # The random forest regressor
    rf = RandomForestRegressor
    # Search for the best random forest
    all_mses, best_rf, best_hyperparams = holdout_grid_search(rf, X_train, y_train, X_valid, y_valid, hyperparams, fixed_hyperparams)

    print(f"Best hyperparameters:\n{best_hyperparams}")
        
    # add fixed hyperparamters to best combination of variable hyperparameters
    best_hyperparams.update(fixed_hyperparams)
    
    return all_mses, best_rf, best_hyperparams
num_comps = np.arange(1, 133)
all_mses, best_rf, best_hyperparams = random_forest_grid_search(X_train, y_train, X_valid, y_valid, hyperparams, fixed_hyperparams=fixed_hyperparams)
def plot_rf_metric(scores, objective, yLabel):
    with plt.style.context('seaborn-v0_8-poster'):
        num_configs = np.arange(1, len(scores)+1)
        plt.plot(num_configs, scores, '-o', color='gray', alpha=0.5)
        idx = np.argmin(scores) if objective == 'min' else np.argmax(scores)
        plt.plot(num_comps[idx], scores[idx], 'P', color='red', ms=10)
        plt.xlabel("Configuration number")
        plt.ylabel(yLabel)
    plt.show()
    return (num_configs[idx], scores[idx])
plot_rf_metric(all_mses, 'min', 'MSE')
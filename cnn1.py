import tensorflow as tf

import pandas as pd

import shap 

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

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


model = tf.keras.models.Sequential([
    tf.keras.layers.Reshape(target_shape=[133, 1], input_shape=(133,)),
    tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.MaxPool1D(pool_size=2),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.MaxPool1D(pool_size=2),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.MaxPool1D(pool_size=2),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.MaxPool1D(pool_size=2),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mse')
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='auto', restore_best_weights=True)
history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=1000, batch_size=32, callbacks=[es])


def get_shap_values(X_test, model, num_rows=None):
    model1 = tf.keras.models.Model(inputs=[model.inputs], outputs=[model.layers[-1].output[:, 0]])
    model2 = tf.keras.models.Model(inputs=[model.inputs], outputs=[model.layers[-1].output[:, 1]])
    kernel_explainer1 = shap.KernelExplainer(model1.predict,X_test)
    kernel_explainer2 = shap.KernelExplainer(model2.predict,X_test)
    kernel_shap_values1 = np.abs(kernel_explainer1.shap_values(X_test[:num_rows]))
    kernel_shap_values2 = np.abs(kernel_explainer2.shap_values(X_test[:num_rows]))
    kernel_shap_values = np.concatenate([kernel_shap_values1, kernel_shap_values2], axis=0)
    return np.mean(kernel_shap_values, axis=0)

def plot_spectra_vs_model_feature_importance(X, wl, features):
    # Plot spectra
    plt.figure(figsize=(16, 8))
    with plt.style.context('seaborn_v0.8-poster'):
        ax1 = plt.subplot(211)
        plt.plot(wl, X.T)
        plt.ylabel("SG - Values")

        ax2 = plt.subplot(212, sharex=ax1)
        plt.scatter(wl, features, color='gray', edgecolors='black', alpha=0.5)
        

        plt.xlabel("Wavelength (nm)")
        plt.ylabel("feature importances")
        plt.savefig("111.png", bbox_inches='tight')  # Save image
        plt.show()

def evaluate(y_true, y_pred):
    print(y_true.shape)
    print(y_pred.shape)

    # 计算均方误差 (MSE)
    mse = mean_squared_error(y_true, y_pred)
    
    # 计算R²值
    r2 = r2_score(y_true, y_pred)
    
    # 计算相对百分比误差 (RPD)
    rpd = np.std(y_true) / np.sqrt(mse)
    
    return mse, r2, rpd

def evaluation():
    train_df = pd.DataFrame(history.history)
    train_df.plot()
    plt.ylim([0.2,1.0])

    _ = evaluate_and_plot(X_valid, y_valid, model)

    svs = get_shap_values(X_test, model, num_rows)
    plot_spectra_vs_model_feature_importance(X_test, wavelengths, svs)
num_rows = 1

wavelengths = [wl for wl in range(350, 2500+1, 10)]
def evaluate_model(model, X, y):
    pred = model.predict(X)
    if(tf.is_tensor(pred)):
        pred = pred.numpy()
        
    return pred, evaluate(y.squeeze(), pred.squeeze())


ncols=1

variables = ['Pathogen Load']

def evaluate_and_plot(X, y, model):
    y_pred, (mse, r2, rpd) = evaluate_model(model, X, y)
    with plt.style.context('seaborn-poster'):
        fig, axes = plt.subplots(ncols=ncols, nrows=1, figsize=(16, 8))
        for var_idx in range(len(variables)):
            title = f'{variables[var_idx]}, MSE: {np.round(mse[var_idx], 2)}, R2: {np.round(r2[var_idx], 2)}, RPD: {np.round(rpd[var_idx], 2)}'
            # Print the result
            print('MSE: %0.4f' % (mse[var_idx]))
            print('R2: %0.4f' % (r2[var_idx]))
            print('RPD: %0.4f' % (rpd[var_idx]))
            # plot the regression
            p = np.polyfit(y[:, var_idx], y_pred[:, var_idx], deg=1)
            axes[var_idx].scatter(y[:, var_idx], y_pred[:, var_idx], color='gray', edgecolors='black', alpha=0.5)
            axes[var_idx].plot(y[:, var_idx], y[:, var_idx], '-k', label='Expectation')
            axes[var_idx].plot(y[:, var_idx], np.polyval(p, y[:, var_idx]),'-.k', label='Prediction regression')
            axes[var_idx].legend()
            axes[var_idx].set_xlabel('Actual')
            axes[var_idx].set_ylabel('Predicted')
            axes[var_idx].set_title(title)
        plt.plot()
        plt.savefig("222.png", bbox_inches='tight')  # Save image
        plt.show()
    return y_pred, mse, r2, rpd



evaluation()
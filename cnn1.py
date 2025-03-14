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

feature_columns = [col for col in data.columns]

dataset = data[feature_columns]
feature_columns = [col for col in data.columns if col != 'ratio']

from sklearn.model_selection import train_test_split

X = data[feature_columns]
data = data.rename(columns={'ratio': 'Pathogen Load'})


y = data['Pathogen Load']  # 目标变量


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=1/9, random_state=42)


min_epochs = 900  # 设定最小 epoch 数
max_epochs = 1000  # 设定最大 epoch 数
batch_size = 32

while True:
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

    es = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        min_delta=0, 
        patience=100, 
        verbose=1, 
        mode='auto', 
        restore_best_weights=True
    )

    print("Starting training...")
    history = model.fit(
        X_train, y_train, 
        validation_data=(X_valid, y_valid), 
        epochs=max_epochs, 
        batch_size=batch_size, 
        callbacks=[es]
    )

    # 检查 epoch 是否 >= 900，否则重新训练
    if len(history.epoch) >= min_epochs:
        print(f"Training completed with {len(history.epoch)} epochs.")
        break
    else:
        print(f"Early stopping occurred too early ({len(history.epoch)} epochs), restarting training...")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def evaluate(y_true, y_pred):

    print(y_true.shape)
    print(y_pred.shape)

    mse = mean_squared_error(y_true, y_pred)
    
    r2 = r2_score(y_true, y_pred)
    
    rpd = np.std(y_true) / np.sqrt(mse)
    
    return mse, r2, rpd


def evaluate_model(model, X, y):
    pred = model.predict(X)
    if(tf.is_tensor(pred)):
        pred = pred.numpy()
        
    return pred, evaluate(y.squeeze(), pred.squeeze())

def evaluation(X_valid, y_valid):
    """ 评估模型，并绘制左右两张图 """
    y_pred, (mse, r2, rpd) = evaluate_model(model, X_valid, y_valid)
    
    if isinstance(y_valid, (pd.DataFrame, pd.Series)):
        y_valid = y_valid.values
    if isinstance(y_pred, (pd.DataFrame, pd.Series)):
        y_pred = y_pred.values

    y_valid = np.array(y_valid).reshape(-1, 1) if y_valid.ndim == 1 else np.array(y_valid)
    y_pred = np.array(y_pred).reshape(-1, 1) if y_pred.ndim == 1 else np.array(y_pred)

    train_df = pd.DataFrame(history.history)

    with plt.style.context('seaborn-v0_8-poster'):
        fig, axes = plt.subplots(ncols=2, figsize=(16, 8))  # 两张并排的图

        title = f'MSE: {mse:.4f}, R2: {r2:.4f}, RPD: {rpd:.4f}'
        p = np.polyfit(y_valid[:, 0], y_pred[:, 0], deg=1)
        
        axes[0].scatter(y_valid[:, 0], y_pred[:, 0], color='gray', edgecolors='black', alpha=0.5)
        axes[0].plot(y_valid[:, 0], y_valid[:, 0], '-k', label='Expectation')
        axes[0].plot(y_valid[:, 0], np.polyval(p, y_valid[:, 0]), '-.k', label='Prediction regression')

        axes[0].legend()
        axes[0].set_xlabel('Actual')
        axes[0].set_ylabel('Predicted')
        axes[0].set_title(title)

        # 右图：MSE 训练曲线
        axes[1].plot(train_df['loss'], label='Training MSE')
        axes[1].plot(train_df['val_loss'], label='Validation MSE')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MSE')
        axes[1].set_title('Model Training MSE Over Epochs')
        axes[1].legend()
        axes[1].set_ylim([20, 200])  # 使用 axes[1] 来设置 y 轴范围
        axes[1].grid(True)

        plt.savefig("evaluation_results.png", bbox_inches='tight')  # 保存图片
        plt.show()

    return y_pred, mse, r2, rpd



evaluation(X_valid, y_valid)  # 显式传递 X_valid, y_valid


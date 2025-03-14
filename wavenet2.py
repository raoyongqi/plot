import tensorflow as tf

import pandas as pd


import numpy as np


from sklearn.metrics import mean_squared_error, r2_score

file_path = 'data/climate_soil_tif.xlsx'  # 替换为你的文件路径
data = pd.read_excel(file_path)

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


y = data['Pathogen Load']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=1/9, random_state=42)

input_x = tf.keras.layers.Input(shape=(133,))
x = tf.keras.layers.Reshape(target_shape=[133, 1])(input_x)
num_blocks = 2
dilation_rates = (1, 2, 4, 8, 16, 32) 
for _ in range(num_blocks):
    for rate in dilation_rates:
        x = tf.keras.layers.Conv1D(filters=32, kernel_size=2, activation="elu", dilation_rate=rate, padding='valid')(x)
        
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Conv1D(filters=32, kernel_size=1)(x)

x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(1)(x)

model = tf.keras.models.Model(inputs=[input_x], outputs=[x])

model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(0.001))
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='auto', restore_best_weights=True)
history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=1000, batch_size=32, callbacks=[es], verbose=0)
model.save('wavenet2blocks.h5')


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
    """ 评估模型，并将预测结果和训练历史保存为 CSV """
    y_pred, (mse, r2, rpd) = evaluate_model(model, X_valid, y_valid)
    
    if isinstance(y_valid, (pd.DataFrame, pd.Series)):
        y_valid = y_valid.values
    if isinstance(y_pred, (pd.DataFrame, pd.Series)):
        y_pred = y_pred.values

    y_valid = np.array(y_valid).reshape(-1, 1) if y_valid.ndim == 1 else np.array(y_valid)
    y_pred = np.array(y_pred).reshape(-1, 1) if y_pred.ndim == 1 else np.array(y_pred)

    results_df = pd.DataFrame({
        'Actual': y_valid[:, 0],
        'Predicted': y_pred[:, 0]
    })
    results_df.to_csv("wavenet2_actual_vs_predicted.csv", index=False)

    train_df = pd.DataFrame(history.history)
    train_df.to_csv("wavenet2_training_history.csv", index=False)

    return y_pred, mse, r2, rpd
    
evaluation(X_valid, y_valid)
# Transition Layer

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


y = data['Pathogen Load']  # 目标变量


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=1/9, random_state=42)
print(X_train.shape)  # 查看训练数据的形状
X_train = np.array(X_train, dtype=np.float32)
X_valid = np.array(X_valid, dtype=np.float32)

filters = 32
dropout_rate = 0.1
input_x = tf.keras.layers.Input(shape=(133,))
print('input_x shape',input_x.shape)  # 查看训练数据的形状

x = tf.keras.layers.Reshape(target_shape=[133, 1])(input_x)
x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.Conv1D(filters=filters, kernel_size=7, strides = 2, name='conv0', padding='same')(x)
x = tf.keras.layers.MaxPool1D(pool_size=3, strides=2)(x)
x = tf.keras.layers.BatchNormalization()(x)


class DenseBlock(tf.keras.layers.Layer):
    def __init__(self, nb_layers, filters, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        self.nb_layers = nb_layers
        self.filters = filters
        self.dropout_rate = dropout_rate
        self.bottleneck_layer = BottleneckLayer(filters, dropout_rate)
        self.bottleneck_layers = [BottleneckLayer(filters, dropout_rate) for _ in range(nb_layers - 1)]
        self.concatenation = tf.keras.layers.Concatenate()
        
    def call(self, x):
        layers_concat = []
        layers_concat.append(x)
        x = self.bottleneck_layer(x)
        layers_concat.append(x)
        for bottleneck_layer in self.bottleneck_layers:
            x = self.concatenation(layers_concat)
            x = bottleneck_layer(x)
            layers_concat.append(x)

        x = self.concatenation(layers_concat)
        
        return x

class TransitionLayer(tf.keras.layers.Layer):
    def __init__(self, filters, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.dropout_rate = dropout_rate
        self.conv1 = tf.keras.layers.Conv1D(filters=self.filters, kernel_size=1, padding='same')
        self.dropout1 = tf.keras.layers.Dropout(rate=self.dropout_rate)
        self.ap1 = tf.keras.layers.AveragePooling1D(pool_size=2, strides=2)
        
    def call(self, x):
        x = tf.keras.activations.relu(x)
        x = self.conv1(x)
        x = self.dropout1(x)
        x = self.ap1(x)
        return x

# Transition Layer
class BottleneckLayer(tf.keras.layers.Layer):
    def __init__(self, filters, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.dropout_rate = dropout_rate
        self.conv1 = tf.keras.layers.Conv1D(filters=4*self.filters, kernel_size=1, padding='same')
        self.dropout1 = tf.keras.layers.Dropout(rate=self.dropout_rate)
        
        self.conv2 = tf.keras.layers.Conv1D(filters=self.filters, kernel_size=3, padding='same')
        self.dropout2 = tf.keras.layers.Dropout(rate=self.dropout_rate)
        
        
    def call(self, x):
        
        x = tf.keras.activations.relu(x)
        x = self.conv1(x)
        x = self.dropout1(x)
        
        x = tf.keras.activations.relu(x)
        x = self.conv2(x)
        x = self.dropout2(x)
        
        return x

x = DenseBlock(nb_layers=6, filters=filters, dropout_rate=dropout_rate, name='dense1')(x)
x = TransitionLayer(filters, dropout_rate=dropout_rate, name='trans1')(x)

x = DenseBlock(nb_layers=12, filters=filters, dropout_rate=dropout_rate, name='dense2')(x)
x = TransitionLayer(filters, dropout_rate=dropout_rate, name='trans2')(x)

x = DenseBlock(nb_layers=48, filters=filters, dropout_rate=dropout_rate, name='dense3')(x)
x = TransitionLayer(filters, dropout_rate=dropout_rate, name='trans3')(x)

x = DenseBlock(nb_layers=32, filters=filters, dropout_rate=dropout_rate, name='densefinal')(x)

x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.activations.relu(x)
x = tf.keras.layers.GlobalAvgPool1D()(x)
x = tf.keras.layers.Dense(1)(x)

model = tf.keras.models.Model(inputs=[input_x], outputs=[x])

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)  # 限制 L2 范数
model.compile(loss="mse", optimizer=optimizer)

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='auto', restore_best_weights=True)
from tensorflow.keras.callbacks import TerminateOnNaN

history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), 
                    epochs=500, batch_size=32, 
                    callbacks=[es, TerminateOnNaN()])


def evaluate(y_true, y_pred):

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
    results_df.to_csv("denseNet_actual_vs_predicted.csv", index=False)

    train_df = pd.DataFrame(history.history)
    train_df.to_csv("denseNet_training_history.csv", index=False)

    return y_pred, mse, r2, rpd
    
evaluation(X_valid, y_valid)


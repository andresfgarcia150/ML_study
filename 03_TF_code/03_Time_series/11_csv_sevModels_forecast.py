"""
==================================================
Andrés Felipe García Albarracín
2021-06-28
==================================================

Reads csv data and tests several forecast models
using adam as optimizer
"""

"""
==================================================
1. Load libraries
==================================================
"""
import tensorflow as tf
import csv
import numpy as np
import time, os
import matplotlib.pyplot as plt

train_portion = 0.8
window_size = 30
batch_size = 128
"""
==================================================
2. Load data
==================================================
"""
filePath = "../Datasets/Sunspots.csv"
series = []
with open(filePath, 'r') as file:
    reader = csv.reader(file, delimiter = ',')
    print(next(reader))
    for line in reader:
        series.append(float(line[2]))

series = np.array(series)

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(1,1,1)
ax.plot(series)
plt.savefig("97_Imags/TimeSeries_11_01.jpg")

"""
==================================================
3. Split data
==================================================
"""
splitLength = int(train_portion*len(series))
train_series = series[:splitLength]
eval_series = series[splitLength:]

train_ds = tf.data.Dataset.from_tensor_slices(train_series)
eval_ds = tf.data.Dataset.from_tensor_slices(eval_series)

"""
==================================================
4. Preprocessing
==================================================
"""
def preprocessing(dsData: tf.data.Dataset, window_size, batch_size):
    dsData = dsData.window(window_size+1, shift=1, drop_remainder=True)
    dsData = dsData.flat_map(lambda w: w.batch(window_size+1))
    dsData = dsData.map(lambda x: (x[:-1], x[-1]))
    dsData = dsData.shuffle(1000)
    dsData = dsData.batch(batch_size).prefetch(1)
    return dsData

train_ds = preprocessing(train_ds, window_size=window_size, batch_size=batch_size)
eval_ds = preprocessing(eval_ds, window_size=window_size, batch_size=batch_size)

"""
==================================================
5. Callbacks
==================================================
"""
def lrFunc(epoch):
    return (1e-8)*10**(epoch/20)

lr_cb = tf.keras.callbacks.LearningRateScheduler(lrFunc)

es_cb = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=25,
    restore_best_weights=True
)

tb_cb = tf.keras.callbacks.TensorBoard(
    log_dir=os.path.join('99_TensorBoard_logs', time.strftime('run_tseries_11_03_%Y%m%d_%H%M%S'))
)

"""
==================================================
6. Model
==================================================
"""
# Bi-directional LSTM
def createAndCompile(optimizer = tf.keras.optimizers.Adam()):
    model = tf.keras.Sequential([
        tf.keras.layers.Lambda(lambda x: tf.expand_dims(x/200, axis=-1), input_shape=[None]),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1),
        tf.keras.layers.Lambda(lambda x: x*200)
    ])

    model.compile(
        loss = 'mse',
        optimizer=optimizer,
        metrics=['mae']
    )

    return model

# Conv1D + LSTM
def createAndCompile(optimizer = tf.keras.optimizers.Adam()):
    model = tf.keras.Sequential([
        tf.keras.layers.Lambda(lambda x: tf.expand_dims(x/200, axis=-1), input_shape=[None]),
        tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, padding='causal', activation='relu'),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1),
        tf.keras.layers.Lambda(lambda x: x*200)
    ])

    model.compile(
        loss = 'mse',
        optimizer=optimizer,
        metrics=['mae']
    )

    return model

# Simple DNN
def createAndCompile(optimizer = tf.keras.optimizers.Adam()):
    model = tf.keras.Sequential([
        tf.keras.layers.Lambda(lambda x: x/200, input_shape=[window_size]),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1),
        tf.keras.layers.Lambda(lambda x: x*200)
    ])

    model.compile(
        loss = 'mse',
        optimizer=optimizer,
        metrics=['mae']
    )

    return model

model = createAndCompile()
model.fit(
    train_ds,
    validation_data=eval_ds,
    epochs=100,
    callbacks=[es_cb, tb_cb]
)

print(model.evaluate(train_ds))
print(model.evaluate(eval_ds))


"""
-----------------------------------------
Model               Train_MAE   Eval_MAE
-----------------------------------------
Bi-LSTM 32          17.4314     16.8199     <=== Winner and fastest to converge!
Conv1D + LSTM 32    17.2343     16.8820
DNN                 17.2242     17.1831
-----------------------------------------
"""
"""
==================================================
Andrés Felipe García Albarracín
2021-06-28
==================================================

Reads csv data and tests several forecast models
using adam as optimizer
"""
import matplotlib.pyplot as plt
import numpy as np

"""
==================================================
1. Load libraries
==================================================
"""
import tensorflow as tf
import csv
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
filePath = "../Datasets/daily-min-temperatures.csv"
series = []
with open(filePath, 'r') as file:
    reader = csv.reader(file, delimiter = ',')
    next(reader)
    for line in reader:
        series.append(float(line[1]))

series = np.array(series)

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(1,1,1)
ax.plot(series)
plt.savefig('97_Imags/TimeSeries_10_01.jpg')

"""
==================================================
3. Splitting
==================================================
"""
splitLength = int(len(series)*train_portion)
train_series = series[:splitLength]
eval_series = series[splitLength:]

train_ds = tf.data.Dataset.from_tensor_slices(train_series)
eval_ds = tf.data.Dataset.from_tensor_slices(eval_series)

"""
==================================================
4. Preprocessing
==================================================
"""
def preprocessing(dsData: tf.data.Dataset, batch_size, window_size):
    dsData = dsData.window(window_size+1, shift = 1, drop_remainder=True)
    dsData = dsData.flat_map(lambda w: w.batch(window_size+1))
    dsData = dsData.map(lambda x: (x[:-1], x[-1]))
    dsData = dsData.shuffle(1000)
    dsData = dsData.batch(batch_size).prefetch(1)
    return dsData

train_ds = preprocessing(dsData=train_ds, batch_size=batch_size, window_size=window_size)
eval_ds = preprocessing(dsData=eval_ds, batch_size=batch_size, window_size=window_size)

"""
==================================================
5. Callbacks
==================================================
"""
es_cb = tf.keras.callbacks.EarlyStopping(
    patience=25,
    restore_best_weights=True,
    monitor='val_loss'
)

tb_cb = tf.keras.callbacks.TensorBoard(
    log_dir=os.path.join('99_TensorBoard_logs', time.strftime('run_tseries_10_%Y%m%d_%H%M%S'))
)

"""
==================================================
5. Model
==================================================
"""
def createAndCompile(optimizer = tf.keras.optimizers.Adam()):
    model = tf.keras.Sequential([
        tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[None]),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    model.compile(
        loss='mse',
        optimizer=optimizer,
        metrics=['mae']
    )

    return model

model = createAndCompile()
model.fit(
    train_ds,
    validation_data = eval_ds,
    epochs = 300,
    callbacks = [es_cb, tb_cb]
)
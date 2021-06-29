"""
==================================================
Andrés Felipe García Albarracín
2021-06-29
==================================================

Reads csv data and an LSTM forecast model
comparing adam and an optimized SGD
"""

"""
==================================================
1. Load libraries
==================================================
"""
import tensorflow as tf
import numpy as np
import csv
import time, os
import matplotlib.pyplot as plt

train_portion = 0.8
window_size = 64
batch_size = 128
"""
==================================================
2. Load data
==================================================
"""
filePath = '../Datasets/Sunspots.csv'
series = []
with open(filePath, 'r') as file:
    reader = csv.reader(file, delimiter = ',')
    next(reader)
    for line in reader:
        series.append(float(line[2]))

series = np.array(series)

"""
==================================================
3. Split data
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
def preprocessing(dsData: tf.data.Dataset, window_size, batch_size):
    dsData = dsData.window(window_size+1, shift=1, drop_remainder=True)
    dsData = dsData.flat_map(lambda w: w.batch(window_size+1))
    dsData = dsData.map(lambda  x: (x[:-1], x[1:]))
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
    restore_best_weights=True,
    patience=25
)

tb_cb = tf.keras.callbacks.TensorBoard(
    log_dir=os.path.join('99_TensorBoard_logs', time.strftime('run_tseries_13_00_%Y%m%d_%H%M%S'))
)

"""
==================================================
6. Model
==================================================
"""
def createAndCompile(optimizer = tf.keras.optimizers.Adam()):
    model = tf.keras.Sequential([
        tf.keras.layers.Lambda(lambda x: tf.expand_dims(x/200, axis=-1), input_shape=[None]),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(window_size),
        tf.keras.layers.Lambda(lambda x: x*200)
    ])

    model.compile(
        loss = 'mse',
        metrics=['mae'],
        optimizer=optimizer
    )

    return model

"""
==================================================
7. Train
==================================================
"""
model = createAndCompile(tf.keras.optimizers.Adam())

model.fit(
    train_ds,
    validation_data=eval_ds,
    epochs=100,
    callbacks=[es_cb, tb_cb]
)

model.evaluate(train_ds)
model.evaluate(eval_ds)

"""
==================================================
8. Predict
==================================================
"""
forecast = []
predictions = []
for k in range(len(eval_series)-window_size):
    pred = model.predict(eval_series[k:k+window_size][np.newaxis])
    predictions.append(pred[0])

predictions = np.array(predictions)

forecast = predictions[:,-1]
error = tf.keras.losses.mean_absolute_error(eval_series[window_size:], forecast)
print(f"Error: {error}")

forecast = []
for k in range(window_size, len(predictions)-window_size):
    vec = np.zeros(window_size)
    for q in range(window_size):
        vec[q] = predictions[k-window_size+q, window_size-q-1]
    forecast.append(np.mean(vec))

forecast = np.array(forecast)
temp = eval_series[window_size:-window_size]
error = tf.keras.losses.mean_absolute_error(temp[:len(forecast)], forecast)
print(f"Averaged error: {error}")
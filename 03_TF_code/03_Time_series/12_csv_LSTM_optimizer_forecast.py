"""
==================================================
Andrés Felipe García Albarracín
2021-06-28
==================================================

Reads csv data and an LSTM forecast model
comparing adam and an optimized SGD
"""
import csv

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
    dsData = dsData.map(lambda  x: (x[:-1], x[-1]))
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
    log_dir=os.path.join('99_TensorBoard_logs', time.strftime('run_tseries_12_04_%Y%m%d_%H%M%S'))
)

"""
==================================================
6. Model
==================================================
"""
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
        metrics=['mae'],
        optimizer=optimizer
    )

    return model

"""
==================================================
7. Tune the learning rate
==================================================
"""
model = createAndCompile(tf.keras.optimizers.SGD(learning_rate=1e-8, momentum=0.9))
model = createAndCompile(tf.keras.optimizers.Adam(learning_rate=1e-8))

history = model.fit(
    train_ds,
    epochs=200,
    callbacks=[lr_cb]
)

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(1,1,1)
x_axis = lrFunc(np.arange(200))
y_axis = history.history['loss']
ax.semilogx(x_axis[84:153], y_axis[84:153])
plt.savefig('97_Imags/TimeSeries_12_03.jpg')

"""
==================================================
8. Train with optimized learning rate
==================================================
"""
model = createAndCompile(tf.keras.optimizers.Adam())
#model = createAndCompile(tf.keras.optimizers.Adam(learning_rate=4e-4))

model.fit(
    train_ds,
    validation_data=eval_ds,
    epochs=100,
    callbacks=[es_cb, tb_cb]
)

print(model.evaluate(train_ds))
print(model.evaluate(eval_ds))

"""
Model: Bi-LSTM 32
-----------------------------------------
Optimizer           Train_MAE   Eval_MAE
-----------------------------------------
Tuned SGD           17.5388     16.8959
Adam                17.4103     16.9919
Tuned Adam          17.5262     16.8881
Adam LSTM 64*       17.3646     16.7897
Adam LSTM 32*       17.1050     16.4877
-----------------------------------------
* The window size is 64
"""


"""
==================================================
9. Predictions
==================================================
"""
forecast = []
for k in range(len(eval_series)-window_size):
    pred = model.predict(eval_series[k:k+window_size][np.newaxis])
    forecast.append(pred[0,0])

forecast = np.array(forecast)
error = tf.keras.losses.mean_absolute_error(eval_series[window_size:], forecast)
print(f"Error: {error}")
"""
==================================================
Andrés Felipe García Albarracín
2021-06-28
==================================================

Generates artificial data and uses a simple DNN model
to forecast
"""

"""
==================================================
1. Load libraries
==================================================
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os, time

seed = 42
rnd = np.random.RandomState(seed)
train_portion = 0.8
window_size = 20
batch_size = 128
"""
==================================================
2. Generate data
==================================================
"""
time_series = np.arange(4 * 365 + 1, dtype ="float32")

def genTrend(time, slope = 0):
    return time*slope

def genSeasonality(time, period, phase_time = 0, amp = 1):
    season_time = ((time + phase_time)%period)/period
    return np.where(season_time < 0.1,
             np.cos(season_time * 7 * np.pi),
             1 / np.exp(5 * season_time))*amp

def genAutoCorrelation(time, spikes = 10, ampS = 1, phi = {1,0}):
    sig = np.zeros(len(time))
    for k in rnd.randint(0,len(time),spikes):
        sig[k] += rnd.rand()*ampS

    for k in range(len(time)):
        for lag, amp in phi.items():
            if k > lag:
                sig[k] += sig[k-lag]*amp
    return sig

def genNoise(time, level):
    return rnd.randn(len(time))*level

series = genTrend(time_series, 0.1)
baseline = 10
amplitude = 40
slope = 0.01
noise_level = 2
period = 365
window_size = 10

# Create the series
series = baseline + genTrend(time_series, slope) + genSeasonality(time_series, period=365, amp=amplitude)
series += genNoise(time_series, noise_level)

"""
==================================================
3. Splitting and dataset creation
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
def preprocessing(dsData: tf.data.Dataset, batch_size = 32, window_size = 20):
    dsData = dsData.window(window_size+1, shift=1, drop_remainder=True)
    dsData = dsData.flat_map(lambda win: win.batch(window_size+1))
    dsData = dsData.map(lambda x: (x[:-1], x[-1]))
    dsData = dsData.shuffle(1000)
    dsData = dsData.batch(batch_size)
    dsData = dsData.prefetch(1)
    return dsData

train_ds = preprocessing(train_ds, batch_size, window_size)
eval_ds = preprocessing(eval_ds, batch_size, window_size)

"""
==================================================
5. Callbacks
==================================================
"""
lr_cb = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8*10**(epoch/20)
)

tb_cb_lr = tf.keras.callbacks.TensorBoard(
    log_dir=os.path.join('99_TensorBoard_logs', time.strftime("run_tseries_lr_01_%Y%m%d_%H%M%S"))
)

es_cb = tf.keras.callbacks.EarlyStopping(
    patience=25,
    restore_best_weights=True,
    monitor='val_loss'
)

tb_cb = tf.keras.callbacks.TensorBoard(
    log_dir=os.path.join('99_TensorBoard_logs', time.strftime("run_tseries_01_%Y%m%d_%H%M%S"))
)

"""
==================================================
6. Model
==================================================
"""
model = tf.keras.Sequential([
    tf.keras.layers.Dense(20, activation='relu', input_shape=[window_size]),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(
    loss = 'mse',
    optimizer=tf.keras.optimizers.SGD(learning_rate=1e-8, momentum=0.9)
)

history = model.fit(
    train_ds,
    epochs=100,
    callbacks=[lr_cb, tb_cb_lr]
)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
loss_series = history.history['loss']
lr_series = 1e-8*10**(np.arange(len(loss_series))/20)
ax.semilogx(lr_series, loss_series)

plt.savefig('97_Imags/TimeSeries_01_lr.jpg')

"""
==================================================
7. Train
==================================================
"""
model = tf.keras.Sequential([
    tf.keras.layers.Dense(20, activation='relu', input_shape=[window_size]),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(
    loss = 'mse',
    optimizer = 'adam',#tf.keras.optimizers.SGD(learning_rate=6e-5, momentum=0.9),
    metrics=['mae']
)

history2 = model.fit(
    train_ds,
    validation_data=eval_ds,
    epochs=100,
    callbacks=[es_cb, tb_cb]
)

model.evaluate(eval_ds)
model.evaluate(train_ds)
"""
==================================================
8. Predict
==================================================
"""
forecast = []
for k in range(len(eval_series)-window_size):
    pred = model.predict(eval_series[k:k+window_size][np.newaxis])
    forecast.append(pred[0][0])

forecast = np.array(forecast)
error = tf.keras.losses.mean_absolute_error(eval_series[window_size:], forecast)
print(error)
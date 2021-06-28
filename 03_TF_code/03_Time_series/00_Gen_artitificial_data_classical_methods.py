"""
==================================================
Andrés Felipe García Albarracín
2021-06-28
==================================================

Generates artificial data and uses classical methods
to forecast data
"""

"""
==================================================
1. Load libraries
==================================================
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
seed = 42
rnd = np.random.RandomState(seed)
"""
==================================================
2. Generate data
==================================================
"""
time = np.arange(4*365+1, dtype = "float32")

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

series = genTrend(time, 0.1)
baseline = 10
amplitude = 40
slope = 0.01
noise_level = 2
period = 365
window_size = 10

# Create the series
series = baseline + genTrend(time, slope) + genSeasonality(time, period=365, amp=amplitude)
series += genNoise(time, noise_level)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(time,series)
plt.savefig("97_Imags/TimeSeries_00_01.jpg")
"""
==================================================
3. Clasical forecasting
==================================================
"""
# 1. Differencing
sig_dif = series[period:] - series[:-period]

# 2. Averaging
sig_dif_avg = np.zeros(len(sig_dif)-window_size)
for step in range(len(sig_dif_avg)):
    sig_dif_avg[step] = np.mean(sig_dif[step:step+window_size])

# 3. Adding an average of the past window
sig_avg = np.zeros(len(series))
for step in range(len(sig_avg)):
    sig_avg[step] = np.mean(series[step:step+window_size])

forecast = np.zeros(len(series)-period-window_size)
for step in range(len(forecast)):
    forecast[step] = sig_dif_avg[step] + sig_avg[step+int(window_size/2)]

fig = plt.figure(figsize=(20,6))
ax = fig.add_subplot(1,2,1)
ax.plot(time[period+window_size:3*period],series[period+window_size:3*period])
ax.plot(time[period+window_size:3*period],forecast[:2*period-window_size])
ax2 = fig.add_subplot(1,2,2)
ax2.plot(time[period+window_size:int(3*period/2)],series[period+window_size:int(3*period/2)])
ax2.plot(time[period+window_size:int(3*period/2)],forecast[:int(period/2)-window_size])
plt.savefig("97_Imags/TimeSeries_00_02.jpg")

mae = tf.keras.metrics.mean_absolute_error(series[period+window_size:3*period], forecast[:2*period-window_size]).numpy()
print(f"MAE: {mae}")
mse = tf.keras.metrics.mean_squared_error(series[period+window_size:3*period], forecast[:2*period-window_size]).numpy()
print(f"MSE: {mse}")
rmse = np.sqrt(mse)
print(f"RMSE: {rmse}")
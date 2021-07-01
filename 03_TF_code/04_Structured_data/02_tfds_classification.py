"""
==================================================
Andrés Felipe García Albarracín
2021-06-29
==================================================

Reads data from tfds and builds a multi-class
classification model
"""

"""
==================================================
1. Load libraries
==================================================
"""
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import time, os

"""
==================================================
2. Load data
==================================================
"""
dsTrain, dsEval = tfds.load('wine_quality', split=['train[:80%]', 'train[80%:]'])

"""
==================================================
3. Cast to numpy
==================================================
"""
train_data = []
train_labels = []
eval_data = []
eval_labels = []

for item in dsTrain:
    vals = [val.numpy() for val in item['features'].values()]
    train_data.append(vals[:-1])
    train_labels.append(vals[-1])

train_data = np.array(train_data)
train_labels = np.array(train_labels)

for item in dsEval:
    vals = [val.numpy() for val in item['features'].values()]
    eval_data.append(vals[:-1])
    eval_labels.append(vals[-1])

eval_data = np.array(eval_data)
eval_labels = np.array(eval_labels)

"""
==================================================
4. Normalize
==================================================
"""
num_cols = train_data.shape[1]
for col in range(num_cols):
    mean_val = np.mean(train_data[:,col])
    std_val = np.std(train_data[:,col])
    train_data[:, col] = (train_data[:,col]-mean_val)/std_val
    eval_data[:, col] = (eval_data[:, col] - mean_val) / std_val

"""
==================================================
5. Callbacks
==================================================
"""
es_cb = tf.keras.callbacks.EarlyStopping(
    patience=50,
    monitor='val_loss',
    restore_best_weights=True
)

tb_cb = tf.keras.callbacks.TensorBoard(
    log_dir=os.path.join('99_TensorBoard_logs', time.strftime('run_struct_02_%Y%m%d_%H%M%S'))
)

"""
==================================================
6. Model
==================================================
"""
model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    loss = 'mse',
    metrics=['mae'],
    optimizer='adam'
)

model.fit(
    x = train_data,
    y = train_labels,
    validation_data=(eval_data, eval_labels),
    epochs=300,
    callbacks=[es_cb, tb_cb]
)
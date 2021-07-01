"""
==================================================
Andrés Felipe García Albarracín
2021-06-29
==================================================

Reads data from tf keras dataset and builds a
regression model
"""
import os.path
import time

"""
==================================================
1. Load libraries
==================================================
"""
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
"""
==================================================
2. Load data
==================================================
"""
dsTrain, dsEval = tfds.load('iris', split=['train[:80%]', 'train[-20%:]'], as_supervised=True)

"""
==================================================
3. Cast to numpy
==================================================
"""
train_data = []
train_labels = []
eval_data = []
eval_labels = []

for data, labels in dsTrain:
    train_data.append(data.numpy())
    train_labels.append(labels.numpy())

train_data = np.array(train_data)
train_labels = np.array(train_labels)

for data, labels in dsEval:
    eval_data.append(data.numpy())
    eval_labels.append(labels.numpy())

eval_data = np.array(eval_data)
eval_labels = np.array(eval_labels)

"""
==================================================
4. Normalize
==================================================
"""

num_cols = train_data.shape[1]
for col in range(num_cols):
    mean_val = np.mean(train_data[:, col])
    std_val = np.mean(train_data[:, col])
    train_data[:, col] = (train_data[:, col]-mean_val)/std_val
    eval_data[:, col] = (eval_data[:, col] - mean_val) / std_val

"""
==================================================
5. Callbacks
==================================================
"""

es_cb = tf.keras.callbacks.EarlyStopping(
    patience=75,
    restore_best_weights=True,
    monitor='val_accuracy'
)

tb_cb = tf.keras.callbacks.TensorBoard(
    log_dir=os.path.join('99_TensorBoard_logs', time.strftime('run_struct_01_%Y%m%d_%H%M%S'))
)
"""
==================================================
5. Model
==================================================
"""
model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
    optimizer='adam'
)

model.fit(
    x = train_data,
    y = train_labels,
    validation_data=(eval_data, eval_labels),
    epochs=300,
    callbacks=[es_cb, tb_cb]
)

dsTrain = tfds.load('rock_you')
dsTrain = dsTrain['train']
elem = []
for item in dsTrain:
    elem.append(str(item['password'].numpy()))

elem = np.array(elem)

for item in dsTrain.take(1):
    print(item)
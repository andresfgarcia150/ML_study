"""
==================================================
Andrés Felipe García Albarracín
2021-06-17
==================================================

Imports tfds with own split
"""
import time

"""
==================================================
1. Load libraries
==================================================
"""
import tensorflow as tf
import tensorflow_datasets as tfds
import time, os

"""
==================================================
2. Download data and preprocessing
==================================================
"""
ds_train, ds_eval = tfds.load('tf_flowers', split=['train[:80%]','train[-20%:]'], as_supervised=True)

def preProcessing(dsData: tf.data.Dataset):
    dsData = dsData.map(lambda imag, label: (tf.image.resize(imag, [150,150]), label))
    dsData = dsData.map(lambda imag, label: (imag/255.0, label))
    dsData = dsData.batch(10).prefetch(1)
    return dsData

ds_train = preProcessing(dsData=ds_train)
ds_eval = preProcessing(dsData=ds_eval)

"""
==================================================
3. Callbacks
==================================================
"""

class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.99:
            print('Accuracy achieved')
            self.model.stop_training = True

my_cb = MyCallback()
es_cb = tf.keras.callbacks.EarlyStopping(
    patience=15,
    restore_best_weights=True,
    monitor='val_accuracy'
)

tb_cb = tf.keras.callbacks.TensorBoard(
    log_dir=os.path.join('99_TensorBoard_logs', time.strftime("run_flowers_%Y%m%d-%H_%M_%S"))
)

"""
==================================================
4. Model
==================================================
"""
base_model = tf.keras.applications.xception.Xception(
    include_top=False,
    weights='imagenet',
    input_shape=(150,150,3)
)

for layer in base_model.layers:
    layer.trainable = False

last_layer = base_model.get_layer('add_11')
last_output = last_layer.output

x = tf.keras.layers.Flatten()(last_output)
x = tf.keras.layers.Dropout(0.4)(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dropout(0.4)(x)
x = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)

model = tf.keras.Model(inputs=base_model.input, outputs = x)

model.compile(
    loss = 'sparse_categorical_crossentropy',
    optimizer = 'adam',
    metrics=['accuracy']
)

history = model.fit(
    ds_train,
    epochs=100,
    validation_data=ds_eval,
    callbacks=[my_cb, es_cb, tb_cb]
)

"""
==================================================
5. Plot
==================================================
"""
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np

fig = plt.figure(figsize = (10,10))
ax = fig.add_subplot(1,2,1)
epochs = np.arange(len(history.history['accuracy'])) + 1
ax.plot(epochs, history.history['accuracy'])
ax.plot(epochs, history.history['val_accuracy'])

ax = fig.add_subplot(1,2,2)
epochs = np.arange(len(history.history['loss'])) + 1
ax.plot(epochs, history.history['loss'])
ax.plot(epochs, history.history['val_loss'])

fig.savefig('figure.png')
"""
==================================================
Andrés Felipe García Albarracín
2021-06-16
==================================================

Imports tfds image with ImageDataGenerator,
uses data augmentation, tensorboard, Dropout and
Transfer Learning
"""
import time

"""
==================================================
1. Load libraries
==================================================
"""
import tensorflow as tf
import tensorflow_datasets as tfds
import os, time

"""
==================================================
2. Load data and process 
==================================================
"""

ds_train, ds_test = tfds.load('rock_paper_scissors', split=['train', 'test'], as_supervised=True)

ds_train = ds_train.map(lambda imag, label: (tf.image.resize(imag, [150,150]), label))
ds_train = ds_train.map(lambda imag, label: (imag/255.0, label))
ds_train = ds_train.batch(10).prefetch(1)

ds_test = ds_test.map(lambda imag, label: (tf.image.resize(imag, [150,150]), label))
ds_test = ds_test.map(lambda imag, label: (imag/255.0, label))
ds_test = ds_test.batch(10).prefetch(1)

"""
==================================================
3. Define the callbacks
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
    log_dir=os.path.join('99_TensorBoard_logs', time.strftime("run_rock_sissors_%Y%m%d-%H%M%S"))
)

"""
==================================================
4. Create the model
==================================================
"""
base_model = tf.keras.applications.inception_v3.InceptionV3(
    include_top=False,
    weights='imagenet',
    input_shape = (150,150,3)
)

for layer in base_model.layers:
    layer.trainable = False

last_output = base_model.get_layer('mixed7').output

x = tf.keras.layers.Flatten()(last_output)
x = tf.keras.layers.Dropout(0.4)(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dropout(0.4)(x)
x = tf.keras.layers.Dense(3, activation=tf.nn.softmax)(x)

model = tf.keras.Model(
    inputs = base_model.input,
    outputs = x
)

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

"""
==================================================
5. Training
==================================================
"""

model.fit(
    ds_train,
    epochs=100,
    validation_data=ds_test,
    callbacks=[my_cb,es_cb,tb_cb]
)

modelFile = os.path.join(os.curdir, '98_Saved_models', '210617_RockPaperSissors.h5')
model.save(modelFile)
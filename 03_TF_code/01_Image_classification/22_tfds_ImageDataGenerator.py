"""
==================================================
Andrés Felipe García Albarracín
2021-06-17
==================================================

Imports tfds with ImageDataGenerator
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
import numpy as np

"""
==================================================
2. Download data and preprocess it
==================================================
"""
ds_train, ds_eval = tfds.load('rock_paper_scissors', split=['train', 'test'], as_supervised=True)
ds_train = ds_train.map(lambda imag, label: (tf.image.resize(imag, [150,150]), label))
ds_eval = ds_eval.map(lambda imag, label: (tf.image.resize(imag, [150,150]), label))
imag_train = []
label_train = []
imag_eval = []
label_eval = []

for imag, label in ds_train:
    imag_train.append(imag.numpy())
    label_train.append(label.numpy())

for imag, label in ds_eval:
    imag_eval.append(imag.numpy())
    label_eval.append(label.numpy())

imag_train = np.array(imag_train)
label_train = np.array(label_train)

imag_eval = np.array(imag_eval)
label_eval = np.array(label_eval)

"""
==================================================
3. Use the ImageDataGenerator
==================================================
"""
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1/255.0,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow(
    x = imag_train,
    y = label_train,
    batch_size=10
)

eval_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1/255.0
)

eval_generator = eval_datagen.flow(
    x = imag_eval,
    y = label_eval,
    batch_size=10
)

"""
==================================================
4. Callbacks 
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
5. Model
==================================================
"""
base_model = tf.keras.applications.inception_v3.InceptionV3(
    include_top=False,
    weights='imagenet',
    input_shape=(150,150,3)
)

for layer in base_model.layers:
    layer.trainable = False

last_output = base_model.get_layer('mixed7').output

x = tf.keras.layers.Flatten()(last_output)
x = tf.keras.layers.Dropout(0.4)(x)
x = tf.keras.layers.Dense(1042, activation='relu')(x)
x = tf.keras.layers.Dropout(0.4)(x)
x = tf.keras.layers.Dense(3, activation=tf.nn.softmax)(x)

model = tf.keras.Model(inputs = base_model.input, outputs = x)

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.fit(
    train_generator,
    steps_per_epoch=252,
    epochs=100,
    validation_data=eval_generator,
    validation_steps=37,
    callbacks=[my_cb,es_cb,tb_cb]
)
"""
==================================================
Andrés Felipe García Albarracín
2021-06-16
==================================================

Imports local image data with ImageDataGenerator,
uses data augmentation, tensorboard, Dropout and
Transfer Learning
"""

"""
==================================================
1. Load libraries
==================================================
"""
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
import time, os, zipfile

"""
==================================================
2. Download data
==================================================
"""

#!wget --no-check-certificate https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip -O ../Datasets/cats_and_dogs_filtered.zip

train_folder = '../Datasets/cats_and_dogs/cats_and_dogs_filtered/train'
eval_folder = '../Datasets/cats_and_dogs/cats_and_dogs_filtered/validation'
'''
zipRef = zipfile.ZipFile('../Datasets/cats_and_dogs_filtered.zip', 'r')
zipRef.extractall('../Datasets/cats_and_dogs')
zipRef.close()
'''

"""
==================================================
3. Preprocessing data
==================================================
"""

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale = 1/255.0,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    directory=train_folder,
    batch_size=10,
    target_size=(150,150),
    class_mode='binary'
)

eval_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1/255.0
)

eval_generator = eval_datagen.flow_from_directory(
    directory=eval_folder,
    batch_size=10,
    target_size=(150,150),
    class_mode='binary'
)

"""
==================================================
5. Callbacks
==================================================
"""
class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.99:
            print('99% Accuracy achieved')
            self.model.stop_training = True

my_cb = MyCallback()
es_cb = tf.keras.callbacks.EarlyStopping(
    patience=15,
    restore_best_weights=True,
    monitor='val_accuracy'
)
tb_cb = tf.keras.callbacks.TensorBoard(
    log_dir=os.path.join('99_TensorBoard_logs', time.strftime("run_%Y%m%d-%H_%M_%S"))
)

"""
==================================================
4. Model
==================================================
"""

base_model = InceptionV3(
    include_top=False,
    weights='imagenet',
    input_shape=(150,150,3)
)

#base_model.load_weights()

for layer in base_model.layers:
    layer.trainable = False

last_layer = base_model.get_layer('mixed7')
last_output = last_layer.output

x = tf.keras.layers.Flatten()(last_output)
x = tf.keras.layers.Dropout(0.4)(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dropout(0.4)(x)
x = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(x)

model = tf.keras.Model(inputs = base_model.input, outputs = x)

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

"""
==================================================
5. Training
==================================================
"""
model.fit(
    train_generator,
    steps_per_epoch=200,
    epochs=100,
    validation_data=eval_generator,
    validation_steps=100,
    callbacks=[my_cb, es_cb, tb_cb]
)
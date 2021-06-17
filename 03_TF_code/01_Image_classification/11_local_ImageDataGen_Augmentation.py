"""
==================================================
Andrés Felipe García Albarracín
2021-06-16
==================================================

Imports local image data with ImageDataGenerator,
uses data augmentation, tensorboard and Dropout
"""

"""
==================================================
1. Load libraries
==================================================
"""
import tensorflow as tf
import zipfile
import time
import os

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
3. Load data and augment it
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

train_generator = train_datagen.flow_from_directory(
    directory=train_folder,
    target_size=(150,150),
    batch_size=10,
    class_mode='binary'
)

eval_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1/255.0
)
eval_generator = eval_datagen.flow_from_directory(
    directory=eval_folder,
    target_size=(150,150),
    batch_size=10,
    class_mode='binary'
)

"""
==================================================
4. Define callbacks
==================================================
"""
class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy')>0.99:
            print('Accuracy achieved. Stop')
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
5. Create the model
==================================================
"""
dropRate = 0.4
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32,(3,3), activation='relu', input_shape=[150,150,3]),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(128,(3,3), activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Dropout(dropRate),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(dropRate),
    tf.keras.layers.Dense(1,activation=tf.nn.sigmoid)
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.fit(
    train_generator,
    steps_per_epoch=200,
    epochs=100,
    validation_data=eval_generator,
    validation_steps=100,
    callbacks=[my_cb, es_cb, tb_cb]
)
"""
==================================================
Andrés Felipe García Albarracín
2021-06-17
==================================================

Imports tf.keras.datasets and use ImageDataGenerator
"""


"""
==================================================
1. Load libraries
==================================================
"""
import tensorflow as tf
import time, os
import numpy as np

"""
==================================================
2. Load data
==================================================
"""
cifar10 = tf.keras.datasets.cifar10

(train_imags, train_labels), (eval_imags, eval_labels) = cifar10.load_data()

train_shape = train_imags.shape
eval_shape = eval_imags.shape

#train_imags = train_imags/255.0 #Rescale done at the ImageDataGenerator
#eval_imags = eval_imags/255.0

"""
==================================================
3. Preprocess data
==================================================
"""
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1/255.0,
    rotation_range=40,
    height_shift_range=0.2,
    width_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow(
    x = train_imags,
    y = train_labels,
    batch_size=10
)

eval_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1/255.0
)

eval_generator = eval_datagen.flow(
    x = eval_imags,
    y = eval_labels,
    batch_size=10
)
"""
==================================================
4. Define the callbacks
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
    log_dir=os.path.join('99_TensorBoard_logs', time.strftime("run_cifar10_%Y%m%d-%H%M%S"))
)

"""
==================================================
5. Model
==================================================
"""
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=[32,32,3]),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.fit(
    train_generator,
    steps_per_epoch=5000,
    epochs=100,
    validation_data=eval_generator,
    validation_steps=1000,
    callbacks=[my_cb,es_cb,tb_cb]
)

"""
==================================================
6. Save model
==================================================
"""
modelFolder = os.path.join(os.curdir, '98_Saved_models', '210617_cifar10')
modelFile = os.path.join(os.curdir, '98_Saved_models', '210617_cifar10.h5')
#os.mkdir(modelFolder)
model.save(modelFolder)
model.save(modelFile)

new_model = tf.keras.models.load_model(modelFile)
new_model.evaluate(eval_generator)
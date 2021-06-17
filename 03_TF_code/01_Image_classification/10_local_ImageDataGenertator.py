"""
==================================================
Andrés Felipe García Albarracín
2021-06-15
==================================================

Imports local image data with ImageDataGenerator
"""

"""
==================================================
1. Load libraries
==================================================
"""
import tensorflow as tf
import zipfile
import os

"""
==================================================
2. Download data
==================================================
"""
# wget --no-check-certificate https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip -O ../Datasets/horse-or-human.zip
# wget --no-check-certificate https://storage.googleapis.com/laurencemoroney-blog.appspot.com/validation-horse-or-human.zip -O ../Datasets/validation-horse-or-human.zip

trainFolder = os.path.join(os.pardir, 'Datasets', 'horse-or-human', 'train')
evalFolder = os.path.join(os.pardir, 'Datasets', 'horse-or-human', 'eval')
'''
zipRef = zipfile.ZipFile('../Datasets/horse-or-human.zip', 'r')
zipRef.extractall(trainFolder)
zipRef = zipfile.ZipFile('../Datasets/validation-horse-or-human.zip', 'r')
zipRef.extractall(evalFolder)
zipRef.close()
'''

"""
==================================================
3. Preprocess data
==================================================
IMPORTANT: Keep small the batch size to avoid 
consuming all of the GPU memory! (e.g. 32)
"""
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1 / 255.0
)
train_generator = train_datagen.flow_from_directory(
    trainFolder,
    target_size=(300, 300),
    batch_size=32,
    class_mode='binary'
)
eval_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1 / 255.0
)
eval_generator = eval_datagen.flow_from_directory(
    directory=evalFolder,
    target_size=(300, 300),
    batch_size=32,
    class_mode='binary'
)
"""
==================================================
4. Callbacks
==================================================
"""


class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('val_accuracy') > 0.9:
            print('Target validation accuracy obtained. Finishing training')
            self.model.stop_training = True


my_cb = MyCallback()
es_cb = tf.keras.callbacks.EarlyStopping(
    patience=10,
    restore_best_weights=True,
    monitor='accuracy'
)
"""
==================================================
5. Model
==================================================
"""
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=[300, 300, 3]),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    # tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.fit(
    train_generator,
    steps_per_epoch=32,
    epochs=32,
    validation_data=eval_generator,
    validation_steps=8,
    callbacks=[my_cb, es_cb]
)

"""
==================================================
6. Miscellaneous
==================================================
Select randomly a file and predict the conv layers'
outputs of the file
"""
layersOut = [layer.output for layer in model.layers if layer.name.startswith('conv2d')]
modelMis = tf.keras.Model(inputs=model.inputs, outputs=layersOut)

human_files = [evalFolder + '/humans/' + file for file in os.listdir(evalFolder + '/humans')]
horse_files = [evalFolder + '/horses/' + file for file in os.listdir(evalFolder + '/horses')]

import random
file = random.choice(human_files + horse_files)

img = tf.keras.preprocessing.image.load_img(
    file,
    target_size=(300,300)
)
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = img_array.reshape(1,300,300,3)
result = modelMis.predict(img_array)

prediction = model.predict(img_array)
print(f'file {file}, prediction {prediction}')
print(f'classes: {train_generator.class_indices}')
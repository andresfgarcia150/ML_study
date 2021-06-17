"""
==================================================
Andrés Felipe García Albarracín
2021-06-11
==================================================

Imports image data from tf.keras.datasets
"""

"""
==================================================
1. Load libraries
==================================================
"""
import tensorflow as tf
import numpy as np

"""
==================================================
2. Load data
==================================================
"""
f_mnist = tf.keras.datasets.fashion_mnist
(train_imags, train_labels), (test_imags, test_labels) = f_mnist.load_data()
train_shape = train_imags.shape
test_shape = test_imags.shape

train_imags = train_imags.reshape(train_shape[0], train_shape[1], train_shape[2],1)/255.0
test_imags = test_imags.reshape(test_shape[0], test_shape[1], test_shape[2],1)/255.0

"""
==================================================
3. Build callbacks
==================================================
"""
class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs['accuracy'] > 0.99:
            print('Training stopped because the accuracy is above 99%')
            self.model.stop_training = True

my_cb = MyCallback()
es_cb = tf.keras.callbacks.EarlyStopping(
    patience = 10,
    restore_best_weights = True,
    monitor='accuracy'
)
"""
==================================================
3. Build the model
==================================================
"""
model1 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28,28,1]),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model2 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(128,(3,3),activation='relu',input_shape=[28,28,1]),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

"""
==================================================
4. Compile the model
==================================================
"""
model1.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model2.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

"""
==================================================
5. Model fit
==================================================
"""
model1.fit(
    x = train_imags,
    y = train_labels,
    validation_data= (test_imags, test_labels),
    epochs=30,
    callbacks=[my_cb, es_cb]
)

model2.fit(
    x = train_imags,
    y = train_labels,
    validation_data=(test_imags, test_labels),
    epochs = 30,
    callbacks=[my_cb, es_cb]
)

"""
==================================================
6. Inspect some outputs
==================================================
"""
layerOut = [layer.output for layer in model2.layers]
model3 = tf.keras.models.Model(inputs = model2.input, outputs = layerOut)

"""
==================================================
7. Predict some results
==================================================
"""
test_predictions = model2.predict(x=test_imags)
test_accuracy = model2.evaluate(x=test_imags, y=test_labels)
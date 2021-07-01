"""
==================================================
Andrés Felipe García Albarracín
2021-06-30
==================================================

Loads data from a csv file, reads it with pandas,
process it and then trains a model
"""

"""
==================================================
1. Load libraries
==================================================
"""
import tensorflow as tf
import pandas as pd

train_portion = 0.8
"""
==================================================
2. Load data
==================================================
"""
file = tf.keras.utils.get_file('abalone_train.csv', "https://storage.googleapis.com/download.tensorflow.org/data/abalone_train.csv")
df = pd.read_csv(
    file,
    names=["Length", "Diameter", "Height", "Whole weight", "Shucked weight",
           "Viscera weight", "Shell weight", "Age"])

x_data = df[df.columns[:-1]].to_numpy()
y_data = df.iloc[:,-1].to_numpy()

"""
==================================================
3. Split
==================================================
"""
splitLength = int(len(x_data)*train_portion)
x_train = x_data[:splitLength]
y_train = y_data[:splitLength]
x_eval = x_data[splitLength:]
y_eval = y_data[splitLength:]

"""
==================================================
4. Callbacks
==================================================
"""
es_cb = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    restore_best_weights=True,
    patience=30
)

"""
==================================================
5. Model
==================================================
"""
model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(
    loss = 'mse',
    metrics=['mae'],
    optimizer='adam'
)

model.fit(
    x = x_train,
    y = y_train,
    validation_data=(x_eval, y_eval),
    epochs=300,
    callbacks=[es_cb]
)
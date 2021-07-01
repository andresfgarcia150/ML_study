"""
==================================================
Andrés Felipe García Albarracín
2021-06-29
==================================================

Reads data from tf keras dataset and builds a
regression model
"""

"""
==================================================
1. Load libraries
==================================================
"""
import tensorflow as tf
import numpy as np
import time, os

"""
==================================================
2. Load data
==================================================
"""
boston_housing = tf.keras.datasets.boston_housing
(x_train, y_train), (x_eval, y_eval) = boston_housing.load_data()

"""
==================================================
3. Normalization (if needed)
==================================================
"""
num_cols = x_train.shape[1]
for col in range(num_cols):
    mean_val = np.mean(x_train[:,col])
    std_val = np.std(x_train[:,col])
    x_train[:, col] = (x_train[:,col]-mean_val)/std_val
    x_eval[:, col] = (x_eval[:, col] - mean_val) / std_val

"""
==================================================
4. Callbacks
==================================================
"""
es_cb = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    restore_best_weights=True,
    patience=75
)
tb_cb = tf.keras.callbacks.TensorBoard(
    log_dir=os.path.join('99_TensorBoard_logs', time.strftime('run_struct_00_%Y%m%d_%H%M%S'))
)

"""
==================================================
5. Model
==================================================
"""
model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x: x*20)
])

model.compile(
    loss = 'mse',
    metrics=['mae', tf.keras.metrics.RootMeanSquaredError()],
    optimizer='adam'
)

model.fit(
    x = x_train,
    y = y_train,
    validation_data=(x_eval, y_eval),
    epochs=300,
    callbacks=[es_cb, tb_cb]
)

print(model.evaluate(x=x_train, y=y_train))
print(model.evaluate(x=x_eval, y=y_eval))
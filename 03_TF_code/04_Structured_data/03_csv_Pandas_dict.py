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
csv_file = tf.keras.utils.get_file('heart.csv','https://storage.googleapis.com/download.tensorflow.org/data/heart.csv')
df = pd.read_csv(csv_file)

"""
==================================================
3. Preprocessing
==================================================
"""
numerics_cols = list(df.columns)
numerics_cols.remove('thal')
numerics_cols.remove('target')
categorical_cols = ['thal']
num_cat = {}

for col in numerics_cols:
    mean_val = df[col].mean()
    std_val = df[col].std()
    df[col] = (df[col]-mean_val)/std_val

for col in categorical_cols:
    new_df = pd.get_dummies(df[col], col)
    df = pd.concat([df, new_df], axis = 1)
    df.pop(col)

# Shuffle
df = df.sample(frac=1).reset_index(drop=True)

target = df.pop('target')

splitLength = int(len(df)*train_portion)
dsTrain = tf.data.Dataset.from_tensor_slices((df[:splitLength].to_dict('list'), target[:splitLength])) # df.to_dict('list')
dsEval = tf.data.Dataset.from_tensor_slices((df[splitLength:].to_dict('list'), target[splitLength:]))

dsTrain = dsTrain.batch(32)
dsEval = dsEval.batch(32)

"""
==================================================
4. Model
==================================================
"""
es_cb = tf.keras.callbacks.EarlyStopping(
    patience=15,
    restore_best_weights=True,
    monitor='val_accuracy'
)

"""
==================================================
5. Model
==================================================
"""
inputs = {col: tf.keras.layers.Input(shape=1, name=col) for col in df.columns}
x = tf.stack(list(inputs.values()), axis=-1)
x = tf.keras.layers.Dense(16, activation='relu')(x)
x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs = inputs, outputs=x)

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.fit(
    dsTrain,
    validation_data=dsEval,
    epochs=100,
    callbacks=[es_cb]
)
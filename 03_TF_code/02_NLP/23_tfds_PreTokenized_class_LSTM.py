"""
==================================================
Andrés Felipe García Albarracín
2021-06-22
==================================================

Loads pre-tokenized data from tfds, and trains
a classification problem.

"""

"""
==================================================
1. Load libraries
==================================================
"""
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import time, os

batch_size = 32
embedding_dim = 16
"""
==================================================
2. Load data
==================================================
"""

(ds_train, ds_eval), info = tfds.load(
    'imdb_reviews/subwords8k',
    as_supervised=True,
    split=['train', 'test'],
    with_info=True
)

tokenizer = info.features['text'].encoder
vocab_size = tokenizer.vocab_size

"""
==================================================
3. Pre-process data
==================================================
"""
ds_train = ds_train.shuffle(1000)
ds_eval = ds_eval.shuffle(1000)

ds_train = ds_train.padded_batch(batch_size=batch_size)
ds_eval = ds_eval.padded_batch(batch_size=batch_size)
"""
==================================================
4. Callbacks
==================================================
"""
class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs['accuracy'] > 0.99:
            print('Achieved accuracy')
            self.model.stop_training = True

my_cb = MyCallback()
es_cb = tf.keras.callbacks.EarlyStopping(
    patience=15,
    restore_best_weights=True,
    monitor='val_accuracy'
)
tb_cb = tf.keras.callbacks.TensorBoard(
    log_dir='./99_TensorBoard_logs/' + time.strftime('run_NLP_23_%Y%m%d-%H%M%S')
)

"""
==================================================
5. Model
==================================================
"""
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_shape=[None]),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(8, return_sequences=True)),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(4)),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(
    loss = 'binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.fit(
    ds_train,
    validation_data=ds_eval,
    epochs=100,
    callbacks=[my_cb, es_cb, tb_cb]
)
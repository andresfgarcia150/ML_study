"""
==================================================
Andrés Felipe García Albarracín
2021-06-19
==================================================

Loads data from tf.keras.datasets
"""
import time

"""
==================================================
1. Load libraries
==================================================
"""
import tensorflow as tf
import numpy as np
import time, os

vocab_size = 10000
embedding_dim = 64
max_length = 2000
trunc_type = 'post'
pad_type = 'post'

"""
==================================================
2. Load data
==================================================
"""
reuters = tf.keras.datasets.reuters
(train_texts, train_labels), (eval_texts, eval_labels) = reuters.load_data(
    num_words=vocab_size
)

"""
==================================================
3. Pad sequences
==================================================
"""
train_padded = tf.keras.preprocessing.sequence.pad_sequences(
    train_texts,
    padding=pad_type,
    truncating=trunc_type,
    maxlen=max_length
)

eval_padded = tf.keras.preprocessing.sequence.pad_sequences(
    eval_texts,
    padding=pad_type,
    truncating=trunc_type,
    maxlen=max_length
)

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
    log_dir='./99_TensorBoard_logs/' + time.strftime('run_NLP_30_%Y%m%d-%H%M%S')
)

"""
==================================================
5. Model
==================================================
"""
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_shape=[max_length]),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(embedding_dim-2, activation='relu'),
    tf.keras.layers.Dense(46, activation=tf.nn.softmax)
])

model.compile(
    loss = 'sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.fit(
    x = train_padded,
    y = train_labels,
    validation_data=(eval_padded, eval_labels),
    epochs=100,
    callbacks=[my_cb, es_cb, tb_cb]
)
"""
==================================================
Andrés Felipe García Albarracín
2021-06-19
==================================================

Loads data from tf.keras.datasets, which is already
tokenized, uses a LSTM model
"""

"""
==================================================
1. Load libraries
==================================================
"""

import tensorflow as tf
import numpy as np
import time, os

vocab_size = 10000
trunc_type = 'post'
pad_type = 'post'
embedding_dim = 16
max_length = 120
"""
==================================================
2. Import data
==================================================
"""

reuters = tf.keras.datasets.reuters
(train_sequences, train_labels), (eval_sequences, eval_labels) = reuters.load_data(
    num_words=vocab_size
)
vocab_size = len(reuters.get_word_index()) + 2
num_classes = len(np.unique(train_labels))

long = 0
for item in train_sequences:
    long = (len(item) if len(item) > long else long)
"""
==================================================
3. Pad sequences
==================================================
"""
train_padded = tf.keras.preprocessing.sequence.pad_sequences(
    train_sequences,
    truncating=trunc_type,
    padding=pad_type,
    maxlen=max_length
)

train_padded = np.array(train_padded)

eval_padded = tf.keras.preprocessing.sequence.pad_sequences(
    eval_sequences,
    truncating=trunc_type,
    padding=pad_type,
    maxlen=max_length
)

eval_padded = np.array(eval_padded)

"""
==================================================
4. Callbacks
==================================================
"""
class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs['accuracy'] > 0.99:
            print('Accuracy achieved')
            self.model.stop_training = True

my_cb = MyCallback()
es_cb = tf.keras.callbacks.EarlyStopping(
    patience=15,
    restore_best_weights=True,
    monitor='val_accuracy'
)
tb_cb = tf.keras.callbacks.TensorBoard(
    log_dir=os.path.join('99_TensorBoard_logs', time.strftime('run_NLP_31_%Y%m%d_%H%M%S'))
)

"""
==================================================
5. Model
==================================================
"""
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_shape=[None]),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16, return_sequences=True)),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(8)),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(16, activation='sigmoid'),
    tf.keras.layers.Dense(num_classes, activation='sigmoid')
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
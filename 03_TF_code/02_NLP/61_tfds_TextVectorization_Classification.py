"""
==================================================
Andrés Felipe García Albarracín
2021-07-02
==================================================

Loads data from TFDS, builds a Text Vectorization layer
and a model for classification
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
import re, string

vocab_size = 10000
max_length = 100
embedding_dim = 32
batch_size = 128
"""
==================================================
2. Load data
==================================================
"""

"""
# Use the following to work with numpy arrays
dsTrain, dsEval = tfds.load('imdb_reviews', as_supervised=True, split=['train', 'test'], batch_size=-1)
train_texts, train_labels = tfds.as_numpy(dsTrain)
eval_texts, eval_labels = tfds.as_numpy(dsEval)
"""

dsTrain, dsEval = tfds.load('imdb_reviews', as_supervised=True, split=['train', 'test'])
dsTrain = dsTrain.batch(batch_size)
dsEval = dsEval.batch(batch_size)

"""
==================================================
3. Initialize the vectorization layer
==================================================
"""
def preprocess_text(input_txt):
    x = tf.strings.lower(input_txt)
    x = tf.strings.regex_replace(x, '<br \>', '')
    x = tf.strings.regex_replace(x, re.escape(string.punctuation), '')
    return x

text_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(
    max_tokens=vocab_size,
    output_mode='int',
    standardize='lower_and_strip_punctuation',
    output_sequence_length=max_length,
    input_shape=[None]
)

dsText = dsTrain.map(lambda x,y: x)

text_layer.adapt(dsText)

"""
==================================================
4. Callbacks
==================================================
"""
class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.99:
            print('Accuracy achieved')
            self.model.stop_training = True

my_cb = MyCallback()

es_cb = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    restore_best_weights=True,
    patience=25
)

tb_cb = tf.keras.callbacks.TensorBoard(
    log_dir=os.path.join('99_TensorBoard_logs', time.strftime('run_NLP_61_%Y%m%d_%H%M%S'))
)

"""
==================================================
5. Model
==================================================
"""
model = tf.keras.Sequential([
    text_layer,
    tf.keras.layers.Embedding(vocab_size, embedding_dim),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    metrics=['accuracy'],
    optimizer='adam'
)

model.fit(
    dsTrain,
    validation_data=dsEval,
    epochs=100,
    callbacks=[my_cb, es_cb, tb_cb]
)
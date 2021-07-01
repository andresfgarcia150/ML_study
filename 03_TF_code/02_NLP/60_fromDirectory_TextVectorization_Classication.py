"""
==================================================
Andrés Felipe García Albarracín
2021-06-30
==================================================

Loads data from a folder, vectors it with Text
Vectorization and solves a classification problem
"""

"""
==================================================
1. Load libraries
==================================================
"""
import tensorflow as tf
import numpy as np
import os
import shutil
import re
import string

vocab_size = 10000
max_length = 100
batch_size = 128
seed = 123
embedding_dim = 64
"""
==================================================
2. Load data
==================================================
"""
url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
file = tf.keras.utils.get_file("aclImdb_v1.tar.gz", url, untar=True)

dataset_dir = os.path.join(os.path.dirname(file), 'aclImdb')
os.listdir(dataset_dir)

train_dir = os.path.join(dataset_dir, 'train')
os.listdir(train_dir)

eval_dir = os.path.join(dataset_dir, 'test')
os.listdir(eval_dir)

remove_dir = os.path.join(train_dir, 'unsup')
shutil.rmtree(remove_dir)

train_ds = tf.keras.preprocessing.text_dataset_from_directory(
    train_dir,
    batch_size=batch_size
    #validation_split=0.2, subset='training', seed=seed
)

eval_ds = tf.keras.preprocessing.text_dataset_from_directory(
    eval_dir,
    batch_size=batch_size
)

"""
==================================================
3. Vectorization
==================================================
"""
# Create a custom standardization function to strip HTML break tags '<br />'.
def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
  return tf.strings.regex_replace(stripped_html,
                                  '[%s]' % re.escape(string.punctuation), '')

vectorizationLayer = tf.keras.layers.experimental.preprocessing.TextVectorization(
    max_tokens=vocab_size,
    standardize=custom_standardization,
    output_mode='int',
    output_sequence_length=max_length
)
text_ds = train_ds.map(lambda x,y: x)
vectorizationLayer.adapt(text_ds)

"""
==================================================
6. Callbacks
==================================================
"""
es_cb = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    restore_best_weights=True,
    patience=25
)

"""
==================================================
5. Model
==================================================
"""
model = tf.keras.Sequential([
    vectorizationLayer,
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
    train_ds,
    validation_data=eval_ds,
    epochs=100,
    callbacks=[es_cb]
)
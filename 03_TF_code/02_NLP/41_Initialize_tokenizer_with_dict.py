"""
==================================================
Andrés Felipe García Albarracín
2021-06-24
==================================================

Loads data from a csv, tokenizes it then creates
another tokenizer out of it
"""

"""
==================================================
1. Load libraries
==================================================
"""
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

vocab_size = 10000
oov_tok = '<OOV>'
"""
==================================================
2. Load data
==================================================
"""
ds_train, ds_eval = tfds.load(
    'imdb_reviews',
    as_supervised=True,
    split=['train', 'test']
)

train_texts = []
train_labels = []
eval_texts = []
eval_labels = []

for text, label in ds_train:
    sentence = str(text.numpy())
    train_texts.append(sentence)
    train_labels.append(label.numpy)

for text, label in ds_eval:
    sentence = str(text.numpy())
    eval_texts.append(sentence)
    eval_labels.append(label.numpy)

train_labels = np.array(train_labels)
eval_labels = np.array(eval_labels)

"""
==================================================
3. Tokenize
==================================================
"""
tokenizer = tf.keras.preprocessing.text.Tokenizer(
    num_words=vocab_size,
    oov_token=oov_tok
)
tokenizer.fit_on_texts(train_texts)
original_word_index = tokenizer.word_index
train_sequences_orig = tokenizer.texts_to_sequences(train_texts)
del tokenizer

tokenizer2 = tf.keras.preprocessing.text.Tokenizer(
    num_words=vocab_size,
    oov_token=oov_tok
)
tokenizer2.word_index = original_word_index
train_sequences = tokenizer2.texts_to_sequences(train_texts)
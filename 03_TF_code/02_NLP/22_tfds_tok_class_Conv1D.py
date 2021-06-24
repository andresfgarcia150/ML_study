"""
==================================================
Andrés Felipe García Albarracín
2021-06-21
==================================================

Loads data from tfds, tokenizes it and
performs multi-class Classification with a
bidirectional LSTM
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

vocab_size = 30000
oov_tok = '<OOV>'
trunc_type = 'post'
pad_type = 'post'
max_length = 120
embedding_dim = 32
train_portion = 0.8

stopwords = [ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", \
              "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do",\
              "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", \
              "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", \
              "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", \
              "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", \
              "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such",\
              "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these",\
              "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until",\
              "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", \
              "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll",\
              "you're", "you've", "your", "yours", "yourself", "yourselves" ]

"""
==================================================
2. Load data
==================================================
"""
ds_train, ds_eval = tfds.load(
    'imdb_reviews',
    split=['train', 'test'],
    as_supervised=True
)

train_texts = []
train_labels = []
eval_texts = []
eval_labels = []

long = 0
for text, label in ds_train:
    text = str(text.numpy())
    words = [word for word in text.split(" ")]# if word not in stopwords]
    long = (len(words) if len(words) > long else long)
    sentence = ' '.join(words)
    sentence = sentence.replace('  ', ' ')
    train_texts.append(sentence)
    train_labels.append(label.numpy())

train_labels = np.array(train_labels)

for text, label in ds_eval:
    text = str(text.numpy())
    words = [word for word in text.split(" ")]# if word not in stopwords]
    long = (len(words) if len(words) > long else long)
    sentence = ' '.join(words)
    sentence = sentence.replace('  ', ' ')
    eval_texts.append(sentence)
    eval_labels.append(label.numpy())

eval_labels = np.array(eval_labels)
"""
==================================================
4. Tokenize
==================================================
"""
tokenizer = tf.keras.preprocessing.text.Tokenizer(
    num_words=vocab_size,
    oov_token=oov_tok
)

tokenizer.fit_on_texts(train_texts)
train_sequences = tokenizer.texts_to_sequences(train_texts)
eval_sequences = tokenizer.texts_to_sequences(eval_texts)

"""
==================================================
5. Padding
==================================================
"""
train_padded = tf.keras.preprocessing.sequence.pad_sequences(
    train_sequences,
    maxlen=max_length,
    truncating=trunc_type,
    padding=pad_type
)

train_padded = np.array(train_padded)

eval_padded = tf.keras.preprocessing.sequence.pad_sequences(
    eval_sequences,
    maxlen=max_length,
    truncating=trunc_type,
    padding=pad_type
)

eval_padded = np.array(eval_padded)

"""
==================================================
6. Callbacks
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
    log_dir=os.path.join('99_TensorBoard_logs', time.strftime('run_NLP_22_%Y%m%d-%H%M%S') )
)

"""
==================================================
7. Model
==================================================
"""
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_shape=[max_length]),
    tf.keras.layers.Conv1D(128, 5, activation='relu'),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dropout(0.6),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dropout(0.6),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    loss = 'binary_crossentropy',
    metrics = ['accuracy'],
    optimizer='adam'
)

model.fit(
    x = train_padded,
    y = train_labels,
    validation_data=(eval_padded, eval_labels),
    epochs=100,
    callbacks=[my_cb, es_cb, tb_cb]
)
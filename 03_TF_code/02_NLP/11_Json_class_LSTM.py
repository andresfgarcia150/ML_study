"""
==================================================
Andrés Felipe García Albarracín
2021-06-21
==================================================

Loads data from a json file, tokenizes it and
performs multi-class Classification with a
bidirectional LSTM
"""

"""
==================================================
1. Load libraries
==================================================
"""
import tensorflow as tf
import numpy as np
import json
import time, os

vocab_size = 10000
oov_tok = '<OOV>'
trunc_type = 'post'
pad_type = 'post'
max_length = 32
embedding_dim = 16
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

texts = []
labels = []

with open('../Datasets/sarcasm.json', 'r') as file:
    fileContents = json.load(file)

long = 0
for content in fileContents:
    sentence = content['headline']
    # In this case, including the stopwords degrades the performance
    words = [word for word in sentence.split(" ")]# if word not in stopwords]
    long = (len(words) if len(words) > long else long)
    sentence = ' '.join(words)
    sentence = sentence.replace('  ', ' ')
    texts.append(sentence)
    labels.append(int(content['is_sarcastic']))

labels = np.array(labels)
"""
==================================================
3. Split data
==================================================
"""
splitLength = int(len(texts)*train_portion)
train_texts = texts[:splitLength]
eval_texts = texts[splitLength:]

train_labels = labels[:splitLength]
eval_labels = labels[splitLength:]

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
    log_dir=os.path.join('99_TensorBoard_logs', time.strftime('run_NLP_11_%Y%m%d-%H%M%S') )
)

"""
==================================================
7. Model
==================================================
"""
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_shape=[max_length]),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16, return_sequences=True)),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(8)),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(10, activation='relu'),
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
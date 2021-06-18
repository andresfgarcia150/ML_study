"""
==================================================
Andrés Felipe García Albarracín
2021-06-18
==================================================

Loads data from tfds, converts into numpy arrays,
tokenizes it and performs multi-class Classification
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

vocab_size = 10000
oov_tok = '<OOV>'
pad_type = 'post'
trunc_type = 'post'
max_length = 2000
embedding_dim = 32
"""
==================================================
2. Download data
==================================================
"""
ds_train, ds_eval = tfds.load('imdb_reviews', split=['train', 'test'], as_supervised=True)

train_texts = []
eval_texts = []
train_labels = []
eval_labels = []

long = 0
for text, label in ds_train:
    sentence = str(text.numpy())
    words = [word for word in sentence.split(' ')] # if word not in stopwords]
    long = (len(words) if len(words) > long else long)
    sentence = ' '.join(words)
    sentence = sentence.replace('  ', ' ')
    train_texts.append(sentence)
    train_labels.append(label.numpy())

for text, label in ds_eval:
    sentence = str(text.numpy())
    words = [word for word in sentence.split(' ')] # if word not in stopwords]
    sentence = ' '.join(words)
    sentence = sentence.replace('  ', ' ')
    eval_texts.append(sentence)
    eval_labels.append(label.numpy())

"""
==================================================
3. Tokenize data
==================================================
"""
tokenizer = tf.keras.preprocessing.text.Tokenizer(
    num_words=vocab_size,
    oov_token=oov_tok
)

tokenizer.fit_on_texts(train_texts)
#tokenizer.fit_on_texts(eval_texts)

train_sequences = tokenizer.texts_to_sequences(train_texts)
eval_sequences = tokenizer.texts_to_sequences(eval_texts)
"""
==================================================
4. Pad sequences
==================================================
"""
train_padded = tf.keras.preprocessing.sequence.pad_sequences(
    train_sequences,
    padding=pad_type,
    truncating=trunc_type,
    maxlen=max_length
)

eval_padded = tf.keras.preprocessing.sequence.pad_sequences(
    eval_sequences,
    padding=pad_type,
    truncating=trunc_type,
    maxlen=max_length
)

train_padded = np.array(train_padded)
eval_padded = np.array(eval_padded)
train_labels = np.array(train_labels)
eval_labels = np.array(eval_labels)

"""
==================================================
5. Callbacks
==================================================
"""
class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') > 0.99):
            print('Accuracy achieved')
            self.model.stop_training = True

my_cb = MyCallback()
es_cb = tf.keras.callbacks.EarlyStopping(
    patience=15,
    restore_best_weights=True,
    monitor='val_accuracy'
)

tb_cb = tf.keras.callbacks.TensorBoard(
    log_dir=os.path.join('99_TensorBoard_logs', time.strftime("run_NLP_20_%Y%m%d-%H%M%S"))
)

"""
==================================================
6. Model
==================================================
"""
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_shape=[max_length]),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    loss = 'binary_crossentropy',
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
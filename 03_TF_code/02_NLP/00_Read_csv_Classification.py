"""
==================================================
Andrés Felipe García Albarracín
2021-06-17
==================================================

Loads data from a csv file, tokenizes it and
performs multi-class Classification
"""

"""
==================================================
1. Load libraries
==================================================
"""
import tensorflow as tf
import csv
import os, time
import numpy as np

vocab_size = 1000
oov_tok = '<OOV>'
trunc_type = 'post'
padd_type = 'post'
max_length = 120
embedding_dim = 16
train_portion = 0.8
"""
==================================================
2. Download data
==================================================
"""
# !wget --no-check-certificate https://storage.googleapis.com/laurencemoroney-blog.appspot.com/bbc-text.csv -O Datasets/bbc-text.csv

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

labels = []
texts = []
with open('../Datasets/bbc-text.csv', 'r') as file:
    reader = csv.reader(file, delimiter = ',')
    next(reader)
    for line in reader:
        labels.append(line[0])
        sentence = line[1]
        """
        # Option 1
        for fword in stopwords:
            sentence = sentence.replace(" " + fword + " ", " ")
            sentence = sentence.replace("  ", " ")
        """
        # Option 2
        words = [word for word in sentence.split(" ") if word not in stopwords]
        sentence = " ".join(words)
        sentence = sentence.replace("  ", " ")
        texts.append(sentence)

"""
==================================================
3. Tokenize data
==================================================
"""
tokenizer = tf.keras.preprocessing.text.Tokenizer(
    num_words = vocab_size,
    oov_token=oov_tok
)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

label_tokenizer = tf.keras.preprocessing.text.Tokenizer()
label_tokenizer.fit_on_texts(labels)
# IMPORTANT: Do not forget to substract 1 to use cero indexing
tokenized_labels = np.array(label_tokenizer.texts_to_sequences(labels))-1

"""
==================================================
4. Pad sequences
==================================================
"""
padded = tf.keras.preprocessing.sequence.pad_sequences(
    sequences,
    truncating=trunc_type,
    padding=padd_type,
    maxlen=max_length
)

padded = np.array(padded)

"""
==================================================
5. Split data
==================================================
"""
splitLength = int(len(padded)*train_portion)

train_padded = padded[:splitLength]
eval_padded = padded[splitLength:]

train_labels = tokenized_labels[:splitLength]
eval_labels = tokenized_labels[splitLength:]

"""
==================================================
6. Callbacks
==================================================
"""
class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.99:
            print("Achieve accuracy")
            self.model.stop_training = True

my_cb = MyCallback()
es_cb = tf.keras.callbacks.EarlyStopping(
    patience=15,
    restore_best_weights=True,
    monitor='val_accuracy'
)
tb_cb = tf.keras.callbacks.TensorBoard(
    log_dir=os.path.join('99_TensorBoard_logs', time.strftime('run_NLP_00_%Y%m%d-%H$M%S'))
)
"""
==================================================
6. Model
==================================================
"""
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(len(label_tokenizer.word_index), activation=tf.nn.softmax)
])

model.compile(
    loss = 'sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.fit(
    x = train_padded,
    y = train_labels,
    validation_data = (eval_padded, eval_labels),
    epochs=100,
    callbacks=[my_cb, es_cb, tb_cb]
)
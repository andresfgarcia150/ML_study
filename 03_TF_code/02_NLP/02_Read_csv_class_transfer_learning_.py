"""
==================================================
Andrés Felipe García Albarracín
2021-06-23
==================================================

Loads data from a csv, tokenizes it and uses
already-trained embeddings
"""

"""
==================================================
1. Load libraries
==================================================
"""
import tensorflow as tf
import numpy as np
import csv, time, os

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

train_portion = 0.8
vocab_size = 10000
oov_tok = '<OOV>'
pad_type = 'post'
trunc_type = 'post'
max_length = 120
embedding_dim = 50
"""
==================================================
2. Load data
==================================================
"""
filePath = "../Datasets/nlp_stanford_sentiment.csv"
texts = []
labels = []
lines = []
with open(filePath, 'r') as file:
    reader = csv.reader(file, delimiter = ',')
    for line in reader:
        lines.append(line)

np.random.shuffle(lines)

for line in lines:
    labels.append(int(line[0]))
    sentence = line[5]
    words = [word for word in sentence.split(" ")]# if word not in stopwords]
    sentence = " ".join(words)
    sentence = sentence.replace("  ", " ")
    texts.append(sentence)

labels = [(label if label == 0 else 1) for label in labels]
labels = np.array(labels)
"""
==================================================
3. Build train and evaluation arrays
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
    padding=pad_type,
    truncating=trunc_type,
    maxlen=max_length
)

train_padded = np.array(train_padded)

eval_padded = tf.keras.preprocessing.sequence.pad_sequences(
    eval_sequences,
    padding=pad_type,
    truncating=trunc_type,
    maxlen=max_length
)

eval_padded = np.array(eval_padded)

"""
==================================================
6. Load weights
==================================================
"""
import sys
import csv
csv.field_size_limit(sys.maxsize)

weightFile = f"../Datasets/glove.6B/glove.6B.{embedding_dim}d.txt"
word_index = {}
for num, key in enumerate(tokenizer.word_index):
    if num == vocab_size-1:
        break
    word_index[key] = tokenizer.word_index[key]

weight_matrix = np.zeros((vocab_size, embedding_dim))
with open(weightFile, 'r') as file:
    for line in file:
        elements = line.split(" ")
        if word_index.get(elements[0]) is not None:
            weight_matrix[word_index.get(elements[0]),:] = np.array(elements[1:]).astype(float)

"""
==================================================
7. Callbacks
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
    log_dir=os.path.join('99_TensorBoard_logs', time.strftime("run_NLP_02_%Y%m%d-%H%M%S"))
)

"""
==================================================
8. Model
==================================================
"""
embedding_layer = tf.keras.layers.Embedding(
    vocab_size,
    embedding_dim,
    #weights=[weight_matrix],
    input_shape=[None]
)
model = tf.keras.Sequential([
    embedding_layer,
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(10, activation='relu'),
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
    batch_size=1024,
    callbacks=[my_cb, es_cb, tb_cb]
)
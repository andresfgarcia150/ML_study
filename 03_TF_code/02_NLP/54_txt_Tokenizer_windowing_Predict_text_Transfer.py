"""
==================================================
Andrés Felipe García Albarracín
2021-06-26
==================================================

Loads data from a txt file, tokenizes it and builds
a model to predict text with pre-trained embeddings
"""

"""
==================================================
1. Load libraries
==================================================
"""
import tensorflow as tf
import numpy as np
import time, os

window_size = 10
trunc_type = 'pre'
embedding_dim = 100
train_portion = 0.9
"""
==================================================
2. Load data
==================================================
"""
filePath = "../Datasets/sonnets.txt"
lines = []
with open(filePath, 'r') as file:
    lines = file.readlines()
# Obs: they include the symbol \n

"""
==================================================
3. Tokenizer
==================================================
"""
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(lines)

orig_sequences = tokenizer.texts_to_sequences(lines)

orig_sequences_flat = []
for ls in orig_sequences:
    for elem in ls:
        orig_sequences_flat.append(elem)
"""
==================================================
4. Build sequences
==================================================
"""
sequences_labels = []
for elem in range(len(orig_sequences_flat)-(window_size)):
    sequences_labels.append(orig_sequences_flat[elem: elem+window_size+1])
"""
==================================================
5. Padding
==================================================
"""
padded_labels = tf.keras.preprocessing.sequence.pad_sequences(
    sequences_labels,
    truncating=trunc_type
)

padded_labels = np.array(padded_labels)
#np.random.shuffle(padded_labels)

"""
==================================================
6. Splitting
==================================================
"""
padded = padded_labels[:,:-1]
labels = padded_labels[:,-1]-1

splitLength = int(len(padded)*train_portion)
train_padded = padded[:splitLength]
eval_padded = padded[splitLength:]
train_labels = labels[:splitLength]
eval_labels = labels[splitLength:]

"""
==================================================
7. Reading weights
==================================================
"""
weightFile = f"../Datasets/glove.6B/glove.6B.{embedding_dim}d.txt"

word_index = tokenizer.word_index
num_words = len(word_index)
weights_matrix = np.zeros((num_words+1, embedding_dim))
with open(weightFile, 'r') as file:
    for line in file:
        elements = line.split(" ")
        if word_index.get(elements[0]) is not None:
            weights_matrix[word_index.get(elements[0]),:] = np.array(elements[1:]).astype(float)

"""
==================================================
8. Callbacks
==================================================
"""
class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.9:
            print('Accuracy achieved')
            self.model.stop_training = True

my_cb = MyCallback()
es_cb = tf.keras.callbacks.EarlyStopping(
    patience=50,
    restore_best_weights=True,
    monitor='val_accuracy'
)

tb_cb = tf.keras.callbacks.TensorBoard(
    log_dir=os.path.join('99_TensorBoard_logs', time.strftime('run_NLP_54_%Y%m%d-%H%M%S'))
)

"""
==================================================
9. Model
==================================================
"""
# A. Simple model without Transfer Learning
model = tf.keras.Sequential([
    tf.keras.layers.Embedding((num_words+1), embedding_dim, input_shape=[None]),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(1024)),
    tf.keras.layers.Dense(num_words, activation=tf.nn.softmax)
])

# B. Simple model with Transfer Learning
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(num_words+1, embedding_dim, weights=[weights_matrix], input_shape=[None]),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(1024)),
    tf.keras.layers.Dense(num_words, activation='softmax')
])

# C. Complex model with Transfer Learning
"""
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(num_words+1, embedding_dim, weights=[weights_matrix], input_shape=[None]),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(1024, return_sequences=True)),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.LSTM(1042),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(1042, activation='relu'),
    tf.keras.layers.Dense(num_words, activation=tf.nn.softmax)
])
"""

model.compile(
    loss = 'sparse_categorical_crossentropy',
    optimizer = 'adam',
    metrics=['accuracy']
)

model.fit(
    x = train_padded,
    y = train_labels,
    validation_data=(eval_padded, eval_labels),
    epochs=100,
    batch_size=32,
    callbacks=[my_cb, es_cb]#, tb_cb]
)

"""
Model comparison
Model   Accuracy    Validation Accuracy
A       0.7294      0.0585
B       0.7317      0.0692
C       0.7084      0.0689
"""

"""
==================================================
10. Text prediction
==================================================
"""
index_word = {}
for word, index in word_index.items():
    index_word[index-1] = word

seed_text = "Help me Obi Wan Kenobi, you're my only hope"
next_words = 100
for iter in range(next_words):
    seq = tokenizer.texts_to_sequences([seed_text])[0]
    new_index = np.argmax(model.predict([seq]))
    new_word = index_word[new_index]
    seed_text = seed_text + " " + new_word
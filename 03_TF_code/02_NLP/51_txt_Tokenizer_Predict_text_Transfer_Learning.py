"""
==================================================
Andrés Felipe García Albarracín
2021-06-24
==================================================

Loads data from a txt file, tokenizes it and builds
a model to predict text
"""

"""
==================================================
1. Load libraries
==================================================
"""
import tensorflow as tf
import numpy as np
import time, os

embedding_dim = 100
trunc_type = 'pre'
pad_type = 'pre'
train_portion = 0.8
"""
==================================================
2. Load data
==================================================
"""
filePath = "../Datasets/irish-lyrics-eof.txt"
lines = []
with open(filePath, 'r') as file:
    lines = file.readlines()

"""
==================================================
3. Tokenize
==================================================
"""
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(lines)
orig_sequences = tokenizer.texts_to_sequences(lines)

"""
==================================================
4. Build sequences
==================================================
"""
sequences_labels = []
for seq in orig_sequences:
    for num in range(1, len(seq)):
        sequences_labels.append(seq[:num+1])

"""
==================================================
5. Padding
==================================================
"""
padded_labels = tf.keras.preprocessing.sequence.pad_sequences(
    sequences_labels,
    padding='pre'
)

padded_labels = np.array(padded_labels)

# Shuffling
np.random.shuffle(padded_labels)

"""
==================================================
6. Splitting
==================================================
"""
padded = padded_labels[:,:-1]
labels = padded_labels[:,-1] - 1

splitLength = int(len(padded)*train_portion)
train_padded = padded[:splitLength]
eval_padded = padded[splitLength:]

train_labels = labels[:splitLength]
eval_labels = labels[splitLength:]

"""
==================================================
7. Load weights
==================================================
"""
weightFile = f"../Datasets/glove.6B/glove.6B.{embedding_dim}d.txt"
word_index = tokenizer.word_index
num_words = len(word_index)

weights_matrix = np.zeros((num_words+1, embedding_dim))
with open(weightFile, 'r') as file:
    for wLine in file:
        elements = wLine.split(' ')
        if word_index.get(elements[0]) is not None:
            weights_matrix[word_index.get(elements[0])] = np.array(elements[1:]).astype(float)

"""
==================================================
9. Callbacks
==================================================
"""
class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.9:
            print('Achieved accuracy')
            self.model.stop_training = True

my_cb = MyCallback()
es_cb = tf.keras.callbacks.EarlyStopping(
    patience=25,
    restore_best_weights=True,
    monitor='val_accuracy'
)

tb_cb = tf.keras.callbacks.TensorBoard(
    log_dir=os.path.join('99_TensorBoard_logs', time.strftime('run_NLP_51_%Y%m%d-%H%M%S'))
)

"""
==================================================
10. Model
==================================================
"""
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(num_words+1, embedding_dim, weights=[weights_matrix], input_shape=[None]),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256)),
    tf.keras.layers.Dense(num_words, activation=tf.nn.softmax)
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.fit(
    x = train_padded,
    y = train_labels,
    validation_data= (eval_padded, eval_labels),
    epochs=100,
    callbacks=[my_cb, es_cb, tb_cb]
)

"""
==================================================
11. Text prediction
==================================================
"""
index_word = {}
for word, index in word_index.items():
    index_word[index-1] = word

seed_text = "I've got a bad feeling about this"
next_words = 100

for iter in range(next_words):
    new_seq = tokenizer.texts_to_sequences([seed_text])[0]
    new_index = np.argmax(model.predict([new_seq]))
    new_word = index_word[new_index]
    seed_text += " " + new_word

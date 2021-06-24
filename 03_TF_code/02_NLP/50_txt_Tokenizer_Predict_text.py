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

trunc_type = 'pre'
pad_type = 'pre' #not needed
embedding_dim = 32
train_portion = 0.8
"""
==================================================
2. Load data
==================================================
"""
filePath = "../Datasets/irish-lyrics-eof.txt"
lines = []
with open(filePath, 'r') as file:
    for line in file:
        lines.append(line.lower().replace("\n", ""))

np.random.shuffle(lines)
"""
==================================================
3. Tokenize
==================================================
"""
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(lines)
orig_sequences = tokenizer.texts_to_sequences(lines)

num_words = len(tokenizer.word_index)

"""
==================================================
4. Build x and y
==================================================
"""
sequences = []
labels = []
for line in orig_sequences:
    for num_word in range(1,len(line)):
        sequences.append(line[:num_word])
        labels.append(line[num_word])

labels = np.array(labels)-1
labels = tf.keras.utils.to_categorical(labels, num_words)
"""
==================================================
4. Padding
==================================================
"""
padded = tf.keras.preprocessing.sequence.pad_sequences(
    sequences,
    truncating=trunc_type
)

"""
==================================================
5. Splitting
==================================================
"""
splitLength = int(len(padded)*train_portion)
train_padded = padded[:splitLength]
eval_padded = padded[splitLength:]
train_labels = labels[:splitLength]
eval_labels = labels[splitLength:]

"""
==================================================
6. Callbacks
==================================================
"""
class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') > 0.9):
            print('Accuracy achieved')
            self.model.stop_training = True

my_cb = MyCallback()
es_cb = tf.keras.callbacks.EarlyStopping(
    patience=15,
    restore_best_weights=True,
    monitor='val_accuracy'
)

tb_cb = tf.keras.callbacks.TensorBoard(
    log_dir=os.path.join('99_TensorBoard_logs', time.strftime("run_NLP_50_%Y%m%d-%H%M%S"))
)

"""
==================================================
7. Model
==================================================
"""
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(num_words+1, embedding_dim, input_shape=[None]),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128)),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(num_words, activation='softmax')
])

model.compile(
    loss = 'categorical_crossentropy',
    metrics=['accuracy'],
    optimizer='adam'
)

model.fit(
    x=train_padded,
    y = train_labels,
    validation_data=(eval_padded, eval_labels),
    epochs=100,
    callbacks=[my_cb, es_cb, tb_cb],
    batch_size=128
)

"""
==================================================
8. Create text
==================================================
"""
inv_index = {}
for word, value in tokenizer.word_index.items():
    inv_index[value-1] = word

seed_text = "I've got a bad feeling about this"
next_words = 100

for rep in range(next_words):
    seq = tokenizer.texts_to_sequences([seed_text])[0]
    #pred = model.predict_classes([seq])
    pred = np.argmax(model.predict([seq]), axis=-1)
    new_word = inv_index[int(pred)]
    seed_text = seed_text + " " + new_word

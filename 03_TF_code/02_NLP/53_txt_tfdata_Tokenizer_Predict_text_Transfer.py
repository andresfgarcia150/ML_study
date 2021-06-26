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
from collections import Counter

num_oov = 1
train_portion = 0.8
window_size = 10
batch_size = 32
embedding_dim = 100
"""
==================================================
2. Load data
==================================================
"""
filePath = "../Datasets/sonnets.txt"
with open(filePath, 'r') as file:
    lines = file.readlines()

splitLength = int(len(lines)*train_portion)
dsTrain = tf.data.Dataset.from_tensor_slices(lines[:splitLength])
dsEval = tf.data.Dataset.from_tensor_slices(lines[splitLength:])
dsData = tf.data.Dataset.from_tensor_slices(lines)

"""
==================================================
3. Tokenize
Obs: It is not a good idea to use the the tokenizer
as it cannot be replied during the cast from text
to sequences. 
==================================================
"""
def preprocesing(dataset: tf.data.Dataset):
    dataset = dataset.map(lambda x: tf.strings.regex_replace(x, b"<br\\s*/?>", b""))
    dataset = dataset.map(lambda x: tf.strings.regex_replace(x, b"[^a-zA-Z' ]", b""))
    dataset = dataset.map(lambda x: tf.strings.split(x))
    return dataset

dsTrain = preprocesing(dsTrain)
dsEval = preprocesing(dsEval)

vocabulary = Counter()
for txt in dsTrain:
    for word in txt:
        vocabulary.update([word.numpy()])
for txt in dsEval:
    for word in txt:
        vocabulary.update([word.numpy()])

word_index = {}
for pos, word_tuple in enumerate(vocabulary.most_common()):
    word, tup = word_tuple
    word_index[word] = pos+1

# Do not use the tokenizer for a pure tf data pipeline (it cannot be called with map)
#tokenizer = tf.keras.preprocessing.text.Tokenizer()
#tokenizer.fit_on_texts(lines)

"""
==================================================
4. Build the look up
==================================================
"""
keys = tf.constant(list(word_index.keys()))
values = tf.constant(np.int64(list(word_index.values())))
initial = tf.lookup.KeyValueTensorInitializer(keys = keys, values = values)
table = tf.lookup.StaticVocabularyTable(initializer=initial, num_oov_buckets=num_oov)

"""
==================================================
5. Preprocessing
==================================================
"""
def processing(dataset: tf.data.Dataset, window_size, batch_size):
    dataset = dataset.map(lambda x: table.lookup(x))
    dataset = dataset.unbatch()
    dataset = dataset.window(window_size+1, shift = 1, drop_remainder=True)
    dataset = dataset.flat_map(lambda ds: ds.batch(window_size+1))
    dataset = dataset.map(lambda x: (x[:-1], x[-1]-1))
    dataset = dataset.shuffle(10000)
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset

dsTrain = processing(dsTrain, window_size, batch_size)
dsEval = processing(dsEval, window_size, batch_size)

labs = []
for elem, item in dsEval.unbatch():
    labs.append(item.numpy())
"""
==================================================
6. Load weights
==================================================
"""
num_words = len(word_index)
weights_matrix = np.zeros((num_words+1+num_oov, embedding_dim))

weightsFile = f'../Datasets/glove.6B/glove.6B.{embedding_dim}d.txt'
with open(weightsFile, 'r') as file:
    for line in file:
        elements = line.split(' ')
        if word_index.get(elements[0]) is not None:
            weights_matrix[word_index.get(elements[0]),:] = np.array(elements[1:]).astype(float)

"""
==================================================
7. Callbacks
==================================================
"""
class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.99:
            print('Accuracy achieved')
            self.model.stop_training = True

my_cb = MyCallback()
es_cb = tf.keras.callbacks.EarlyStopping(
    patience=30,
    restore_best_weights=True,
    monitor='val_accuracy'
)
tb_cb = tf.keras.callbacks.TensorBoard(
    log_dir=os.path.join('99_TensorBoard_logs', time.strftime('run_NLP_53_%Y%m%d_%H%M%S'))
)

"""
==================================================
8. Model
==================================================
"""
model = tf.keras.Sequential([
    tf.keras.layers.Embedding((num_words+1+num_oov), embedding_dim, weights=[weights_matrix], input_shape=[None]),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(1024)),
    tf.keras.layers.Dense(num_words, activation=tf.nn.softmax)
])

model.compile(
    loss = 'sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.fit(
    dsTrain,
    validation_data=dsEval,
    epochs=100,
    callbacks=[my_cb, es_cb, tb_cb]
)

"""
==================================================
9. Predictions
==================================================
"""
index_word = {}
for word, index in word_index.items():
    index_word[index-1] = word

seed_text = "Help me Obi Wan Kenobi, you're my only hope"
next_words = 100
# ToDo Complete
#for iter in range(next_words):

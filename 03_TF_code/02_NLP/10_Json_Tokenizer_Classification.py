"""
==================================================
Andrés Felipe García Albarracín
2021-06-17
==================================================

Loads data from a json file, tokenizes it and
performs multi-class Classification
"""
import numpy as np

"""
==================================================
1. Load libraries
==================================================
"""
import tensorflow as tf
import json
import time, os
import numpy as np

# Parameters
vocab_size = 1000
oov_tok = '<OOV>'
padd_type = 'post'
trunc_type = 'post'
max_length = 20
train_portion = 0.8
embedding_dim = 16
"""
==================================================
2. Download data
==================================================
"""
#!wget --no-check-certificate https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sarcasm.json -O /tmp/sarcasm.json

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

filePath = '../Datasets/sarcasm.json'

with open(filePath,'r') as file:
    fileContent = json.load(file)

texts = []
labels = []
long = 0

for content in fileContent:
    labels.append(int(content['is_sarcastic']))
    sentence = content['headline']
    # In this case, including the stopwords degrades the performance
    words = [word for word in sentence.split(' ')]# if word not in stopwords]
    long = (len(words) if len(words) > long else long)
    sentence = ' '.join(words)
    sentence = sentence.replace("  ", " ")
    texts.append(sentence)

labels = np.array(labels)
"""
==================================================
3. Tokenize data
==================================================
"""
tokenizer = tf.keras.preprocessing.text.Tokenizer(
    num_words=vocab_size,
    oov_token=oov_tok
)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

"""
==================================================
4. Pad sequences
==================================================
"""
padded = tf.keras.preprocessing.sequence.pad_sequences(
    sequences,
    padding=padd_type,
    truncating=trunc_type,
    maxlen=max_length
)

padded = np.array(padded)
"""
==================================================
5. Split into train and validation
==================================================
"""
splitLength = int(len(sequences)*train_portion)

train_padded = padded[:splitLength]
eval_padded = padded[splitLength:]

train_labels = labels[:splitLength]
eval_labels = labels[splitLength:]

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
    log_dir=os.path.join('99_TensorBoard_logs', time.strftime("run_NLP_10_%Y%m%d-%H%M%S"))
)
"""
==================================================
6. Model
==================================================
"""
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_shape=[max_length]),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
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
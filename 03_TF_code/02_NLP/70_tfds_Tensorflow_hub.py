"""
==================================================
Andrés Felipe García Albarracín
2021-07-02
==================================================

Loads data from TFDS, builds a Text Vectorization layer
and a model for classification
"""

"""
==================================================
1. Load libraries
==================================================
"""
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import time, os

batch_size = 64
embedding_dim = 50 # 50 or 128
"""
==================================================
2. Load data
==================================================
"""
dsTrain, dsEval = tfds.load('imdb_reviews', as_supervised=True, split=['train', 'test'], batch_size=batch_size)

"""
==================================================
3. Callbacks 
==================================================
"""
class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.99:
            print('Accuracy achieved')
            self.model.stop_training = True

my_cb = MyCallback()

es_cb = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    restore_best_weights=True,
    patience=25
)

tb_cb = tf.keras.callbacks.TensorBoard(
    log_dir=os.path.join('99_TensorBoard_logs', time.strftime('run_NLP_70_%Y%m%d_%H%M%S'))
)

"""
==================================================
3. Model
==================================================
"""
# This Hub model tokenizes, pads, cast into an embedding and Reduces the sequence embedding
model_url = f"https://tfhub.dev/google/nnlm-en-dim{embedding_dim}/2"
emb_layer = hub.KerasLayer(model_url, trainable=True, input_shape=[], dtype=tf.string)

model = tf.keras.Sequential([
    emb_layer,
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    metrics=['accuracy'],
    optimizer='adam'
)

model.fit(
    dsTrain,
    validation_data=dsEval,
    epochs=100,
    callbacks=[es_cb, my_cb, tb_cb]
)

"""
==================================================
Andrés Felipe García Albarracín
2021-06-17
==================================================

Loads and saves a model
"""


"""
==================================================
1. Load libraries
==================================================
"""
import tensorflow as tf
import tensorflow_datasets as tfds

"""
==================================================
2. Load data
==================================================
"""
ds_train, ds_eval = tfds.load('rock_paper_scissors', split=['train', 'test'], as_supervised=True)

def preprocessing(dsData: tf.data.Dataset):
    dsData = dsData.map(lambda imag, label: (tf.image.resize(imag, [150,150]), label))
    dsData = dsData.map(lambda imag, label: (imag/255.0, label))
    dsData = dsData.batch(10).prefetch(1)
    return dsData

ds_train = preprocessing(dsData=ds_train)
ds_eval = preprocessing(dsData=ds_eval)

"""
==================================================
3. Load model and predict
==================================================
"""
modelFile = './98_Saved_models/210617_RockPaperSissors.h5'
model = tf.keras.models.load_model(modelFile)
model.evaluate(ds_eval)
model.evaluate(ds_train)
modelFile2 = './98_Saved_models/210617_RockPaperSissors_v2.h5'
model.save(modelFile2)
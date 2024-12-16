from .configurations import *

import os
import glob

import tensorflow as tf
import keras
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.random.set_seed(1)


import numpy as  np
import datetime
import pandas as pd
import shutil

import math

keras.saving.get_custom_objects().clear()

@keras.saving.register_keras_serializable()
class LSTM_SA(keras.Model):
    def __init__(self, lstm_units, dense_units, dropout_rate1, dropout_rate2,
                 **kwargs):
        super(LSTM_SA, self).__init__(**kwargs)
        
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout_rate1 = dropout_rate1
        self.dropout_rate2 = dropout_rate2
        
        self.lstm = keras.layers.LSTM(lstm_units)
        self.dropout1 = keras.layers.Dropout(dropout_rate1)
        self.dense = keras.layers.Dense(dense_units)
        self.dropout2 = keras.layers.Dropout(dropout_rate2)
        
        self.classify = keras.layers.Dense(1, activation='sigmoid')
        
    def call(self, inputs, training=False):
        x = self.lstm(inputs)
        x = self.dropout1(x, training=training)
        x = self.dense(x)
        x = self.dropout2(x, training=training)
        x = self.classify(x)
        return x

    def get_config(self):
        config = super(LSTM_SA, self).get_config().copy()
        config.update({
            'lstm_units': self.lstm_units,
            'dense_units': self.dense_units,
            'dropout_rate1': self.dropout_rate1,
            'dropout_rate2': self.dropout_rate2
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

@keras.saving.register_keras_serializable()
def customized_loss(true, pred):
    true = tf.squeeze(true, axis=-1)
    pred = tf.squeeze(pred, axis=-1)
    losss = LOSS_FUNCTION(true, pred)
    # save for debugging
    # np.save("../model/debug/true_pred.npy", np.hstack([true.numpy(), pred.numpy()]))
    print(true, pred)
    return losss

CUSTOM_OBJECTS_TASKED = {
    'LSTMCell': LSTM_SA,
    'customized_loss': customized_loss,
}

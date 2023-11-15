import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer, Dropout

SEQUENCE_LENGTH = 200

class PositionalEncoding(Layer): 
    def __init__(self, num_hiddens, dropout, max_len=SEQUENCE_LENGTH, **kwargs):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.dropout = dropout
        self.max_len = max_len
        self.dropout_layer = Dropout(dropout)
        
        self.P = np.zeros((1, max_len, num_hiddens))
        X = np.arange(max_len, dtype=np.float32).reshape(-1,1)/np.power(10000, np.arange(0, num_hiddens, 2, dtype=np.float32) / num_hiddens)
        self.P[:, :, 0::2] = np.sin(X)
        self.P[:, :, 1::2] = np.cos(X)

    def call(self, X, **kwargs):
        X = X + self.P[:, :X.shape[1], :]
        return self.dropout_layer(X, **kwargs)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_hiddens": self.num_hiddens,
            "dropout": self.dropout,
            "max_len": self.max_len,
            "P": self.P
        })
        return config

gpus = tf.config.experimental.list_physical_devices('GPU')
print("Num of GPUS: ", len(gpus))
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

model = tf.keras.models.load_model("checkpoints\epoch_09_loss0.311.h5", compile=False, custom_objects={"PositionalEncoding": PositionalEncoding})

print(model.summary())


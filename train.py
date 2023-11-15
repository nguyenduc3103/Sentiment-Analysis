import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import string
import re
import pickle
import os
import spacy
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers
from tensorflow.keras.layers import TextVectorization, Dense, Flatten, LSTM, Bidirectional, BatchNormalization, Input, Embedding, Dropout, Reshape, MultiHeadAttention, LayerNormalization, Add, Layer
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import Callback, CSVLogger, ModelCheckpoint
from tensorflow.keras.metrics import CategoricalAccuracy, FalsePositives, FalseNegatives, TruePositives, TrueNegatives, Precision, Recall, AUC

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001
SEED = 0
BUFFER_SIZE = 10000
DROPOUT_PROB = 0.1
HIDDEN_UNITS = 128
VOCAB_SIZE = 30000
SEQUENCE_LENGTH = 200
EMBEDDING_DIM = 50

gpus = tf.config.experimental.list_physical_devices('GPU')
print("Num of GPUS: ", len(gpus))
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

np.random.seed(SEED)
tf.random.set_seed(SEED)

df = pd.read_csv("dataset\\clean.csv", index_col=False)

X = df["review"].ravel()
y = df["sentiment"].ravel()

y = np_utils.to_categorical(y, 2)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, shuffle=True, stratify=y)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=SEED, shuffle=True, stratify=y_test)

X_train = tf.data.Dataset.from_tensor_slices(X_train)
X_test = tf.data.Dataset.from_tensor_slices(X_test)
y_train = tf.data.Dataset.from_tensor_slices(y_train)
y_test = tf.data.Dataset.from_tensor_slices(y_test)
X_val = tf.data.Dataset.from_tensor_slices(X_val)
y_val = tf.data.Dataset.from_tensor_slices(y_val)

train_dataset = tf.data.Dataset.zip((X_train, y_train))
test_dataset = tf.data.Dataset.zip((X_test, y_test))
val_dataset = tf.data.Dataset.zip((X_val, y_val))

with open("dataset\\vectorizer.pickle", "rb") as file:
    data = pickle.load(file)
    
vector_layer = TextVectorization.from_config(data["config"])
vector_layer.set_weights(data["weights"])

def vectorizer(review, label):
    return vector_layer(review), label

train_dataset = (
    train_dataset
    .shuffle(buffer_size = BUFFER_SIZE, reshuffle_each_iteration = True)
    .map(vectorizer, num_parallel_calls = tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

test_dataset = (
    test_dataset
    .shuffle(buffer_size = BUFFER_SIZE, reshuffle_each_iteration = True)
    .map(vectorizer, num_parallel_calls = tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

val_dataset = (
    val_dataset
    .shuffle(buffer_size = BUFFER_SIZE, reshuffle_each_iteration = True)
    .map(vectorizer, num_parallel_calls = tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

vocab = vector_layer.get_vocabulary()
word_index = dict(zip(vocab, range(len(vocab))))

pathGlove = f"dataset\\glove.6B\\glove.6B.{EMBEDDING_DIM}d.txt"

embeddings_index = {}
with open(pathGlove, encoding="utf8") as file:
    for line in file.readlines():
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs

NUMOFTOKENS = len(vocab) + 2
hits = 0
misses = 0

embedding_matrix = np.zeros((NUMOFTOKENS, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
        hits += 1
    else:
        misses += 1
print("Converted %d words (%d misses)" % (hits, misses))

embedding_layer = Embedding(
    NUMOFTOKENS,
    EMBEDDING_DIM,
    embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
    trainable=False,
)

class PositionalEncoding(Layer): 
    def __init__(self, num_hiddens, dropout, max_len=SEQUENCE_LENGTH):
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

def AttentionBlock():
    inp = Input(shape=(SEQUENCE_LENGTH,))
    embedding = embedding_layer(inp)
    position = PositionalEncoding(num_hiddens=EMBEDDING_DIM, dropout=DROPOUT_PROB, max_len=SEQUENCE_LENGTH)(embedding)
    attention = MultiHeadAttention(num_heads=8, key_dim=EMBEDDING_DIM)(position, position, embedding)
    add_1 = Add()([embedding, attention])
    norm_1 = LayerNormalization()(add_1)

    connected = Dense(EMBEDDING_DIM, activation="relu")(norm_1)
    add_2 = Add()([norm_1, connected])
    norm_2 = LayerNormalization()(add_2)
    return Model(inp, norm_2)

attention = AttentionBlock()

model = Sequential([
    Input(shape=(SEQUENCE_LENGTH,)),
    attention,
    Bidirectional(LSTM(HIDDEN_UNITS, return_sequences=True)),
    Bidirectional(LSTM(HIDDEN_UNITS // 2)),
    Dense(HIDDEN_UNITS, activation="relu"),
    Dropout(DROPOUT_PROB),
    Dense(HIDDEN_UNITS // 2, activation="relu"),
    Dropout(DROPOUT_PROB),
    Dense(HIDDEN_UNITS // 4, activation="relu"),
    Dense(2, activation="softmax")
])

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint('checkpoints/epoch_{epoch:02d}_loss{val_loss:.3f}.h5')

csv_callback = CSVLogger('logs.csv', separator=',', append=True)

metrics = [TruePositives(name='tp'),
           FalsePositives(name='fp'), 
           TrueNegatives(name='tn'), 
           FalseNegatives(name='fn'), 
           Precision(name='precision'), 
           Recall(name='recall'),
           AUC(name='auc'),
           CategoricalAccuracy(name="accuracy")]

model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
              metrics=metrics)

history = model.fit(train_dataset,
                    validation_data=val_dataset,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    callbacks=[model_checkpoint_callback,
                               csv_callback]
                   )

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model_loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('model_accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
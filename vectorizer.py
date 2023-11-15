import pandas as pd
import tensorflow as tf
import pickle

from utils import preprocessing
from keras.utils import np_utils
from tensorflow.keras.layers import TextVectorization
from sklearn.model_selection import train_test_split

SEED = 0
MAX_TOKENS = 30000
SEQUENCE_LENGTH = 200

df = pd.read_csv("dataset\\clean.csv", index_col=False)

X = df["review"].ravel()
y = df["sentiment"].ravel()

y = np_utils.to_categorical(y, 2)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, shuffle=True, stratify=y)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=SEED, shuffle=True, stratify=y_test)

X_train = tf.data.Dataset.from_tensor_slices(X_train)
X_val = tf.data.Dataset.from_tensor_slices(X_val)

data_vectorizer = tf.data.Dataset.concatenate(X_train, X_val)

vector_layer = TextVectorization(
    standardize=preprocessing,
    max_tokens=MAX_TOKENS,
    output_mode='int',
    output_sequence_length=SEQUENCE_LENGTH
)

vector_layer.adapt(data_vectorizer)
print(len(vector_layer.get_vocabulary()))

with open("dataset//vectorizer.pickle", "wb") as file:
    data = {
        "config": vector_layer.get_config(),
        "weights": vector_layer.get_weights(),
    }
    pickle.dump(data, file)


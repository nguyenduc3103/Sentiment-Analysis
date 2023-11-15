import pandas as pd
import tensorflow as tf
from utils import lemmatizer_clean

gpus = tf.config.experimental.list_physical_devices('GPU')
print("Num of GPUS: ", len(gpus))
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

df = pd.read_csv("dataset\\IMDB_Dataset.csv", index_col=False)
df = df.replace({"positive": 1, "negative": 0})

lemma_df = lemmatizer_clean(df)

new_df = pd.DataFrame(lemma_df, columns=["review", "sentiment"]) 

new_df.to_csv("dataset//clean.csv", index=False)








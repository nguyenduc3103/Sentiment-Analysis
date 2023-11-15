import string
import re
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import spacy
from nltk.corpus import stopwords, words

stop_words = set(stopwords.words('english'))
stop_words.update(['subject', 'http', "im", "hes", "shes", "theyre"])
stop_words.difference_update(set(["but", "aren'", "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'don', "don't", 'hadn', "hadn't", 'hasn', "hasn't", "haven't", 'isn', "isn't", 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'nor', 'not', 'shan', "shan't", 'shouldn', "shouldn't", 't', 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]))

lemmatizer = spacy.load("en_core_web_sm")

en_vocab = set(words.words())

def preprocessing(sent):
    lower = tf.strings.lower(sent)
    tag_removal = tf.strings.regex_replace(lower,"<[^>]+>", "")
    punctuation_removal = tf.strings.regex_replace(tag_removal, "[%s]"%re.escape(string.punctuation), "")
    digit_removal = tf.strings.regex_replace(punctuation_removal, "[%s]"%re.escape(string.digits), "")
    stopword_removal = tf.strings.regex_replace(digit_removal, r'\b(' + r'|'.join(stop_words) + r')\b\s*',"")
    return stopword_removal

def lemmatizer_clean(df):
    res = []
    reviews = df["review"]
    sentiment = df["sentiment"]
    for i, (r, s) in enumerate(zip(reviews, sentiment)):
        if not isinstance(r, str):
            continue
        doc = lemmatizer(r.lower())
        tokens = [token.lemma_ for token in doc]
        res_token = []
        for token in tokens:
            if token in en_vocab:
                res_token.append(token)
        txt = " ".join(res_token)
        res.append([txt, s])
        print(f"Item: {i + 1}", end="\r")
    return res

def hist_plot(df, name):
    reviews = df["review"]
    len_reviews = []
    for i, review in enumerate(reviews):
        if not isinstance(review, str):
            continue
        preprocessing_review = preprocessing(review)
        lenReview = tf.strings.split(preprocessing_review).shape[0]
        lenReviews.append(len_reviews)
        print(f"Item: {i + 1}", end="\r")

    fig, ax = plt.subplots(1, 1, figsize =(10, 7), tight_layout = True)
    ax.set_xlabel('Length', fontsize=15)
    ax.set_ylabel('Number of Tokens', fontsize=15)
    ax.set_title(f'Distribution of Sentence Lengths in the {name} dataset', fontsize=20)
    ax.hist(lenReviews, bins=range(min(lenReviews), max(lenReviews), 1))
    ax.tick_params(labelsize = 13)
    plt.savefig(f'img//{name}.png')
    plt.show()
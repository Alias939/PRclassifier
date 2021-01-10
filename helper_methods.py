# Following guide https://medium.com/@sarin.samarth07/glove-word-embeddings-with-keras-python-code-52131b0c8b1d
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.layers import Dense, Embedding, Input, Add, Dot, Reshape, Flatten
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.sequence import skipgrams
from tensorflow.python.keras.models import Model, load_model
from sklearn.preprocessing import OneHotEncoder

import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def get_glove_embedding(path):
    """
    Method to get the embeddings of each word using glove
    :param path: Path to glove files
    :return: Embedding for each word
    """
    embeddings_dict = {}
    with open(path + "glove.6B.300d.txt", 'r', encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
    return embeddings_dict


def find_closest_embeddings(embedding):
    """
    Returns words that are closest to embedded word
    :param embedding: Embedding of chosen word
    :return: List of words closest to that embedded word
    """
    return sorted(embeddings_dict.keys(), key=lambda word: spatial.distance.euclidean(embeddings_dict[word], embedding))
# find_closest_embeddings(embeddings_dict["king"])[1:5]


def get_embedding_matrix(series, embeddings_dict):
    """
    Get padded sentence (using unique ids for every word) and embedding matrix for 300 connected words
    :param series: DataFrame Series containing either premise or hypothesis
    :param embeddings_dict: Embedding dict for each word using glove
    :return: padded sentence, embedding matrix and size of vocabulary
    """
    series = series.astype(str)

    token = Tokenizer()
    token.fit_on_texts(series)
    seq = token.texts_to_sequences(series)

    print(series[0])
    print(seq[0])

    print("the max sentence length is:", max([len(x) for x in seq]))
    pad_seq = pad_sequences(seq, maxlen=100)
    vocab_size = len(token.word_index)+1
    print(vocab_size)

    embedding_matrix = np.zeros((vocab_size,300))
    for word, i in token.word_index.items():
        embedding_value = embeddings_dict.get(word)# using glove embedding for each word
        if embedding_value is not None:
            embedding_matrix[i] = embedding_value

    print(pad_seq[4])
    return pad_seq, embedding_matrix, vocab_size


def baseline_sum_sentence_embeddings(pad_seq, embedding_matrix):
    """
    Use baseline embedding of sentence (baseline of the paper)
    Baseline: Sum all embeddings of the words together to get one embedding for the whole sentence.
    :param pad_seq: Padded sequence of either premise or hypothesis
    :param embedding_matrix: Embedded Matrix of every word
    :return: Embedding that is the sum of all embeddings of the words of the sentence
    """
    # Sum embeddings of words together to get embedding of sentence (baseline sentence embedding model)
    sentence_embs = []
    for i in pad_seq:
        sentence_embs.append(sum([embedding_matrix[x] for x in i]))
    sentence_embs = np.array(sentence_embs)
    return sentence_embs


def encode_labels(gold_label):
    """
    One hot encoding of the labels
    :param gold_label: DataFrame Series of gold_labels
    :return: One hot encoding of each label-
    """
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(np.array(gold_label).reshape(-1, 1))

    print(enc.categories_)

    enc_gold_label = enc.transform(np.array(gold_label).reshape(-1, 1)).toarray()

    print(enc_gold_label[2])
    return enc_gold_label

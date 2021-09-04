import numpy as np
import pandas as pd


def normalize(data):
    return data / np.linalg.norm(data, axis=1, keepdims=True)


def get_max_lengths(dataset_path):
    word_length_list = []
    sent_length_list = []

    df = pd.read_pickle(dataset_path)
    trans = df['tran']

    for tran in trans:
        sentences = tran.split('\n')
        sent_length_list.append(len(sentences))
        for sentence in sentences:
            words = sentence.split()
            word_length_list.append(len(words))

    sorted_word_length = sorted(word_length_list)
    sorted_sent_length = sorted(sent_length_list)

    return sorted_word_length, sorted_sent_length

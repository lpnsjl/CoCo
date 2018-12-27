# -*- coding:utf-8 -*-
__author__ = "andrew"
"""
加载标准rnn所需数据集
"""
import nltk
import numpy as np
import csv
import itertools


def loadStandardRNNData(filename, vocabulary_size=8000):
    unknown = "UNKNOWN_TOKEN"
    start = "START_TOKEN"
    end = "END_TOKEN"

    # 从csv评论中读取评论
    with open(filename, "r") as f:
        reader = csv.reader(f, skipinitialspace=True)
        next(reader)
        # 用nltk把文本切割成句子, 并加上开始与结束符
        sentences = itertools.chain(*[nltk.sent_tokenize(x[0].lower()) for x in reader])
        sentences = ["%s %s %s" % (start, x, end) for x in sentences]
    # 对句子分词
    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    # 获取最常用的vocabulary_size的单词
    vocab = word_freq.most_common(vocabulary_size - 1)
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unknown)
    word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])
    for i, sent in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w in word_to_index else unknown for w in sent]

    X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
    y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])
    return X_train, y_train


if __name__ == "__main__":
    X_train, y_train = loadStandardRNNData("../data/reddit-comments-2015-08.csv")
    print(X_train[0])
    print(X_train[1])
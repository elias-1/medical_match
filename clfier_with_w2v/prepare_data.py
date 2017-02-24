#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2016 www.drcubic.com, Inc. All Rights Reserved
#
"""
File: sentiment_data_preprocess.py
Author: shileicao(shileicao@stu.xjtu.edu.cn)
Date: 17-2-4 下午8:51
"""
import json
import os
import sys
from copy import deepcopy

import jieba
import np
import w2v

MAX_SENTENCE_LEN = 50
MAX_WORD_LEN = 6
SPLIT_RATE = 0.8

jieba.load_userdict(os.path.join('../data', 'words.txt'))


def tokenizer(sentence):
    return jieba.lcut(sentence, cut_all=False)


def stat_max_len(data):
    max_word_len = 0
    max_sentence_len = 0
    for key in data:
        for sentence in data[key]:
            temp_max_word_len = max(
                [len(word) for word in tokenizer(sentence)])
            temp_max_sentence_len = len(tokenizer(sentence))
            if max_word_len < temp_max_word_len:
                max_word_len = temp_max_word_len
            if max_sentence_len < temp_max_sentence_len:
                max_sentence_len = temp_max_sentence_len
    print 'max sentence len:%d, max word len:%d' % (max_sentence_len,
                                                    max_word_len)


def data_shuffle(x, y=None):
    # Randomly shuffle data
    x_temp = deepcopy(x)
    if y:
        y_temp = deepcopy(y)
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    if y:
        return x_temp[shuffle_indices], y_temp[shuffle_indices]
    else:
        return x_temp[shuffle_indices]


def build_dataset(data):
    x_train_data = []
    x_test_data = []
    y_train_data = []
    y_test_data = []
    for key in data:
        one_label_data = data[key]
        one_label_data = data_shuffle(one_label_data)
        split_index = int(SPLIT_RATE * len(one_label_data))
        x_train_data.append(one_label_data[:split_index])
        y_train_data = [key] * split_index
        x_test_data.append(one_label_data[split_index:])
        y_test_data = [key] * (len(one_label_data) - split_index)

    x_train_data, y_train_data = data_shuffle(x_train_data, y_train_data)
    x_test_data, y_test_data = data_shuffle(x_test_data, y_test_data)
    return zip(x_train_data, y_train_data), zip(x_test_data, y_test_data)


def generate_net_input(data, out, word_vob, char_vob):
    #vob_size = word_vob.GetTotalWord()
    for x_text, y in data:
        words = tokenizer(x_text)
        nl = len(words)
        wordi = []
        chari = []
        if nl > MAX_SENTENCE_LEN:
            nl = MAX_SENTENCE_LEN
        for ti in range(nl):
            word = words[ti]
            word_idx = word_vob.GetWordIndex(word)
            wordi.append(str(word_idx))
            chars = list(word)
            nc = len(chars)
            if nc > MAX_WORD_LEN:
                lc = chars[nc - 1]
                chars[MAX_WORD_LEN - 1] = lc
                nc = MAX_WORD_LEN
            for i in range(nc):
                char_idx = char_vob.GetWordIndex(word)
                chari.append(str(char_idx))
            for i in range(nc, MAX_WORD_LEN):
                chari.append("0")
        for i in range(nl, MAX_SENTENCE_LEN):
            wordi.append("0")
            for ii in range(MAX_WORD_LEN):
                chari.append('0')
        line = " ".join(wordi)
        line += " "
        line += " ".join(chari)
        line += " "
        input_line = line + y
        out.write("%s\n" % (input_line))


def main(argc, argv):
    if argc < 6:
        print(
            "Usage:%s <data> <word_vob> <char_vob> <train_output> <test_output>"
            % (argv[0]))

    train_output = open(argv[4], "w")
    test_output = open(argv[5], "w")

    word_vob = w2v.Word2vecVocab()
    word_vob.Load(argv[2])
    char_vob = w2v.Word2vecVocab()
    char_vob.Load(argv[3])
    with open(argv[1], 'r') as f:
        data = json.load(f)
        stat_max_len(data)
        train_data, test_data = build_dataset(data)
        generate_net_input(train_data, train_output, word_vob, char_vob)
        generate_net_input(test_data, test_output, word_vob, char_vob)

    train_output.close()
    test_output.close()


if __name__ == "__main__":
    main(len(sys.argv), sys.argv)

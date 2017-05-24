#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 www.drcubic.com, Inc. All Rights Reserved
#
"""
File: build_fasttext_data.py
Author: shileicao(shileicao@stu.xjtu.edu.cn)
Date: 2017/5/24 21:22
"""
import os
import sys


def build_dataset(data_file, max_sentence_len, train_or_test):
    data_fp = open(data_file, 'r')

    fasttext_data_fp = open(train_or_test + '.fasttext', 'w')
    while True:
        sample_x = data_fp.readline().strip().split()
        if not sample_x:
            break
        fasttext_data_fp.write(' '.join(sample_x[:max_sentence_len]) + ' ' +
                               '__label__' + sample_x[max_sentence_len] + '\n')

    data_fp.close()
    fasttext_data_fp.close()


def main(argc, argv):
    if argc < 4:
        print('Usage:%s <train_dir> <test_dir> <max_sentence_len>' % (argv[0]))
        exit(1)

    build_dataset(argv[1], argv[3], 'train')
    build_dataset(argv[2], argv[3], 'test')


if __name__ == '__main__':
    main(len(sys.argv), sys.argv)

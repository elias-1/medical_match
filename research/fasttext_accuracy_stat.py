#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 www.drcubic.com, Inc. All Rights Reserved
#
"""
File: fasttext_accuracy_stat.py
Author: shileicao(shileicao@stu.xjtu.edu.cn)
Date: 2017/5/24 21:56
"""

import sys


def compute_accuracy(label, ground_truth, max_sentence_len):
    label_fp = open(label, 'r')
    ground_truth_fp = open(ground_truth, 'r')

    total = 0
    accuracy = 0
    while True:
        label_x = label_fp.readline().strip().split()
        if not label_x:
            break
        label_x = label_x[max_sentence_len]
        ground_truth_x = ground_truth_fp.readline().strip()

        print(label_x)
        print(ground_truth_x)

        if label_x == ground_truth_x:
            accuracy += 1
        total += 1
    print('Error rate: %f' % (1 - float(accuracy) / total))

    label_fp.close()
    ground_truth_fp.close()


def main(argc, argv):
    if argc < 3:
        print('Usage:%s <label> <ground_truth> <max_sentence_len>' % (argv[0]))
        exit(1)

    compute_accuracy(argv[1], argv[2], int(argv[3]))


if __name__ == '__main__':
    main(len(sys.argv), sys.argv)

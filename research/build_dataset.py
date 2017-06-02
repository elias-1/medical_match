#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 www.drcubic.com, Inc. All Rights Reserved
#
"""
File: build_dataset.py
Author: shileicao(shileicao@stu.xjtu.edu.cn)
Date: 2017/6/2 11:06

python build_dataset.py data/research_data.csv data/train.txt data/test.txt
"""

import csv
import random
import sys

MAX_SENTENCE_LEN = 80
SPLIT_RATE = 0.8
ENTITY_TYPES = ['@d@', '@s@', '@l@', '@o@', '@m@', '@dp@', '@bp@']

RESEARCH_LABEL = {
    '1': '0',
    '10': '1',
    '21': '2',
    '23': '3',
    '24': '4',
    '29': '5',
    '7': '6'
}


def data_shuffle(x, y=None):
    indexes = range(len(x))
    random.shuffle(indexes)
    x_temp = [x[i] for i in indexes]
    if y:
        assert (len(x) == len(y))
        y_temp = [y[i] for i in indexes]
        return x_temp, y_temp
    else:
        return x_temp


def output_data(data, out_name):
    with open(out_name, 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(data)


def build_dataset(data, train_out, test_out):
    class_data = {}
    train_data = []
    test_data = []
    for row in data:
        if row[0] not in RESEARCH_LABEL:
            continue
        row[0] = RESEARCH_LABEL[row[0]]
        if row[0] in class_data:
            class_data[row[0]].append(row)
        else:
            class_data[row[0]] = [row]

    for key in class_data:
        split_index = int(SPLIT_RATE * len(class_data[key]))
        train_data.extend(class_data[key][:split_index])
        test_data.extend(class_data[key][split_index:])

    output_data(data_shuffle(train_data), train_out)
    output_data(data_shuffle(test_data), test_out)


def main(argc, argv):
    if argc < 4:
        print('Usage:%s <data> <train_output> <test_output>' % (argv[0]))
        exit(1)

    with open(argv[1], 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        data = [row for row in csv_reader]
        build_dataset(data, argv[2], argv[3])


if __name__ == '__main__':
    main(len(sys.argv), sys.argv)

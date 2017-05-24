#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 www.drcubic.com, Inc. All Rights Reserved
#
'''
File: prepare_ner_data.py
Author: shileicao(shileicao@stu.xjtu.edu.cn)
Date: 17-3-3 下午9:59
'''

import csv
import os
import random
import sys

SPLIT_RATE = 0.8


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


def words2labels(words, entity_with_types):
    entity_labels = ['0'] * len(words)
    entities = entity_with_types.keys()
    entities = sorted(entities, key=lambda entity: len(entity), reverse=True)

    entity_location = []
    for entity in entities:
        entity_len = len(entity)
        if entity_len == 1:
            for i, word in enumerate(words):
                if word == entity:
                    entity_location.append([entity, (i, i)])
            words = words.replace(entity, '@')
        else:
            for i, word in enumerate(words[:1 - entity_len]):
                if words[i:i + entity_len] == entity:
                    entity_location.append([entity, (i, i + entity_len - 1)])
            words = words.replace(entity, '@' * entity_len)

    for i, entity_loc in enumerate(entity_location):
        entity = entity_location[0]
        loc = entity_loc[1]
        entity_labels[loc[0]] = 'B-' + entity
        j = 1
        while loc[0] + j <= loc[1]:
            entity_labels[loc[0] + j] = 'I' + entity
            j += 1

    sort_index = sorted(
        range(len(entity_location)),
        key=lambda index: entity_location[index][1][0])
    entity_location = [entity_location[index] for index in sort_index]
    return entity_labels, entity_location


def output_data(data, data_dir, train_or_test):
    output_intent = []
    output_utterance = []
    output_slots = []
    for row in data:
        row = [item.decode('utf-8') for item in row if item.strip() != '']
        entity_with_types = {
            entity_with_type.split('/')[0]: entity_with_type.split('/')[1]
            for entity_with_type in row[2:]
        }
        output_intent.append(row[0])
        output_utterance.append(list(row[1]))
        seq_out = words2labels(row[1], entity_with_types)
        output_slots.append(seq_out)

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    with open(os.path.join(data_dir, train_or_test + '.label'), 'w') as f:
        for intent in output_intent:
            f.write(intent + '\n')

    with open(os.path.join(data_dir, train_or_test + '.seq.in'), 'w') as f:
        for utterance in output_utterance:
            f.write(' '.join(utterance) + '\n')

    with open(os.path.join(data_dir, train_or_test + '.seq.out'), 'w') as f:
        for slots in output_slots:
            f.write(' '.join(slots) + '\n')


def build_dataset(data, train_dir, test_dir):
    class_data = {}
    train_data = []
    test_data = []
    for row in data:
        if row[0] in class_data:
            class_data[row[0]].append(row)
        else:
            class_data[row[0]] = [row]
    class_stat = {}
    for key in class_data:
        if len(class_data[key]) < 70:
            continue
        class_stat[key] = len(class_data[key])

    for key in class_data:
        if len(class_data[key]) < 70:
            continue
        split_index = int(SPLIT_RATE * len(class_data[key]))
        train_data.extend(class_data[key][:split_index])
        test_data.extend(class_data[key][split_index:])
    train_data = data_shuffle(train_data)
    test_data = data_shuffle(test_data)
    output_data(train_data, train_dir, 'train')
    output_data(test_data, test_dir, 'test')


def main(argc, argv):
    if argc < 4:
        print('Usage:%s <data> <train_dir> <test_dir>' % (argv[0]))
        exit(1)

    with open(argv[1], 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        data = [row for row in csv_reader]
        build_dataset(data, argv[2], argv[3])


if __name__ == '__main__':
    main(len(sys.argv), sys.argv)

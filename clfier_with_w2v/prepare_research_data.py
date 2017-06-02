#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 www.drcubic.com, Inc. All Rights Reserved
#
"""
File: prepare_research_data.py
Author: shileicao(shileicao@stu.xjtu.edu.cn)
Date: 2017/6/2 11:06

bazel build :prepare_research_data
./bazel-bin/prepare_research_data ../research/data/ chars_vec_100.txt 2 common

"""

import csv
import os
import sys

import w2v

MAX_SENTENCE_LEN = 30
MAX_COMMON_LEN = 5
SPLIT_RATE = 0.8
ENTITY_TYPES = ['@d@', '@s@', '@l@', '@o@', '@m@', '@dp@', '@bp@']


def get_data(data_path):
    with open(data_path, 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        data = [row for row in csv_reader]
    return data


def data_to_ids(input_data, char_vob):
    vob_size = char_vob.GetTotalWord()
    entity_common_ids = {
        entity_type: str(vob_size + i)
        for i, entity_type in enumerate(ENTITY_TYPES)
    }
    data = []
    data_common = []
    intent_labels = []
    entity_labels = []
    for row in input_data:
        row = [item.decode('utf-8') for item in row if item.strip() != '']
        intent_labels.append(row[0])

        tokens = list(row[1])
        sample_x = []
        sample_len = len(tokens)
        for token in tokens:
            idx = char_vob.GetWordIndex(token)
            sample_x.append(str(idx))
        for i in range(sample_len, MAX_SENTENCE_LEN):
            sample_x.append('0')
        data.append(sample_x[:MAX_SENTENCE_LEN])

        sample_common_x = []
        entity_with_types = {
            entity_with_type.split('/')[0]: entity_with_type.split('/')[1]
            for entity_with_type in row[2:]
        }
        entities = entity_with_types.keys()
        entities = sorted(
            entities, key=lambda entity: len(entity), reverse=True)
        entity_location = []
        for entity in entities:
            entity_len = len(entity)
            if entity_len == 1:
                for i, word in enumerate(row[1]):
                    if word == entity:
                        entity_location.append([entity, (i, i)])
            else:
                for i, word in enumerate(row[1][:1 - entity_len]):
                    if row[1][i:i + entity_len] == entity:
                        entity_location.append(
                            [entity, (i, i + entity_len - 1)])
        sort_index = sorted(
            range(len(entity_location)),
            key=lambda index: entity_location[index][1][0])
        entity_location = [entity_location[index] for index in sort_index]
        i = 0
        current = 0
        while i < len(tokens):
            if current < len(entity_location) and i == entity_location[
                    current][1][0]:
                entity_type = entity_with_types[entity_location[current][0]]
                sample_common_x.append(entity_common_ids[entity_type])
                i = entity_location[current][1][1] + 1
                current += 1
            else:
                idx = char_vob.GetWordIndex(tokens[i])
                sample_common_x.append(str(idx))
                i += 1
        for i in range(len(sample_common_x), MAX_SENTENCE_LEN):
            sample_common_x.append('0')
        data_common.append(sample_common_x[:MAX_SENTENCE_LEN])

        entity_x = []
        for entity, _ in entity_location:
            entity_type = entity_with_types[entity]
            entity_x.append(str(ENTITY_TYPES.index(entity_type) + 1))
        for i in range(len(entity_x), MAX_COMMON_LEN):
            entity_x.append('0')
        entity_labels.append(entity_x[:MAX_COMMON_LEN])

    return data, data_common, intent_labels, entity_labels


def prepare_data_for_normal(train, test, data_dir, task_name):
    def make_data_set(data, filename):
        data_set = []
        for sample_x, intent_x in zip(data[0], data[1]):
            data_set.append(sample_x + [
                intent_x,
            ])

        with open(filename, 'w') as f:
            for line in data_set:
                f.write(' '.join(line) + '\n')

    make_data_set(train, os.path.join(data_dir, task_name + '_train.txt'))
    make_data_set(test, os.path.join(data_dir, task_name + '_test.txt'))


def prepare_data_for_dkgam(train, test, data_dir, task_name):
    def make_data_set(data, filename):
        data_set = []
        for sample_x, entity_x, intent_x in zip(data[0], data[1], data[2]):
            data_set.append(sample_x + [
                intent_x,
            ] + entity_x)

        with open(filename, 'w') as f:
            for line in data_set:
                f.write(' '.join(line) + '\n')

    make_data_set(train, os.path.join(data_dir, task_name + '_train.txt'))
    make_data_set(test, os.path.join(data_dir, task_name + '_test.txt'))


def process_data(data_dir, char_vob, output_type, output_name):
    train = get_data(os.path.join(data_dir, 'train.txt'))
    test = get_data(os.path.join(data_dir, 'test.txt'))
    train_data, train_common_data, train_intent_labels, train_entity_labels = data_to_ids(
        train, char_vob)
    test_data, test_common_data, test_intent_labels, test_entity_labels = data_to_ids(
        test, char_vob)

    if output_type == 1:
        train = [train_data, train_intent_labels]
        test = [test_data, test_intent_labels]
        prepare_data_for_normal(train, test, data_dir, output_name)

    elif output_type == 2:
        train = [train_common_data, train_intent_labels]
        test = [test_common_data, test_intent_labels]
        prepare_data_for_normal(train, test, data_dir, output_name)
    elif output_type == 3:
        train = [train_common_data, train_entity_labels, train_intent_labels]
        test = [test_common_data, test_entity_labels, test_intent_labels]
        prepare_data_for_dkgam(train, test, data_dir, output_name)
    else:
        raise ValueError('--output_type must be in [1,2]')


def main(argc, argv):
    if argc < 5:
        print('Usage:%s <data_dir> <char_vob> <output_type> <output_name>' %
              (argv[0]))
        exit(1)

    char_vob = w2v.Word2vecVocab()
    char_vob.Load(argv[2])

    process_data(argv[1], char_vob, int(argv[3]), argv[4])


if __name__ == '__main__':
    main(len(sys.argv), sys.argv)

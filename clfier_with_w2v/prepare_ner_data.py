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
import pprint
import random
import sys

import w2v

MAX_SENTENCE_LEN = 80
ENTITY_TYPES = ['@d@', '@s@', '@l@', '@o@', '@m@', '@dp@', '@bp@']
"""ENTITY_TYPES
len([PAD, O]) + len(ENTITY_TYPES) * len([S B M E])
"""

SPLIT_RATE = 0.8


def stat_max_len(data):
    max_sentence_len = max([len(row[1].strip()) for row in data])
    entity_tags = []
    for row in data:
        for entity_with_type in row[2:]:
            if '@' not in entity_with_type:
                continue
            type = entity_with_type.split('/')[1].strip()
            if type not in entity_tags:
                entity_tags.append(type)
    print 'max sentence len: %d' % max_sentence_len
    print 'entity types: %s' % ' '.join(entity_tags)


def build_dataset(data):
    class_data = {}
    train_data = []
    test_data = []
    for row in data:
        if row[0] in class_data:
            class_data[row[0]].append(row)
        else:
            class_data[row[0]] = [row]
    for key in class_data:
        split_index = int(SPLIT_RATE * len(class_data[key]))
        train_data.extend(class_data[key][:split_index])
        test_data.extend(class_data[key][split_index:])

    return data_shuffle(train_data), data_shuffle(test_data)


def words2labels(words, entity_with_types):
    entity_labels = ['1'] * len(words) + ['0'] * (MAX_SENTENCE_LEN - len(words)
                                                  )
    # 得了手足病，手坏了咋办。,手足病/@d@,手/@bp@  排序后对处理过得实体替换掉可以避免手被替换两次
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

    print entity_location
    for i, entity_loc in enumerate(entity_location):
        print entity_loc[0]
        entity_index = ENTITY_TYPES.index(entity_with_types[entity_loc[0]]
                                          .decode('utf-8'))
        loc = entity_loc[1]

        if loc[0] == loc[1]:
            entity_labels[loc[0]] = str(entity_index * 4 + 2)
        else:
            entity_labels[loc[0]] = str(entity_index * 4 + 3)
            entity_labels[loc[1]] = str(entity_index * 4 + 5)
        if loc[1] - loc[0] > 1:
            for mid in range(loc[0] + 1, loc[1]):
                entity_labels[mid] = str(entity_index * 4 + 4)
    return entity_labels, entity_location


def generate_ner_line(ner_out, char_vob, words, labeli):
    nl = len(words)
    chari = []
    if nl > MAX_SENTENCE_LEN:
        nl = MAX_SENTENCE_LEN
    for ti in range(nl):
        char = words[ti]
        idx = char_vob.GetWordIndex(char)
        chari.append(str(idx))
    for i in range(nl, MAX_SENTENCE_LEN):
        chari.append("0")

    line = " ".join(chari)
    line += " "
    ner_line = line + " ".join(labeli)
    ner_out.write("%s\n" % (ner_line))
    return ner_line


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


def processLine(ner_out, data, char_vob):
    for row in data:
        row = [item.decode('utf-8') for item in row if item.strip() != '']
        entity_with_types = {
            entity_with_type.split('/')[0]: entity_with_type.split('/')[1]
            for entity_with_type in row[2:]
        }
        entity_labels, _ = words2labels(row[1], entity_with_types)
        generate_ner_line(ner_out, char_vob, row[1], entity_labels)


def main(argc, argv):
    if argc < 5:
        print('Usage:%s <data> <char_vob> <ner_train_output> <ner_test_output>'
              % (argv[0]))
        exit(1)

    char_vob = w2v.Word2vecVocab()
    char_vob.Load(argv[2])

    ner_train_out = open(argv[3], 'w')
    ner_test_out = open(argv[4], 'w')

    with open(argv[1], 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        data = [row for row in csv_reader]
        stat_max_len(data)
        train_data, test_data = build_dataset(data)
        processLine(ner_train_out, train_data, char_vob)
        processLine(ner_test_out, test_data, char_vob)

    ner_train_out.close()
    ner_test_out.close()


if __name__ == '__main__':
    main(len(sys.argv), sys.argv)

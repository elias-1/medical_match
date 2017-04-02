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
import pprint
import random
import sys

import jieba
import w2v
from utils import (ENTITY_TYPES, MAX_COMMON_LEN, MAX_SENTENCE_LEN,
                   MAX_SENTENCE_LEN2, MAX_WORD_LEN)

jieba.load_userdict(os.path.join('../data', 'words.txt'))

SPLIT_RATE = 0.8

RESEARCH_LABEL = {
    '1': '0',
    '10': '1',
    '21': '2',
    '23': '3',
    '24': '4',
    '29': '5',
    '7': '6'
}


def tokenizer(sentence):
    return jieba.lcut(sentence, cut_all=False)


def stat_max_len(data):
    max_sentence_len = max([len(row[1].strip()) for row in data])
    max_sentence_len2 = max([len(jieba.lcut(row[1].strip())) for row in data])
    entity_tags = []
    for row in data:
        for entity_with_type in row[2:]:
            if '@' not in entity_with_type:
                continue
            type = entity_with_type.split('/')[1].strip()
            if type not in entity_tags:
                entity_tags.append(type)
    print('max sentence len(char level): %d' % max_sentence_len)
    print('max sentence len(word level): %d' % max_sentence_len2)
    print('entity types: %s' % ' '.join(entity_tags))


def build_dataset(data):
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

    pprint.pprint(class_stat)
    for key in class_data:
        if len(class_data[key]) < 70:
            continue
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

    # print entity_location
    for i, entity_loc in enumerate(entity_location):
        # print entity_loc[0]
        entity_index = ENTITY_TYPES.index(entity_with_types[entity_loc[0]]
                                          .decode('utf-8'))
        loc = entity_loc[1]

        if loc[0] == loc[1]:
            entity_labels[loc[0]] = str(entity_index * 4 + 2)
        else:
            entity_labels[loc[0]] = str(entity_index * 4 + 3)
            entity_labels[loc[1]] = str(entity_index * 4 + 5)
        if loc[1] - loc[0] > 1:
            for mid in xrange(loc[0] + 1, loc[1]):
                entity_labels[mid] = str(entity_index * 4 + 4)

    sort_index = sorted(
        xrange(len(entity_location)),
        key=lambda index: entity_location[index][1][0])
    entity_location = [entity_location[index] for index in sort_index]
    return entity_labels, entity_location


def generate_ner_line(ner_out, char_vob, words, labeli):
    nl = len(words)
    chari = []
    if nl > MAX_SENTENCE_LEN:
        nl = MAX_SENTENCE_LEN
    for ti in xrange(nl):
        char = words[ti]
        idx = char_vob.GetWordIndex(char)
        chari.append(str(idx))
    for i in xrange(nl, MAX_SENTENCE_LEN):
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


def entity_id_to_common(chari, entity_location, aspects_id_in_vob):
    current = 0
    common_chari = []
    for i, word_id in enumerate(chari):
        if current < len(entity_location) and i >= entity_location[current][1][
                0] and i <= entity_location[current][1][1]:
            if i == entity_location[current][0]:
                common_chari.append(aspects_id_in_vob[current])

            if i == entity_location[current][1][1]:
                current += 1
        else:
            common_chari.append(chari[i])

    return common_chari


def generate_clfier_line(clfier_cout,
                         char_vob,
                         words_with_class,
                         entity_location,
                         entity_with_types,
                         for_research=False):
    if for_research:
        label_id = RESEARCH_LABEL[words_with_class[0]]
    else:
        label_id = str(int(words_with_class[0]) - 1)

    chars = words_with_class[1]

    vob_size = char_vob.GetTotalWord()
    chari = []
    nl = len(chars)
    for ti in xrange(nl):
        char = chars[ti]
        idx = char_vob.GetWordIndex(char)
        chari.append(str(idx))

    if entity_location:
        aspects_id_in_vob = [
            str(
                ENTITY_TYPES.index(entity_with_types[chars[loc[1][0]:loc[1][1]
                                                           + 1]]) + vob_size)
            for loc in entity_location
        ]

        chari = entity_id_to_common(chari, entity_location, aspects_id_in_vob)

    nl = len(chari)
    if nl > MAX_SENTENCE_LEN:
        clfier_line = ' '.join(chari[:MAX_SENTENCE_LEN]) + ' ' + label_id
    else:
        for i in xrange(nl, MAX_SENTENCE_LEN):
            chari.append('0')
        clfier_line = ' '.join(chari) + ' ' + label_id

    clfier_cout.write("%s\n" % (clfier_line))
    return clfier_line


def last_index(loc0, loc, current):
    last_i = 0
    for i in xrange(current, len(loc)):
        if loc[i] == loc0:
            last_i = i
    return last_i


def refine_tokenizer2common(words, entity_location):
    loc = []
    i = 0
    for word in words:
        loc.extend([i, i + len(word) - 1])
        i += len(word)
    for _, entity_loc in entity_location:
        loc.extend([entity_loc[0], entity_loc[1]])

    loc.sort()
    current = 0
    for _, entity_loc in entity_location:
        loc_index1 = loc.index(entity_loc[0], current)
        if loc_index1 % 2 == 0:
            loc_index2 = last_index(entity_loc[1], loc, current + 1)
            for i in xrange(loc_index2 - loc_index1 - 1):
                loc.pop(loc_index1 + 1)
            if loc_index2 % 2 == 0:
                loc.insert(loc_index1 + 2, entity_loc[1] + 1)
        else:
            loc_index2 = last_index(entity_loc[1], loc, current + 1)
            for i in xrange(loc_index2 - loc_index1 - 1):
                loc.pop(loc_index1 + 1)
            loc.insert(loc_index1, entity_loc[0] - 1)
            if loc_index2 % 2 == 0:
                loc.insert(loc_index1 + 3, entity_loc[1] + 1)
        current = loc_index1

    chars = ''.join(words)
    result_words = []
    for i in xrange(len(loc) / 2):
        result_words.append(chars[loc[2 * i]:loc[2 * i + 1] + 1])
    common_index = []
    for _, entity_loc in entity_location:
        common_index.append(loc.index(entity_loc[0]) / 2)

    return result_words, common_index


def generate_clfier2_line(clfier_cout2,
                          char_vob,
                          word_vob,
                          words_with_class,
                          entity_location,
                          entity_with_types,
                          for_research=False):

    word_vob_size = word_vob.GetTotalWord()
    if for_research:
        label_id = RESEARCH_LABEL[words_with_class[0]]
    else:
        label_id = str(int(words_with_class[0]) - 1)

    words = tokenizer(words_with_class[1])
    words, common_index = refine_tokenizer2common(words, entity_location)
    nl = len(words)
    if nl > MAX_SENTENCE_LEN2:
        nl = MAX_SENTENCE_LEN2
    wordi = []
    chari = []
    current_index = 0
    for ti in xrange(nl):
        word = words[ti]
        if current_index < len(common_index) and ti == common_index[
                current_index]:
            idx = ENTITY_TYPES.index(entity_with_types[word]) + word_vob_size
            current_index += 1
        else:
            idx = word_vob.GetWordIndex(word)
        wordi.append(str(idx))

        chars = list(word)
        nc = len(chars)
        if nc > MAX_WORD_LEN:
            lc = chars[nc - 1]
            chars[MAX_WORD_LEN - 1] = lc
            nc = MAX_WORD_LEN
        for i in xrange(nc):
            char_idx = char_vob.GetWordIndex(chars[i])
            chari.append(str(char_idx))
        for i in xrange(nc, MAX_WORD_LEN):
            chari.append("0")
    for i in xrange(nl, MAX_SENTENCE_LEN2):
        wordi.append("0")
        for ii in xrange(MAX_WORD_LEN):
            chari.append('0')

    line = " ".join(wordi)
    line += " "
    line += " ".join(chari)
    line += " "
    clfier_line = line + label_id

    clfier_cout2.write("%s\n" % (clfier_line))
    return clfier_line


def generate_research_line(out, char_vob, word_vob, words_with_class):
    label_id = RESEARCH_LABEL[words_with_class[0]]
    words = tokenizer(words_with_class[1])

    nl = len(words)
    if nl > MAX_SENTENCE_LEN2:
        nl = MAX_SENTENCE_LEN2
    wordi = []
    chari = []
    for ti in xrange(nl):
        word = words[ti]
        idx = word_vob.GetWordIndex(word)
        wordi.append(str(idx))
        chars = list(word)
        nc = len(chars)
        if nc > MAX_WORD_LEN:
            lc = chars[nc - 1]
            chars[MAX_WORD_LEN - 1] = lc
            nc = MAX_WORD_LEN
        for i in xrange(nc):
            char_idx = char_vob.GetWordIndex(chars[i])
            chari.append(str(char_idx))
        for i in xrange(nc, MAX_WORD_LEN):
            chari.append("0")
    for i in xrange(nl, MAX_SENTENCE_LEN2):
        wordi.append("0")
        for ii in xrange(MAX_WORD_LEN):
            chari.append('0')

    line = " ".join(wordi)
    line += " "
    line += " ".join(chari)
    line += " "
    clfier_line = line + label_id

    out.write("%s\n" % (clfier_line))
    return clfier_line


def generate_research_attend_line(clfier_cout, char_vob, word_vob,
                                  words_with_class, entity_location,
                                  entity_with_types):
    word_vob_size = word_vob.GetTotalWord()

    label_id = RESEARCH_LABEL[words_with_class[0]]

    words = tokenizer(words_with_class[1])
    words, common_index = refine_tokenizer2common(words, entity_location)
    nl = len(words)
    if nl > MAX_SENTENCE_LEN2:
        nl = MAX_SENTENCE_LEN2
    wordi = []
    chari = []

    common_index_ids = [
        str(
            ENTITY_TYPES.index(entity_with_types[words[eindex]]) +
            word_vob_size) for eindex in common_index
    ]

    for ti in xrange(nl):
        word = words[ti]
        idx = word_vob.GetWordIndex(word)
        wordi.append(str(idx))

        chars = list(word)
        nc = len(chars)
        if nc > MAX_WORD_LEN:
            lc = chars[nc - 1]
            chars[MAX_WORD_LEN - 1] = lc
            nc = MAX_WORD_LEN
        for i in xrange(nc):
            char_idx = char_vob.GetWordIndex(chars[i])
            chari.append(str(char_idx))
        for i in xrange(nc, MAX_WORD_LEN):
            chari.append("0")
    for i in xrange(nl, MAX_SENTENCE_LEN2):
        wordi.append("0")
        for ii in xrange(MAX_WORD_LEN):
            chari.append('0')

    for i in xrange(len(common_index_ids), MAX_COMMON_LEN):
        common_index_ids.append("0")

    line = " ".join(wordi)
    line += " "
    line += " ".join(chari)
    line += " "
    clfier_line = line + label_id + " " + " ".join(
        common_index_ids[:MAX_COMMON_LEN])

    clfier_cout.write("%s\n" % (clfier_line))
    return clfier_line


def generate_research_char_attend_line(clfier_cout, char_vob, words_with_class,
                                       entity_location, entity_with_types):
    char_vob_size = char_vob.GetTotalWord()
    label_id = RESEARCH_LABEL[words_with_class[0]]

    chars = words_with_class[1]
    nl = len(chars)
    if nl > MAX_SENTENCE_LEN:
        nl = MAX_SENTENCE_LEN
    chari = []

    common_index_ids = [
        str(
            ENTITY_TYPES.index(entity_with_types[chars[entity_loc[
                0]:entity_loc[1] + 1]]) + char_vob_size)
        for _, entity_loc in entity_location
    ]

    for ti in xrange(nl):
        char = chars[ti]
        idx = char_vob.GetWordIndex(char)
        chari.append(str(idx))

    for i in xrange(nl, MAX_SENTENCE_LEN):
        chari.append("0")

    for i in xrange(len(common_index_ids), MAX_COMMON_LEN):
        common_index_ids.append("0")

    line = " ".join(chari)
    line += " "
    clfier_line = line + label_id + " " + " ".join(
        common_index_ids[:MAX_COMMON_LEN])

    clfier_cout.write("%s\n" % (clfier_line))
    return clfier_line


def processLine(out, output_type, data, char_vob, word_vob):
    for row in data:
        row = [item.decode('utf-8') for item in row if item.strip() != '']
        entity_with_types = {
            entity_with_type.split('/')[0]: entity_with_type.split('/')[1]
            for entity_with_type in row[2:]
        }
        entity_labels, entity_location = words2labels(row[1],
                                                      entity_with_types)
        if output_type == '1':
            generate_ner_line(out, char_vob, row[1], entity_labels)
        elif output_type == '2':
            generate_clfier_line(out, char_vob, row[:2], entity_location,
                                 entity_with_types)
        elif output_type == '3':
            generate_clfier2_line(out, char_vob, word_vob, row[:2],
                                  entity_location, entity_with_types)
        elif output_type == '4':
            generate_clfier2_line(
                out,
                char_vob,
                word_vob,
                row[:2],
                entity_location,
                entity_with_types,
                for_research=True)
        elif output_type == '5':
            generate_research_line(out, char_vob, word_vob, row[:2])
        elif output_type == '6':
            generate_research_attend_line(out, char_vob, word_vob, row[:2],
                                          entity_location, entity_with_types)
        elif output_type == '7':
            generate_research_char_attend_line(
                out, char_vob, row[:2], entity_location, entity_with_types)

        elif output_type == '8':
            generate_clfier_line(
                out,
                char_vob,
                row[:2],
                entity_location,
                entity_with_types,
                for_research=True)
        else:
            raise ValueError('output type error!')


"""
output_type:
1   ner
2   clfier_common
3   clfier_common2  with char-level support
4   research_common  with char-level support
5   research_clfier just clfier
6   research_clfier_attend
7   research_char_clfier_attend
8   research_char_clfier_common
"""


def main(argc, argv):
    if argc < 6:
        print(
            'Usage:%s <data> <char_vob> <train_output> <test_output> <output_type>'
            % (argv[0]))
        exit(1)

    char_vob = w2v.Word2vecVocab()
    char_vob.Load(argv[2])

    word_vob = w2v.Word2vecVocab()
    word_vob.Load('words_vec_100.txt')

    train_out = open(argv[3], 'w')
    test_out = open(argv[4], 'w')

    with open(argv[1], 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        data = [row for row in csv_reader]
        stat_max_len(data)
        train_data, test_data = build_dataset(data)
        processLine(train_out, argv[5], train_data, char_vob, word_vob)
        processLine(test_out, argv[5], test_data, char_vob, word_vob)

    train_out.close()
    test_out.close()


if __name__ == '__main__':
    main(len(sys.argv), sys.argv)

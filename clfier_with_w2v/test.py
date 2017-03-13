#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 www.drcubic.com, Inc. All Rights Reserved
#
"""
File: test.py.py
Author: shileicao(shileicao@stu.xjtu.edu.cn)
Date: 17-3-13 上午11:21
"""
import jieba


def tokenizer(sentence):
    return jieba.lcut(sentence, cut_all=False)


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


if __name__ == '__main__':
    sentence = u'大二学生上吊身亡，近日一则大二学生上吊身亡的新闻引发了网友关注。'
    words = tokenizer(sentence)
    entity_location = [[u'大二学', (0, 2)], [u'吊身亡', (5, 7)], [u'新闻', (22, 23)],
                       [u'引发了', (24, 26)], [u'关注。', (29, 31)]]
    for entity, entity_loc in entity_location:
        assert (sentence[entity_loc[0]:entity_loc[1] + 1] == entity)
    result_words, common_index = refine_tokenizer2common(words,
                                                         entity_location)
    print '-'.join(result_words)
    print common_index

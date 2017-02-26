#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2016 www.drcubic.com, Inc. All Rights Reserved
#
"""
File: es_input.py
Author: shileicao(shileicao@stu.xjtu.edu.cn)
Date: 17-2-26 下午8:15
"""
import json

from es_match import create_index, hanzi2pinyin


def insert2es(words):
    id = 0
    for word in words:
        doc = {}
        doc['Name'] = word
        doc['Pinyin'] = ' '.join(hanzi2pinyin(word))
        create_index(id, doc)
        id += 1


if __name__ == "__main__":
    words = []
    with open('../data/name-idlist-dict-all.json', 'r') as f:
        data = json.load(f)
        for key in data.keys():
            words.append(key)

    insert2es(words)

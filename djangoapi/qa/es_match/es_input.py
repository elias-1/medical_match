#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 www.drcubic.com, Inc. All Rights Reserved
#
"""
File: es_input.py
Author: shileicao(shileicao@stu.xjtu.edu.cn)
Date: 17-2-26 下午8:15
"""
import json

import pypinyin
from elasticsearch import Elasticsearch

es = Elasticsearch()


def hanzi2pinyin(word):
    return [ch[0] for ch in pypinyin.pinyin(word, style=pypinyin.NORMAL)]


def encode_pinyin(word):
    pinyin_ = []
    pinyin = hanzi2pinyin(word)
    for ch in pinyin:
        pinyin_.append('@'.join(list(ch)))
    return '@@'.join(pinyin_)


def encode_pinyin2(word):
    pinyin = hanzi2pinyin(word)
    return ' '.join(pinyin)


def encode_pinyin3(word):
    pinyin_ = []
    pinyin = hanzi2pinyin(word)
    return ''.join(pinyin)


def create_index(id, doc):
    es.index(index="entity-index", doc_type='search', id=id, body=doc)


def encode_entity_type(id_list):
    entity_types = []
    for entity_id in id_list:
        # print entity_id
        if entity_id[1].isdigit():

            if entity_id[0] not in entity_types:
                entity_types.append(entity_id[0])
        else:
            if entity_id[:2] not in entity_types:
                entity_types.append(entity_id[:2])

    return '@' + '-'.join(sorted(entity_types)) + '@'


def insert2es(entitys_with_types):
    id = 0
    for word, entity_type, indexs in entitys_with_types:
        for entity_id in indexs:
            doc = {}
            doc['Name'] = word
            doc['Pinyin'] = encode_pinyin(word)
            doc['Pinyin2'] = encode_pinyin2(word)
            doc['Pinyin3'] = encode_pinyin3(word)
            doc['Entity_type'] = entity_type
            doc['Entity_id'] = entity_id
            create_index(id, doc)
            id += 1
            print('Insert %d' % id)


if __name__ == "__main__":
    words = []
    entity_types = []
    indexs = []
    with open('../data/qadata/name-idlist-dict-all.json', 'r') as f:
        data = json.load(f)
        for key in data.keys():
            words.append(key)
            entity_types.append(encode_entity_type(data[key]))
            indexs = data[key]

    insert2es(zip(words, entity_types, indexs))

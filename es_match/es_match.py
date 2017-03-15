#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 www.drcubic.com, Inc. All Rights Reserved
#
"""
File: es_match.py
Author: shileicao(shileicao@stu.xjtu.edu.cn)
Date: 17-2-26 上午10:51
"""

# import pprint
import pypinyin
from elasticsearch import Elasticsearch

es = Elasticsearch([{"host": "localhost", "port": 9200}])


def hanzi2pinyin(word):

    return [ch[0] for ch in pypinyin.pinyin(word, style=pypinyin.NORMAL)]


def encode_pinyin(word):
    pinyin_ = []
    pinyin = hanzi2pinyin(word)
    for ch in pinyin:
        pinyin_.append('@'.join(list(ch)))
    return '@@'.join(pinyin_)


def create_index(id, doc):
    es.index(index="entity-index", doc_type='search', id=id, body=doc)


def refresh_index():
    es.indices.refresh(index="entity-index")


def search_index(query_string, return_number=1):
    query_pinyin = encode_pinyin(query_string)
    res1 = es.search(
        index='entity-index',
        doc_type='search',
        body={
            'size': return_number,
            'query': {
                "query_string": {
                    'fields': ['Name'],
                    "query": query_string
                }
            }
        })

    res2 = es.search(
        index='entity-index',
        doc_type='search',
        body={
            'size': return_number,
            'query': {
                "query_string": {
                    'fields': ['Pinyin'],
                    "query": query_pinyin
                }
            }
        })

    answers = res1['hits']['hits'] + res2['hits']['hits']

    answers = sorted(
        answers, cmp=(lambda x, y: 1 if x['_score'] < y['_score'] else -1))
    # pprint.pprint(answers)

    result_names = []
    entity_types = []
    if return_number >= 1:
        for i in xrange(min(return_number, len(answers))):
            result_names.append(answers[i]['_source']['Name'])
            entity_types.append(answers[i]['_source']['Entity_type'])
    else:
        result_names.append(answers[0]['_source']['Name'])
        entity_types.append(answers[0]['_source']['Entity_type'])
    return result_names, entity_types

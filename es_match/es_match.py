#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2016 www.drcubic.com, Inc. All Rights Reserved
#
"""
File: es_match.py
Author: shileicao(shileicao@stu.xjtu.edu.cn)
Date: 17-2-26 上午10:51
"""

import pprint

import pypinyin
from elasticsearch import Elasticsearch

es = Elasticsearch([{"host": "localhost", "port": 9200}])


def hanzi2pinyin(word):
    return [
        pypinyin.pinyin(
            ch, style=pypinyin.NORMAL)[0][0] for ch in list(word)
    ]


def create_index(id, doc):
    es.index(index="entity-index", doc_type='search', id=id, body=doc)


def refresh_index():
    es.indices.refresh(index="entity-index")


def search_index(query_string, return_number=1):
    query_pinyin = ' '.join(hanzi2pinyin(query_string))
    res1 = es.search(
        index='entity-index',
        doc_type='search',
        body={
            'size': 2,
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
            'size': 2,
            'query': {
                "query_string": {
                    'fields': ['Pinyin'],
                    "query": query_pinyin
                }
            }
        })

    # res3 = es.search(index='entity-index', doc_type='search',
    #                  body={'size': 2, 'query': {
    #                     "bool": {"must": {"query_string": {"query": query_string}}}}})
    #
    #
    # res4 = es.search(index='entity-index', doc_type='search',
    #                  body={'size': 2,
    #                        'query': {"match_phrase": {"Firstletterspace": {"query": ' '.join(query_string) + ' '}}}})

    answers = res1['hits']['hits'] + res2['hits']['hits']

    answers = sorted(
        answers, cmp=(lambda x, y: 1 if x['_score'] < y['_score'] else -1))
    pprint.pprint(answers)
    result_names = []
    if return_number > 1:
        for i in xrange(min(result_names, len(answers))):
            result_names.append(answers[i]['_source']['Name'])
    else:
        result_names.append(answers[0]['_source']['Name'])
    return result_names

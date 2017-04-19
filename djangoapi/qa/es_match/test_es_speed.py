#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 www.drcubic.com, Inc. All Rights Reserved
#
"""
File: test_es_speed.py
Author: shileicao(shileicao@stu.xjtu.edu.cn)
Date: 2017/4/19 21:01
"""
import time

import pypinyin
from elasticsearch import Elasticsearch
from fuzzywuzzy.fuzz import ratio

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
    pinyin = hanzi2pinyin(word)
    return ''.join(pinyin)


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

    res3 = es.search(
        index='entity-index',
        doc_type='search',
        body={
            'size': return_number,
            'query': {
                "fuzzy": {
                    'Pinyin3': encode_pinyin3(query_string)
                }
            }
        })

    result_names = []
    entity_type_dict = {}
    max_score = 0
    max_item = ''
    if len(res1['hits']['hits']) > 0:
        fuzz = res1['hits']['hits'][0]['_source']['Name']
        if fuzz == query_string:
            result_names.append(fuzz)
            entity_type_dict[fuzz] = res1['hits']['hits'][0]['_source'][
                'Entity_type']
            max_score = res1['hits']['hits'][0]['_score']
            max_item = fuzz
        else:
            fuzz = res2['hits']['hits'][0]['_source']['Pinyin']
            if fuzz == query_pinyin:
                result_names.append(res2['hits']['hits'][0]['_source']['Name'])
                key = res2['hits']['hits'][0]['_source']['Name']
                entity_type_dict[key] = res2['hits']['hits'][0]['_source'][
                    'Entity_type']
                max_score = res2['hits']['hits'][0]['_score']
                max_item = res2['hits']['hits'][0]['_source']['Name']
                #print '222'

            else:
                if len(res3['hits']['hits']) > 0:
                    word_ratio = ratio(
                        query_string,
                        res3['hits']['hits'][0]['_source']['Name'])
                    if word_ratio > 40:
                        result_names.append(
                            res3['hits']['hits'][0]['_source']['Name'])
                        key = res3['hits']['hits'][0]['_source']['Name']
                        entity_type_dict[key] = res3['hits']['hits'][0][
                            '_source']['Entity_type']
                        max_score = res3['hits']['hits'][0]['_score']
                        max_item = res3['hits']['hits'][0]['_source']['Name']
                        #print '333'

    answers = res1['hits']['hits'] + res2['hits']['hits'] + res3['hits'][
        'hits']
    answers = sorted(
        answers, cmp=(lambda x, y: 1 if x['_score'] < y['_score'] else -1))
    re_num = return_number - len(result_names)
    entity_types = []

    if re_num >= 1:
        for item in answers:
            if item['_source']['Name'] not in result_names:
                result_names.append(item['_source']['Name'])
                entity_types.append(item['_source']['Entity_type'])
                entity_type_dict[item['_source']['Name']] = item['_source'][
                    'Entity_type']
                if item['_score'] > max_score:
                    max_score = item['_score']
                    max_item = item['_source']['Name']
                re_num -= 1
                if re_num == 0:
                    break
    score_dict = {}
    score_dict[u'max_score'] = max_score
    score_dict[u'max_item'] = max_item
    return result_names, score_dict, entity_type_dict


def main():
    i = 0
    start = time.time()
    while (i < 10):
        search_index(u'感冒', 5)
        i += 1
        print i

    end = time.time()
    print('use time: %f' % (end - start))


if __name__ == "__main__":
    main()

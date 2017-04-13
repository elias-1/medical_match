#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 www.drcubic.com, Inc. All Rights Reserved
#
"""
File: entity_refine.py
Author: mengtingzhan(476615360@qq.com), shileicao(shileicao@stu.xjtu.edu.cn)
Date: 2017/3/28 9:42
"""
import json
import time

import es_match
from django.db.models import Q

from ..models import Entity_relation


def entity_refine(entity_result, type_result):
    entities = entity_result
    result_json = {}
    result_list = []
    type_list = []
    entity_dict = {}
    score_list = []
    if len(entities) == 0:
        return result_list, type_list
    if len(entities) == 1:
        entity_dict[entities[0]], _, type_dict = es_match.search_index(
            entities[0], 1)
        result_json[u'entity'] = entity_dict
        result_list.append(entity_dict[entities[0]][0])
        type_list.append(type_dict[entity_dict[entities[0]][0]])
        return result_list, type_list
    all_type_dict = {}
    for entity in entities:
        entity_dict[entity], score_dict, type_dict = es_match.search_index(
            entity, 5)
        score_list.append(score_dict)
        for key in type_dict:
            all_type_dict[key] = type_dict[key]
    result_json[u'entity'] = entity_fuzz(entity_dict, score_list)
    result_dict = {}
    for key in result_json[u'entity']:
        e_name = result_json[u'entity'][key]
        result_list.append(e_name)
        type_list.append(all_type_dict[e_name])
        result_dict[e_name] = all_type_dict[e_name]
    return result_list, type_list


def entity_fuzz(entity_dict, score_list):
    entitys = entity_dict
    exact_list = []
    fuzz_list = []
    refine_result = {}
    for enti in entitys:
        if enti in entitys[enti]:
            exact_list.append(enti)
            refine_result[enti] = enti
        else:
            fuzz_list.append(enti)
    #print len(fuzz_list)
    #如果全部匹配
    if len(fuzz_list) == 0:
        return refine_result
    #只有模糊匹配,取分值最高者加入exact_list
    if len(exact_list) == 0 and len(fuzz_list) != 0:
        #return refine_result
        score_sort = sorted(
            score_list, key=lambda s: s[u'max_score'], reverse=True)
        exact_list.append(score_sort[0][u'max_item'])
    #存在全匹配和模糊匹配
    if len(fuzz_list) > 0 and len(exact_list) > 0:
        fuzz_candidates = search_candidates(exact_list)
        for name in fuzz_list:
            #print 'fu:  ' + name
            flag = 0
            for item in entitys[name]:
                iname = item.encode('utf-8')
                if iname in fuzz_candidates:
                    refine_result[name] = item
                    flag = 1
                    break
            if flag == 0:
                refine_result[name] = entitys[name][0]
        return refine_result

    return refine_result


def search_candidates(exact_list):
    fuzz_candi_set = set([])
    for name in exact_list:
        #print 'exact:  ' + name
        sql_result = Entity_relation.objects.get(
            Q(entity_name1=name) | Q(entity_name2=name))

        for en_result in sql_result:
            if en_result.entity_name1 == name:
                fuzz_candi_set.add(en_result.entity_name2)
            else:
                fuzz_candi_set.add(en_result.entity_name1)
    return fuzz_candi_set


if __name__ == "__main__":
    stime = time.clock()
    result = {}
    result['en'], result['type'] = entity_refine(
        [u'感冒', u'鼻涕多', u'喉咙痒', u'咳嗽'])
    dstr = json.dumps(result, ensure_ascii=False, indent=4)
    dstr = unicode.encode(dstr, 'utf-8')
    with open('qa_result.json', 'wb') as f:
        f.write(dstr)
    etime = time.clock()

    print "read: %f s" % (etime - stime)

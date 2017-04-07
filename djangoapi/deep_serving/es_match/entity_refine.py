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
from StringIO import StringIO

#from deep_serving.es_match import es_match
import es_match
import psycopg2
import pycurl

conn = psycopg2.connect(
    'dbname=kgdata user=dbuser password=112233 host=127.0.0.1')


def ops_api(url):
    storage = StringIO()
    try:
        nurl = url
        c = pycurl.Curl()
        c.setopt(pycurl.URL, nurl)
        c.setopt(pycurl.HTTPHEADER, ['Content-Type: application/json'])
        c.setopt(pycurl.CONNECTTIMEOUT, 3)
        c.setopt(c.WRITEFUNCTION, storage.write)
        c.perform()
        c.close()
    except:
        return 2
    response = storage.getvalue()
    res = json.loads(response)
    return res


def search_sql(sql):
    try:
        cur = conn.cursor()
        cur.execute(sql)
        result_set = cur.fetchall()
        cur.close()
        return result_set
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        return None


def entity_refine(entity_result):
    entities = entity_result
    result_json = {}
    result_list = []
    type_list = []
    entity_dict = {}
    score_list = []
    if len(entities) == 0:
        result_list.append('none_enti')
        type_list.append('none_type')
        return result_list, type_list
    if len(entities) == 1:
        entity_dict[entities[0]], _, type_dict = es_match.search_index(
            entities[0], 1)
        result_json[u'entity'] = entity_dict
        result_list.append(entity_dict[entities[0]][0])
        type_list.append(type_dict.values())
        return result_list, type_list
    all_type_dict = {}
    for entity in entities:
        entity_dict[entity], score_dict, type_dict = es_match.search_index(
            entity, 5)
        score_list.append(score_dict)
        all_type_dict = dict(all_type_dict, **type_dict)
    result_json[u'entity'] = entity_fuzz(entity_dict, score_list)
    #result_list = result_json[u'entity'].values()
    for key in result_json[u'entity']:
        e_name = result_json[u'entity'][key]
        result_list.append(e_name)
        type_list.append(all_type_dict[e_name])
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
        sql = """SELECT DISTINCT entity_name2 
                  FROM entity_relation 
                  where entity_name1 = '%s'
                  union 
                  SELECT DISTINCT entity_name1
                  FROM entity_relation
                  where entity_name2 = '%s'"""
        sql_result = search_sql(sql % (name, name))
        #print len(sql_result)
        for en_result in sql_result:
            fuzz_candi_set.add(en_result[0])
    return fuzz_candi_set


if __name__ == "__main__":
    stime = time.clock()
    result = {}
    result = entity_refine("感冒鼻涕多，喉咙痒总是咳嗽，请问医生需要吃什么药？（女，29岁）")
    dstr = json.dumps(result, ensure_ascii=False, indent=4)
    dstr = unicode.encode(dstr, 'utf-8')
    with open('qa_result.json', 'wb') as f:
        f.write(dstr)
    etime = time.clock()

    print "read: %f s" % (etime - stime)

# encoding:UTF-8

import json
import time
from StringIO import StringIO

import chardet
import psycopg2
import pycurl
from es_match import es_match

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
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    return result_set


def entity_identify(sentence):
    question = sentence
    result_json = {}
    url = '1.85.37.136:9999/qa/strEntity/?q={"q":"' + question + '","num":5}'
    entity_result = ops_api(url)
    if entity_result[u'return'] != 0:
        result_json[u'return'] = 1
        return result_json

    entities = entity_result[u'content'][u'entity']
    entity_dict = {}
    score_list = []
    for enti in entities:
        for key in enti:
            entity_dict[key], score_dict, _ = es_match.search_index(key, 5)
            score_list.append(score_dict)
    if len(entity_dict) > 1:
        result_json[u'entity'] = entity_fuzz(entity_dict, score_list)

    return result_json


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
    print len(fuzz_list)
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
            print 'fu:  ' + name
            flag = 0
            for item in entitys[name]:
                iname = item.encode('utf-8')
                if iname in fuzz_candidates:
                    refine_result[name] = item
                    flag = 1
                    break
            if flag == 0:
                refine_result[name] = entitys[name]
        return refine_result

    return refine_result


def search_candidates(exact_list):
    fuzz_candi_set = set([])
    for name in exact_list:
        print 'exact:  ' + name
        sql1 = """SELECT DISTINCT entity_name2 FROM entity_relation where entity_name1 = \'""" + name + """\' union SELECT DISTINCT entity_name1 FROM entity_relation where entity_name2 = \'""" + name + """\';"""
        sql_result = search_sql(sql1)
        print len(sql_result)
        for en_result in sql_result:
            fuzz_candi_set.add(en_result[0])
    return fuzz_candi_set


if __name__ == "__main__":
    stime = time.clock()
    result = entity_identify("感冒鼻涕多，喉咙痒总是咳嗽，请问医生需要吃什么药？（女，29岁）")
    dstr = json.dumps(result, ensure_ascii=False, indent=4)
    dstr = unicode.encode(dstr, 'utf-8')
    with open('qa_result.json', 'wb') as f:
        f.write(dstr)
    etime = time.clock()

    print "read: %f s" % (etime - stime)

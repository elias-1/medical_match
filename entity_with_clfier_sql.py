# encoding:UTF-8

import json
import time

#import chardet
import jieba
import jieba.posseg
import psycopg2
#from clfier_with_w2v.sentence_clfier import SentenceClfier
from es_match import es_match_server

with open('data/common_words.json', 'rb') as f:
    data = f.read()
common_data = json.loads(data)
common_words = common_data['data']

#sent_ner = Ner()
#sentence_clfier = SentenceClfier()127.0.0.1
conn = psycopg2.connect(
    'dbname=kgdata user=elias password=112233 host=127.0.0.1')

jieba.load_userdict('data/words.txt')
dp_data = [
    "喉", "肋", "心", "脑", "脚", "肝", "肠", "肚", "肩", "骨", "耳", "足", "头", "脸", "鼻",
    "肺", "咽", "眼", "肾", "胃", "胆", "手", "筋", "背", "舌", "牙", "口", "腰", "腹", "胸",
    "脾", "嘴", "腿"
]


def en_candidate(segs, common_words):
    s = []
    for i in segs:
        if i.word not in common_words and not i.flag == u't' and not i.flag == u'm':
            s.append(i.word)
    name_index = 0
    en_sets = set([])
    '''
    如果分词结果有一个字的词，那就与其上一个的词合并
    '''

    for sname in s:
        if len(sname) == 1 and name_index > 0 and sname not in dp_data:
            en_sets.add(s[name_index - 1] + sname)
            if s[name_index - 1] in en_sets:
                en_sets.remove(s[name_index - 1])
        else:
            en_sets.add(sname)
        name_index += 1
    return en_sets


def search_sql(sql):
    try:
        cur = conn.cursor()
        cur.execute(sql)
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)


def entity_identify(sentence):
    question = sentence
    result_json = {}

    #result_json[u'label'] = sentence_clfier(sentence)
    seg = jieba.posseg.cut(question)
    en_candis = en_candidate(seg, common_words)
    #en_candis = sent_ner(sentence)
    fuzzy_entity_result = []
    entity_dict = {}
    for name in en_candis:
        print name.encode('utf-8')
        es_results, _ = es_match_server.search_index(name, 5)
        entity_dict[name] = es_results

    #result_json[u'entity'] = fuzzy_entity_result
    result_json[u'entity'] = entity_fuzz(entity_dict)

    return result_json


def entity_fuzz(entity_dict):
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
    if len(fuzz_list) > 0 and len(exact_list) > 0:
        fuzz_candidates = search_candidates(exact_list)
        for name in fuzz_list:
            for item in entityts[name]:
                if item in fuzz_candidates:
                    refine_result[name] = item
                    break
    return refine_result


def search_candidates(exact_list):
    fuzz_candi_set = set([])
    for name in exact_list:
        #sql2 = "SELECT Distinct entity_name2 FROM entity_relation where entity_name1==\'" + name + "\';"
        sql = "SELECT entity_name1, entity_name2 FROM entity_relation where entity_name2==\'" + name + "\' or entity_name2==\'" + name + "\';"
        sql_result = search_sql(sql)
        print type(sql_result)
        for en_result in sql_result:
            fuzz_candi_set.add(en_result[1])
            fuzz_candi_set.add(en_result[2])
    return fuzz_candi_set


if __name__ == "__main__":
    stime = time.clock()
    result = entity_identify(u'感冒，咳嗽，发骚吃什么？')
    dstr = json.dumps(result, ensure_ascii=False, indent=4)
    dstr = unicode.encode(dstr, 'utf-8')
    with open('qa_result.json', 'wb') as f:
        f.write(dstr)
    entity_fuzz(result)
    etime = time.clock()

    print "read: %f s" % (etime - stime)

# encoding:UTF-8

import codecs
import csv
import json
import pprint
import sys
import time

#import exact_match.mm
#import fuzzy_match.fuzzy_match
#from clfier import sentence_clfier
#from clfier.sentence_clfier import tokenizer
import es_match
import jieba
import jieba.posseg
import pypinyin

with open('../data/merge_split2.json', 'rb') as f:
    data = f.read()
common_data = json.loads(data)
common_words = common_data['data']

jieba.load_userdict('../data/words.txt')
dp_data = ["喉", "肋", "心", "脑", "脚", "肝", "肠", "肚", "肩", "骨", "耳", "足", "头",
           "脸", "鼻", "肺", "咽", "眼", "肾", "胃", "胆", "手", "筋", "背", "舌", "牙",
           "口", "腰", "腹", "胸", "脾", "嘴", "腿"]


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


def entity_identify(sentence):
    question = sentence
    result_json = {}
    #result_json[u'type'] = sentence_clfier.sentence_clfier(sentence)
    seg = jieba.posseg.cut(question)
    en_candis = en_candidate(seg, common_words)
    fuzzy_entity_result = set([])
    entity_dict = {}
    for name in en_candis:
        print name + '----'
        es_results, _ = es_match.search_index(name, 1)
        print len(es_results)
        print type(es_results)
        for es_result in es_results:
            print es_result
            fuzzy_entity_result.add(es_result)

    result_json[u'entity'] = list(fuzzy_entity_result)

    return result_json


if __name__ == "__main__":
    stime = time.clock()
    result = entity_identify(u'感冒，发骚，咳嗽吃什么药？')
    dstr = json.dumps(result, ensure_ascii=False, indent=4)
    dstr = unicode.encode(dstr, 'utf-8')
    with open('qa_result.json', 'wb') as f:
        f.write(dstr)
    etime = time.clock()
    print "read: %f s" % (etime - stime)

# encoding:UTF-8

import json
import time

import chardet
from clfier_with_w2v.ner import Ner
#import jieba
#import jieba.posseg
from clfier_with_w2v.sentence_clfier import SentenceClfier
from es_match import es_match

with open('data/common_words.json', 'rb') as f:
    data = f.read()
common_data = json.loads(data)
common_words = common_data['data']

sent_ner = Ner()
sentence_clfier = SentenceClfier()

#jieba.load_userdict('data/words.txt')
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


def entity_identify(sentence):
    question = sentence
    result_json = {}

    result_json[u'label'] = sentence_clfier(sentence)
    #seg = jieba.posseg.cut(question)
    #en_candis = en_candidate(seg, common_words)
    en_candis = sent_ner(sentence)
    fuzzy_entity_result = []
    entity_dict = {}
    for name in en_candis:
        print name.encode('utf-8')
        es_results, _ = es_match.search_index(name, 30)
        en = {}
        en[name] = es_results
        fuzzy_entity_result.append(en)

    result_json[u'entity'] = list(fuzzy_entity_result)

    return result_json


def sent_entity(sentence, num):
    question = sentence
    result_json = {}
    en_candis = sent_ner(sentence)
    fuzzy_entity_result = []
    entity_dict = {}
    for name in en_candis:
        print name.encode('utf-8')
        es_results, _ = es_match.search_index(name, num)
        en = {}
        en[name] = es_results
        fuzzy_entity_result.append(en)

    result_json[u'entity'] = fuzzy_entity_result

    return result_json


def sent_label(sentence):
    question = sentence
    result_json = {}
    result_json[u'label'] = sentence_clfier(sentence)
    return result_json


if __name__ == "__main__":
    stime = time.clock()
    result = entity_identify(u'咳嗽吃什么？')
    dstr = json.dumps(result, ensure_ascii=False, indent=4)
    dstr = unicode.encode(dstr, 'utf-8')
    with open('qa_result.json', 'wb') as f:
        f.write(dstr)
    etime = time.clock()
    print "read: %f s" % (etime - stime)

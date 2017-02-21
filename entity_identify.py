# encoding:UTF-8

import codecs
import csv
import json
import pprint
import sys
import time

import exact_match.mm
import fuzzy_match.fuzzy_match
import jieba
import jieba.posseg
import pypinyin
from clfier import sentence_clfier
from clfier.sentence_clfier import tokenizer


def decode_entity_type(word):
    return word


def hanzi_list2pinyin(hanzi_list):
    return [
        pypinyin.pinyin(word,
                        style=pypinyin.NORMAL)[0][0] for word in hanzi_list
    ]


def get_word_list(entity_name_file):
    f = open(entity_name_file, 'r')
    json_file = json.load(f)
    hanzi_list_result = []
    pinyin_list_result = []
    entity_with_types = []
    for word in json_file:
        entity_type = encode_entity_type(json_file[word])
        entity_with_types.append(entity_type + word)
        hanzi_list = [entity_type, ]
        pinyin_list = [entity_type, ]
        word_list = list(word)
        hanzi_list.extend(word_list)
        pinyin_list.extend(hanzi_list2pinyin(word_list))
        hanzi_list_result.append(hanzi_list)
        pinyin_list_result.append(pinyin_list)
    f.close()
    return hanzi_list_result, pinyin_list_result, entity_with_types


def get_common_word(filename):
    """
    从文件中获取common word词表
    :param filename: common word文件，一个json文件
    :return: 一个包含common word词表的list，每个词是unicode形式
    """
    json_file = open(filename, 'r')
    common_word_list = json.load(json_file)
    return common_word_list['data']


#输入为id列表，如果id的第二位是数字，那么加入id第一位;反之将前两位加入。返回格式‘@type-...-type@'
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


def exact_entity_extract(entity_info):
    return [i[2] + '/' + i[3] for i in entity_info]


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
        if len(sname) == 1 and name_index > 0:
            en_sets.add(s[name_index - 1] + sname)
            if s[name_index - 1] in en_sets:
                en_sets.remove(s[name_index - 1])
        else:
            en_sets.add(sname)
        name_index += 1
    return en_sets


def entity_identify(sentence):
    #print sentence_clfier.sentence_clfier(sentence)

    entity_name_file = 'data/name-idlist-dict-all.json'
    output_file_name = 'data/entity_identify_80_percentage.csv'
    common_words_file = 'data/merge_split2.json'

    jieba.load_userdict('data/words.txt')
    hanzi_list, pinyin_list, entity_with_types = get_word_list(
        entity_name_file)
    hanzi_bseg = exact_match.mm.BMMSeg()
    hanzi_bseg.add_words(
        zip(hanzi_list, entity_with_types), decode_entity_type)

    pinyin_bseg = exact_match.mm.BMMSeg()
    pinyin_bseg.add_words(
        zip(pinyin_list, entity_with_types), decode_entity_type)

    fuzzy = fuzzy_match.fuzzy_match.FuzzyMatch(threshold=80)
    common_words = get_common_word(common_words_file)
    fuzzy.add_common_words(common_words)

    words_fuzzy = fuzzy.get_word_list(entity_name_file)
    fuzzy.add_words(words_fuzzy)
    del hanzi_list, pinyin_list, entity_with_types, words_fuzzy

    questions = [sentence]
    result_json = {}
    result_json[u'type'] = sentence_clfier.sentence_clfier(sentence)

    for question in questions:

        question_hanzi_list = list(question)
        hanzi_entity_info = hanzi_bseg.entity_identify(question_hanzi_list)
        question_pinyin_list = hanzi_list2pinyin(question_hanzi_list)
        pinyin_entity_info = pinyin_bseg.entity_identify(question_pinyin_list)
        hanzi_entity_result = exact_entity_extract(hanzi_entity_info)
        pinyin_entity_result = exact_entity_extract(pinyin_entity_info)

        seg = jieba.posseg.cut(question)
        en_candis = en_candidate(seg, common_words)

        fuzzy_entity_result = []
        for name in en_candis:
            #print name
            fuzzy_hanzi = fuzzy.entity_identify(name)
            fuzzy_pinyin = fuzzy.pinyin_entity_identify(name)
            if len(fuzzy_hanzi) == 0 or len(fuzzy_pinyin) == 0:
                for item in fuzzy_hanzi:
                    fuzzy_entity_result.append(item)
                for item in fuzzy_pinyin:
                    p_list = []
                    p_list.extend(item['name'])
                    for sn in p_list:
                        if sn in name:
                            fuzzy_entity_result.append(item)
                            break
            else:
                if fuzzy_hanzi[0]['ratio'] < fuzzy_pinyin[0]['ratio']:
                    fuzzy_entity_result.append(fuzzy_pinyin[0])
                else:
                    fuzzy_entity_result.append(fuzzy_hanzi[0])
        final_enti = []
        final_enti_dict = []
        # print 'hanzi'
        for enitem in hanzi_entity_result:
            items = enitem.split('/')
            item = items[0]
            item_dict = {}
            item_dict[u'name'] = items[0]
            item_dict[u'type'] = items[1]
            item_dict[u'ratio'] = 100
            if item not in final_enti and len(item) > 1:
                for en in pinyin_entity_result:
                    pen = en.split('/')
                    if item in pen[0]:
                        item = pen[0]
                final_enti.append(item)
                final_enti_dict.append(item_dict)
                print item
        #print 'pinyin'
        for enitem in pinyin_entity_result:
            items = enitem.split('/')
            item = items[0]
            item_dict = {}
            item_dict[u'name'] = items[0]
            item_dict[u'type'] = items[1]
            item_dict[u'ratio'] = 100
            if item not in final_enti and len(item) > 2:
                final_enti.append(item)
                final_enti_dict.append(item_dict)
                #print item
                #print 'fu_hanzi'
        for enitem in fuzzy_entity_result:
            items = enitem[u'name'].split('/')
            item = items[0]
            item_dict = {}
            item_dict[u'name'] = items[0]
            item_dict[u'type'] = items[1]
            item_dict[u'ratio'] = enitem[u'ratio']
            if item not in final_enti:
                for enname in final_enti:
                    if item in enname:
                        item = enname
                        break
                if item not in final_enti:
                    final_enti.append(item)
                    final_enti_dict.append(item_dict)
                    #print item

        result_json[u'entity'] = final_enti_dict

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

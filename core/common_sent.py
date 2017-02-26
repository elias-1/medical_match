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
#from clfier import sentence_clfier
#from clfier.sentence_clfier import tokenizer
import pypinyin


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


def exact_entity_extract_with_location(entity_info):
    entities = []
    for i in entity_info:
        entity_dict = {}
        entity_dict[u'location'] = []
        for index in range(i[0], i[1] + 1):
            entity_dict[u'location'].append(index)
        entity_dict[u'name'] = i[2]
        entity_dict[u'type'] = i[3]
        entities.append(entity_dict)
    return entities


def en_candidate(sentence, segs, common_words):
    '''
    分词后去掉过滤条件，找到每个词的位置，以词为key，位置为value加入can_dic字典
    '''
    s = []
    can_dic = {}
    for i in segs:
        offset = 0
        if i.word not in common_words and not i.flag == u't' and not i.flag == u'm':
            can_dic[i.word] = []
            start_location = sentence.find(i.word, offset)
            if start_location > -1:
                for li in range(start_location, start_location + len(i.word)):
                    can_dic[i.word].append(li)
            s.append(i.word)
        offset += len(i.word)

    name_index = 0
    en_sets = set([])
    dp_data = ["喉", "肋", "心", "脑", "脚", "肝", "肠", "肚", "肩", "骨", "耳", "足", "头",
               "脸", "鼻", "肺", "咽", "眼", "肾", "胃", "胆", "手", "筋", "背", "舌", "牙",
               "口", "腰", "腹", "胸", "脾", "嘴", "腿"]
    final_enti = []
    '''
    如果分词结果有一个字的词，那就与其上一个的词合并
    '''
    for sname in s:
        final_can_dict = {}

        if len(sname) == 1 and name_index > 0 and sname not in dp_data:
            en_sets.add(s[name_index - 1] + sname)
            if can_dic.has_key(s[name_index - 1]) and can_dic.has_key(sname):
                final_can_dict[u'name'] = s[name_index - 1] + sname
                final_can_dict[u'location'] = []
                if (can_dic[sname][0] + 1) > can_dic[s[name_index - 1]][0]:
                    for i in range(can_dic[s[name_index - 1]][0],
                                   can_dic[sname][0] + 1):
                        final_can_dict[u'location'].append(i)
                    final_enti.append(final_can_dict)
                    #print final_can_dict[u'name']
                    #print final_can_dict[u'location']
            if s[name_index - 1] in en_sets:
                en_sets.remove(s[name_index - 1])
            for item in final_enti:
                if item[u'name'] == s[name_index - 1]:
                    final_enti.remove(item)
        else:
            en_sets.add(sname)
            final_can_dict[u'name'] = sname
            final_can_dict[u'location'] = can_dic[sname]
            final_enti.append(final_can_dict)
            #print final_can_dict[u'name']
            #print final_can_dict[u'location']

        name_index += 1

    return final_enti


def entity_identify():
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

    #questions = [sentence]
    result_json = {}
    sent_json = {}
    #result_json[u'type'] = sentence_clfier.sentence_clfier(sentence)
    with open('data/add_qa_11.json', 'rb') as f:
        data = f.read()
    qa_dict = json.loads(data)
    #sent=u'感冒,发骚了，应该吃什么药？'
    c = 0
    for key in qa_dict:
        sent_json[key] = []
        for sentence in qa_dict[key]:
            c += 1
            print str(c) + sentence
            question_hanzi_list = list(sentence)
            hanzi_entity_info = hanzi_bseg.entity_identify(question_hanzi_list)
            question_pinyin_list = hanzi_list2pinyin(question_hanzi_list)
            pinyin_entity_info = pinyin_bseg.entity_identify(
                question_pinyin_list)
            hanzi_entity_result = exact_entity_extract_with_location(
                hanzi_entity_info)
            pinyin_entity_result = exact_entity_extract_with_location(
                pinyin_entity_info)

            seg = jieba.posseg.cut(sentence)
            en_candis = en_candidate(sentence, seg, common_words)
            fuzzy_entity_result = []
            for en_can in en_candis:
                name = en_can[u'name']
                fuzzy_hanzi = fuzzy.entity_identify(name)
                fuzzy_pinyin = fuzzy.pinyin_entity_identify(name)
                if len(fuzzy_hanzi) == 0 or len(fuzzy_pinyin) == 0:
                    for item in fuzzy_hanzi:
                        #print '1'+name
                        items = item[u'name'].split('/')
                        fuzzy_dict = {}
                        fuzzy_dict[u'name'] = items[0]
                        fuzzy_dict[u'type'] = items[1]
                        fuzzy_dict[u'ratio'] = item[u'ratio']
                        fuzzy_dict[u'location'] = en_can[u'location']
                        fuzzy_entity_result.append(fuzzy_dict)
                    for item in fuzzy_pinyin:
                        #print '2'+name
                        p_list = []
                        p_list.extend(item[u'name'])
                        for sn in p_list:
                            if sn in name:
                                items = item[u'name'].split('/')
                                fuzzy_dict = {}
                                fuzzy_dict[u'name'] = items[0]
                                fuzzy_dict[u'type'] = items[1]
                                fuzzy_dict[u'ratio'] = item[u'ratio']
                                fuzzy_dict[u'location'] = en_can[u'location']
                                fuzzy_entity_result.append(fuzzy_dict)
                                break
                else:
                    if fuzzy_hanzi[0]['ratio'] < fuzzy_pinyin[0]['ratio']:
                        #print '3'+name
                        items = fuzzy_pinyin[0][u'name'].split('/')
                        fuzzy_dict = {}
                        fuzzy_dict[u'name'] = items[0]
                        fuzzy_dict[u'type'] = items[1]
                        fuzzy_dict[u'ratio'] = fuzzy_pinyin[0][u'ratio']
                        fuzzy_dict[u'location'] = en_can[u'location']
                        fuzzy_entity_result.append(fuzzy_dict)
                    else:
                        #print '4'+name
                        items = fuzzy_hanzi[0][u'name'].split('/')
                        fuzzy_dict = {}
                        fuzzy_dict[u'name'] = items[0]
                        fuzzy_dict[u'type'] = items[1]
                        fuzzy_dict[u'ratio'] = fuzzy_hanzi[0][u'ratio']
                        fuzzy_dict[u'location'] = en_can[u'location']
                        fuzzy_entity_result.append(fuzzy_dict)
            final_enti = []
            final_enti_dict = []
            name_set = set([])
            for item in pinyin_entity_result:
                name_set.add(item[u'name'])

            for enitem in hanzi_entity_result:
                item = enitem[u'name']
                item_dict = {}
                item_dict[u'name'] = enitem[u'name']
                item_dict[u'type'] = enitem[u'type']
                item_dict[u'location'] = enitem[u'location']
                item_dict[u'ratio'] = 100
                if item not in final_enti and len(item) > 1:
                    for p_item in pinyin_entity_result:

                        if item in p_item[u'name'] and len(item) < len(p_item[
                                u'name']):
                            item_dict[u'name'] = p_item[u'name']
                            item_dict[u'type'] = p_item[u'type']
                            item_dict[u'location'] = p_item[u'location']
                    final_enti.append(item)
                    final_enti_dict.append(item_dict)
                    #print item_dict[u'name']
            for enitem in pinyin_entity_result:
                item = enitem[u'name']
                item_dict = {}
                item_dict[u'name'] = enitem[u'name']
                item_dict[u'type'] = enitem[u'type']
                item_dict[u'location'] = enitem[u'location']
                item_dict[u'ratio'] = 100
                if item not in final_enti and len(item) > 2:
                    final_enti.append(item)
                    final_enti_dict.append(item_dict)
            for enitem in fuzzy_entity_result:
                item = enitem[u'name']
                if item not in final_enti:
                    for enname in final_enti:
                        if item in enname:
                            item = enname
                            break
                    if item not in final_enti:
                        final_enti.append(item)
                        final_enti_dict.append(enitem)
                    #print item

                    #result_json[u'entity']=final_enti_dict
            entity_list = final_enti_dict
            rep_dict = {}
            location_list = []
            for en_dict in entity_list:
                print en_dict[u'name']
                location = en_dict[u'location']
                location_list.extend(location)
                start_l = location[0]
                rep_dict[start_l] = en_dict[u'type']
            new_sent = u''
            for i in range(0, len(sentence)):
                if i in location_list and rep_dict.has_key(i):
                    new_sent = new_sent + rep_dict[i]
                elif i not in location_list:
                    new_sent = new_sent + sentence[i]
            #return new_sent
            sent_json[key].append(new_sent)
    #return result_json
    return sent_json


def sentence_to_common(sentence, entity_data):
    entity_data = entity_identify(sentence)

    entity_list = entity_data[u'entity']
    rep_dict = {}
    location_list = []
    for en_dict in entity_list:
        location = en_dict[u'location']
        location_list.extend(location)
        start_l = location[0]
        rep_dict[start_l] = en_dict[u'type']
    new_sent = u''
    for i in range(0, len(sentence)):
        if i in location_list and rep_dict.has_key(i):
            new_sent = new_sent + rep_dict[i]
        elif i not in location_list:
            new_sent = new_sent + sentence[i]
    return new_sent


if __name__ == "__main__":
    stime = time.clock()
    result = entity_identify()
    dstr = json.dumps(result, ensure_ascii=False, indent=4)
    dstr = unicode.encode(dstr, 'utf-8')
    with open('qa_add_result.json', 'wb') as f:
        f.write(dstr)
    etime = time.clock()
    print "read: %f s" % (etime - stime)

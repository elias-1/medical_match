# encoding:UTF-8

import codecs
import csv
import json
import pprint
import sys

import exact_match.mm
import fuzzy_match.fuzzy_match
import pypinyin


def questions_from_json(filename):
    """
    从json文件中将所有问题生成一个list
    :param filename:
    :return:
    """
    result = []
    with open(filename) as jsonfile:
        questions_json = json.load(jsonfile)
        for key in questions_json.keys():
            for question in questions_json[key]:
                yield question


def decode_entity_type(word, reverse=False):
    if reverse:
        word = word[::-1]
    return word[0]


def hanzi_list2pinyin(hanzi_list):
    return [pypinyin.pinyin(word, style=pypinyin.NORMAL)[0][0]
            for word in hanzi_list]


def get_word_list(entity_name_file):
    f = open(entity_name_file, 'r')
    json_file = json.load(f)
    hanzi_list_result = []
    pinyin_list_result = []
    for word in json_file:
        entity_type = encode_entity_type(json_file[word])
        hanzi_list = [entity_type, ]
        pinyin_list = [entity_type, ]
        word_list = list(word)
        hanzi_list.extend(word_list)
        pinyin_list.extend(hanzi_list2pinyin(word_list))
        hanzi_list_result.append(hanzi_list)
        pinyin_list_result.append(pinyin_list)
    f.close()
    return hanzi_list_result, pinyin_list_result


def get_common_word(filename):
    """
    从文件中获取common word词表
    :param filename: common word文件，一个json文件
    :return: 一个包含common word词表的list，每个词是unicode形式
    """
    json_file = open(filename, 'r')
    common_word_list = json.load(json_file)
    return common_word_list['data']


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


def entity_extract(entity_info, question_hanzi_list):

    entity_with_type = []
    for loc_with_type in entity_info:
        entity_type = loc_with_type[2]
        entity = question_hanzi_list[loc_with_type[0]:loc_with_type[1] + 1]
        entity_with_type.append(''.join(entity) + '/' + entity_type)
    return entity_with_type


def entity_identify(argc, argv):
    if argc < 4:
        print(
            "Usage:%s <entity_name_file> <output_file_name> <question_file_name>"
            % (argv[0]))
    entity_name_file = argv[1]
    output_file_name = argv[2]
    question_file_name = argv[3]
    common_words_file = argv[4]
    # entity_name_file = 'data/name-idlist-dict-all.json'
    # output_file_name = 'data/entity_identify_80_percentage.csv'
    # question_file_name = 'data/qa3.json'
    # common_words_file='data/merge_split2.json'

    hanzi_list, pinyin_list = get_word_list(entity_name_file)
    hanzi_bseg = exact_match.mm.BMMSeg()
    hanzi_bseg.add_words(hanzi_list, decode_entity_type)

    pinyin_bseg = exact_match.mm.BMMSeg()
    pinyin_bseg.add_words(pinyin_list, decode_entity_type)

    fuzzy = fuzzy_match.fuzzy_match.FuzzyMatch(threshold=80)
    common_words = get_common_word(common_words_file)
    fuzzy.add_common_words(common_words)

    words_fuzzy = fuzzy.get_word_list(entity_name_file)
    fuzzy.add_words(words_fuzzy)

    csvfile = open(output_file_name, 'wb')
    csvfile.write(codecs.BOM_UTF8)
    csvwriter = csv.writer(csvfile)
    # #
    questions = questions_from_json(question_file_name)
    for question in questions:
        one_line = []
        question_hanzi_list = list(question)
        hanzi_entity_info = hanzi_bseg.entity_identify(question_hanzi_list)
        question_pinyin_list = hanzi_list2pinyin(question_hanzi_list)
        pinyin_entity_info = pinyin_bseg.entity_identify(question_pinyin_list)
        hanzi_entity_result = entity_extract(hanzi_entity_info,
                                             question_hanzi_list)
        pinyin_entity_result = entity_extract(pinyin_entity_info,
                                              question_hanzi_list)

        fuzzy_entity_result = fuzzy.entity_identify(question)
        fuzzy_pinyin_entity_result = fuzzy.pinyin_entity_identify(question)

        one_line_with_question = [question]
        oneline = []
        #########################################################################
        print question
        print "hanzi:",
        for i in hanzi_entity_result:
            print i
        print 'pinyin:',
        for i in pinyin_entity_result:
            print i
        print "fuzzy",
        for i in fuzzy_entity_result:
            print i
        print "fuzzy_pinyin",
        for i in fuzzy_pinyin_entity_result:
            print i
        #########################################################################

        one_line.extend(hanzi_entity_result)
        one_line.extend(pinyin_entity_result)
        one_line.extend(fuzzy_entity_result)
        one_line.extend(fuzzy_pinyin_entity_result)
        one_line = set(one_line)
        one_line_with_question.extend(list(one_line))
        for i in one_line_with_question:
            print i,
        print ''
        csvwriter.writerow(one_line_with_question)
    csvfile.close()


if __name__ == "__main__":
    entity_identify(len(sys.argv), sys.argv)

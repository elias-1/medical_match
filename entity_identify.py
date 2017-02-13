# encoding:UTF-8

import codecs
import csv
import json
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
    return hanzi_list_result, pinyin_list_result
    f.close()


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
        print ''.join(entity)
        entity_with_type.append(''.join(entity) + '/' + entity_type)
    return entity_with_type


def entity_identify(argc, argv):
    # if argc < 4:
    #     print(
    #         "Usage:%s <entity_name_file> <output_file_name> <question_file_name>"
    # #         % (argv[0]))
    # entity_name_file = argv[1]
    # output_file_name = argv[2]
    # question_file_name = argv[3]
    entity_name_file = 'data/name-idlist-dict-all.json'
    output_file_name = 'data/entity_identify.csv'
    question_file_name = 'data/qa3.json'

    hanzi_list, pinyin_list = get_word_list(entity_name_file)
    hanzi_bseg = exact_match.mm.BMMSeg()
    hanzi_bseg.add_words(hanzi_list, decode_entity_type)

    pinyin_bseg = exact_match.mm.BMMSeg()
    pinyin_bseg.add_words(pinyin_list, decode_entity_type)

    # fuzzy = fuzzy_match.fuzzy_match.FuzzyMatch()
    # words_fuzzy = fuzzy.get_word_list(entity_name_file)
    # fuzzy.add_words(words_fuzzy)

    csvfile = open(output_file_name, 'wb')
    csvfile.write(codecs.BOM_UTF8)
    csvwriter = csv.writer(csvfile)

    questions = questions_from_json(question_file_name)
    for question in questions:
        one_line = []
        question_hanzi_list = list(question)
        hanzi_entity_info = hanzi_bseg.entity_identify(question_hanzi_list)
        question_pinyin_list = hanzi_list2pinyin(question_hanzi_list)
        # print "question:",question_pinyin_list
        # print question_hanzi_list
        pinyin_entity_info = pinyin_bseg.entity_identify(question_pinyin_list)
        hanzi_entity_result = entity_extract(hanzi_entity_info,
                                             question_hanzi_list)
        pinyin_entity_result = entity_extract(pinyin_entity_info,
                                              question_hanzi_list)

        # fuzzy_entities_result = fuzzy.entity_identify(question)

        # print fuzzy_entities_result
        # print pinyin_entity_result
        # print hanzi_entity_result
        one_line_with_question = [question]
        one_line_with_question.extend(one_line)
        # print one_line_with_question[0]
        csvwriter.writerow(one_line_with_question)
    csvfile.close()


if __name__ == "__main__":
    entity_identify(len(sys.argv), sys.argv)

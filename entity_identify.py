# encoding:UTF-8

import codecs
import csv
import json

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
    return [pypinyin.pinyin(word, style=pypinyin.NORMAL)[0][0] for word in hanzi_list]


def get_word_list(entity_name_file):
    f = open(entity_name_file, 'r')
    json_file = json.load(f)
    for word in json_file:
        entity_type = encode_entity_type(json_file[word])
        hanzi_list = [entity_type,]
        pinyin_list = [entity_type,]
        word_list = list(word)
        hanzi_list.extend(word_list)
        pinyin_list.extend(hanzi_list2pinyin(word_list))

        yield hanzi_list, pinyin_list
    f.close()


def encode_entity_type(id_list):
    entity_types = []
    for entity_id in id_list:
        print entity_id
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
        entity = question_hanzi_list[entity_info[0]:entity_info[1]+1]
        entity_with_type.append(entity + '/' + entity_type)
    return entity_with_type


def entity_identify():
    entity_name_file = 'data/name-idlist-dict-all.json'

    words_exact = get_word_list(entity_name_file)
    hanzi_bseg = exact_match.mm.BMMSeg()
    hanzi_bseg.add_words(words_exact[0], decode_entity_type)


    pinyin_bseg = exact_match.mm.BMMSeg()
    pinyin_bseg.add_words(words_exact[1], decode_entity_type)

    fuzzy = fuzzy_match.fuzzy_match.FuzzyMatch()
    words_fuzzy = fuzzy.get_word_list(entity_name_file)
    fuzzy.add_words(words_fuzzy)

    output_file_name = 'data/entity_identify.csv'
    csvfile = open(output_file_name, 'wb')
    csvfile.write(codecs.BOM_UTF8)
    csvwriter = csv.writer(csvfile)

    question_file_name = 'data/qa3.json'
    questions = questions_from_json(question_file_name)
    for question in questions:
        question_hanzi_list = list(question)
        hanzi_entity_info = hanzi_bseg.entity_identify(question_hanzi_list)
        question_pinyin_list = hanzi_list2pinyin(question_hanzi_list)
        pinyin_entity_info = pinyin_bseg.entity_identify(question_pinyin_list)
        fuzzy_entities = fuzzy.entity_identify(question)

        one_line = []
        hanzi_entity_result = entity_extract(hanzi_entity_info, question_hanzi_list)
        pinyin_entity_result = entity_extract(hanzi_entity_info, question_hanzi_list)

        for entity in fuzzy_entities:
            if entity not in one_line:
                one_line.append(entity)

        one_line_with_question = [question]
        one_line_with_question.extend(one_line)
        print one_line_with_question[0]
        csvwriter.writerow(one_line_with_question)
    csvfile.close()


if __name__ == "__main__":
    entity_identify()



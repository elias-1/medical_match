# encoding:UTF-8

import codecs
import csv
import json

import exact_match.mm
import fuzzy_match.fuzzy_match
import pypinyin

#
# def chinese_word2pinyin(word_list):
#     pinyin_word=[]
#     for word in word_list:
#         yield pypinyin.pinyin(list(word),pypinyin.NORMAL)


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
            result.extend(questions_json[key])
    return result


def decode_entity_type(word, reverse=False):
    if reverse:
        word = word[::-1]
    return word[0]


def get_word_list(entity_name_file, pinyin):
    f = open(entity_name_file, 'r')
    json_file = json.load(f)
    for word in json_file:
        entity_type = encode_entity_type(json_file[word])
        entity_with_type = [entity_type]
        if pinyin:
            entity_with_type.extend(pypinyin.pinyin(word,
                                                    style=pypinyin.NORMAL))
        else:
            entity_with_type.extend(list(word))
        yield entity_with_type, rtt
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


def entity_identify():
    entity_name_file = 'data/name-idlist-dict-all.json'

    words_exact = get_word_list(entity_name_file, pinyin=False)
    exact_bseg = exact_match.mm.BMMSeg()
    exact_bseg.add_words(words_exact, encode_entity_type)

    words_pinyin = get_word_list(entity_name_file, pinyin=True)
    pinyin_bseg = exact_match.mm.BMMSeg()
    pinyin_bseg.add_words(words_pinyin, encode_entity_type)

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
        exact_entities = exact_bseg.entity_identify(question)
        fuzzy_entities = fuzzy.entity_identify(question)
        # pinyin_entities = pinyin.entity_identify(question)
        one_line = []
        for entity in exact_entities:
            if entity not in one_line:
                one_line.append(entity)
        for entity in fuzzy_entities:
            if entity not in one_line:
                one_line.append(entity)
        # for entity in pinyin_entities:
        #     if entity not in one_line:
        #         one_line.append(entity)
        one_line_with_question = [question]
        one_line_with_question.extend(one_line)
        print one_line_with_question[0]
        csvwriter.writerow(one_line_with_question)
    csvfile.close()


if __name__ == "__main__":
    # words = get_word_list('data/name-idlist-dict-all.json', pinyin=False)
    # for word in words:
    #     for character in word:
    #         print character,
    #     print ""
    # entity_identify()

    print

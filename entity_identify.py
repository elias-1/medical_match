# encoding:UTF-8

import codecs
import csv
import json

import exact_match.mm
import fuzzy_match.fuzzy_match
import pinyin_match.mm
import pypinyin


def preprocess_chinese_sentence(sent):
    pass

def chinese_word2pinyin(word_list):
    pass



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


def entity_identify():
    entity_name_file = 'data/name-idlist-dict-all.json'

    bseg = exact_match.mm.BMMSeg()
    words_exact = bseg.get_word_list(entity_name_file)
    bseg.add_words(words_exact)

    fuzzy = fuzzy_match.fuzzy_match.FuzzyMatch()
    words_fuzzy = fuzzy.get_word_list(entity_name_file)
    fuzzy.add_words(words_fuzzy)

    pinyin = pinyin_match.mm.BMMSeg()
    words_pinyin = pinyin.get_word_list(entity_name_file)
    pinyin.add_words(words_pinyin)

    output_file_name = 'data/entity_identify.csv'
    csvfile = open(output_file_name, 'wb')
    csvfile.write(codecs.BOM_UTF8)
    csvwriter = csv.writer(csvfile)

    question_file_name = 'data/qa3.json'
    questions = questions_from_json(question_file_name)
    for question in questions:
        exact_entities = bseg.entity_identify(question)
        fuzzy_entities = fuzzy.entity_identify(question)
        pinyin_entities = pinyin.entity_identify(question)
        one_line = []
        for entity in exact_entities:
            if entity not in one_line:
                one_line.append(entity)
        for entity in fuzzy_entities:
            if entity not in one_line:
                one_line.append(entity)
        for entity in pinyin_entities:
            if entity not in one_line:
                one_line.append(entity)
        one_line_with_question = [question]
        one_line_with_question.extend(one_line)
        print one_line_with_question[0]
        csvwriter.writerow(one_line_with_question)
    csvfile.close()


if __name__ == "__main__":
    entity_identify()

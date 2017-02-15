#!/usr/bin/env python
# coding=utf8

import json

import jieba
import pypinyin
from fuzzywuzzy.fuzz import ratio


class FuzzyMatch(object):
    def __init__(self, threshold=50):
        self._word_list = None
        self._threshold = threshold
        self._common_list = None

    @staticmethod
    def get_word_list(filename):
        """
            extract medical words and its class from a file and return them
        :param filename:a json file that contains words and its class.
            All the file is a json object and it contains objects.
            Each key-value is an item where the key is its name and value is class.
        :return: a list of tuple where each tuple is an item and each tuple contains two elements,the first is name and the
            second is class label
        """
        with open(filename, "r") as eneity_names_file:
            entity_names = json.load(eneity_names_file)
            names_with_type = []
            for entity_name in entity_names:
                name_with_type = {}
                name_with_type['name'] = entity_name
                name_with_type['type'] = entity_names[entity_name]
                name_with_type['pinyin'] = " ".join(
                    a[0]
                    for a in pypinyin.pinyin(entity_name,
                                             style=pypinyin.NORMAL))
                names_with_type.append(name_with_type)
            return names_with_type

    def add_words(self, word_list):
        self._word_list = word_list

    def add_common_words(self, common_word_list):
        self._common_list = common_word_list

    def _most_similar_words(self, key_word, pinyin=False):
        """

        :param key_word: word that need to match,string
        :param _word_list:words to search, a list of lists
        :param limit: If limit==0,than return all words whose ratio is larger than threshold,if limit == 1,than return the
        most similar word. If some words has the same similarity and word number is more than limit, than word at the front
        of _word_list
         is returned. Notice that if no word has similarity higher than threshold,than None will be returned
        :param threshold:if treshold is zero,than the most similar word will be returned, else words whose ratio more than
        threshold will be returned.
        :return: Words in _word_list that is most similar to key_word. More information, see para limit

        """
        result = []
        if pinyin:
            key_word = " ".join(a[0]
                                for a in pypinyin.pinyin(
                                    key_word, style=pypinyin.NORMAL))
            for word in self._word_list:
                word_ratio = ratio(key_word, word['pinyin'])
                if word_ratio > self._threshold:
                    result.append(word['name'] + '/' +
                                  self._encode_entity_type(word['type']))
        else:
            for word in self._word_list:
                word_ratio = ratio(key_word, word['name'])
                if word_ratio > self._threshold:
                    result.append(word['name'] + '/' +
                                  self._encode_entity_type(word['type']))
        return None if len(result) == 0 else result

    def _encode_entity_type(self, id_list):
        entity_types = []
        for entity_id in id_list:
            if entity_id[1].isdigit():
                if entity_id[0] not in entity_types:
                    entity_types.append(entity_id[0])
            else:
                if entity_id[:2] not in entity_types:
                    entity_types.append(entity_id[:2])

        return '@' + '-'.join(sorted(entity_types)) + '@'

    def entity_identify(self, sentence, pinyin=False):
        """
        Find entities in a sentence using fuzzy match
        :param sentence:sentence to identify
        :param threshold:if threshold is zero,than the most similar word will be
            returned, else words whose ratio more than threshold will be returned.
        :param pinyin
        :return:
        """
        sentence_word = jieba.cut(sentence)
        sentence_word = [word for word in sentence_word
                         if word not in self._common_list]
        entities_with_type = []
        for word in sentence_word:
            if len(word) > 1:
                word_matched = self._most_similar_words(word, pinyin=True)
                if word_matched:
                    entities_with_type.extend(word_matched)
        return entities_with_type

    def pinyin_entity_identify(self, sentence):
        return self.entity_identify(sentence, pinyin=True)

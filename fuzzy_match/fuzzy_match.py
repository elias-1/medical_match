#!/usr/bin/env python
# coding=utf8

import json

import jieba
from fuzzywuzzy.fuzz import ratio


class FuzzyMatch(object):
    def __init__(self, word_list=None):
        self.word_list = word_list

    def get_word_list(self, filename):
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
                names_with_type.append(name_with_type)
            return names_with_type

    def add_words(self, word_list):
        self.word_list = word_list

    def _most_similar_words(self, key_word, limit=0, threshold=80):
        """
        :param key_word: word that need to match,string
        :param word_list:words to search, a list of lists
        :param limit: If limit==0,than return all words whose ratio is larger than threshold,if limit == 1,than return the
        most similar word. If some words has the same similarity and word number is more than limit, than word at the front
        of word_list
         is returned. Notice that if no word has similarity higher than threshold,than None will be returned
        :param threshold:if treshold is zero,than the most similar word will be returned, else words whose ratio more than
        threshold will be returned.
        :return: Words in word_list that is most similar to key_word. More information, see para limit

        """
        if limit == 1:
            result = None
            max_ratio = 0
            for word in self.word_list:
                # print word
                word_ratio = ratio(key_word, word['name'])
                if word_ratio > max_ratio:
                    result = word
                    max_ratio = word_ratio
            if max_ratio < threshold:
                return None
            else:
                return [word['name'] + self._encode_entity_type(word['type'])]
        elif limit == 0:
            result = []
            for word in self.word_list:
                # print word
                word_ratio = ratio(key_word, word['name'])
                if word_ratio > threshold:
                    result.append(word['name'] + self._encode_entity_type(word[
                        'type']))
            return None if len(result) == 0 else result

    def _encode_entity_type(id_list):
        entity_types = []
        for entity_id in id_list:
            if entity_id[1].isdigit():
                if entity_id[0] not in entity_types:
                    entity_types.append(entity_id[0])
            else:
                if entity_id[:2] not in entity_types:
                    entity_types.append(entity_id[:2])

        return '@' + '+'.join(sorted(entity_types))

    def entity_identify(self, sentence, threshold=50):
        """
        Find entities in a sentence using fuzzy match
        :param sentence:sentence to identify
        :param threshold:if threshold is zero,than the most similar word will be
            returned, else words whose ratio more than threshold will be returned.
        :return:
        """
        sentence_word = jieba.cut(sentence)
        entities_with_type = []
        for word in sentence_word:
            if len(word) > 1:
                word_matched = self._most_similar_words(word, self.word_list)
                if word_matched:
                    entities_with_type.extend(word_matched)
        return entities_with_type

#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2016 www.drcubic.com, Inc. All Rights Reserved
#
"""
File: sentiment_data_preprocess.py
Author: shileicao(shileicao@stu.xjtu.edu.cn)
Date: 17-2-4 下午8:51
"""
import re
import sys
from copy import deepcopy
from xml.etree import ElementTree

import w2v
from utils import (ASPECT, ASPECT_MAPPING, CHARACTERS, MAX_SENTENCE_LEN,
                   MAX_WORD_LEN, entity_id_to_common, get_chars_index)
"""Sentiment polarity
postive
negative
postive and negative
No comment.
"""

TOKENIZER_RE = re.compile(r"[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w]+",
                          re.UNICODE)
"""ENTITY_LABEL
len([PAD, O]) + len(['RESTAURANT', 'SERVICE', 'FOOD']) * len([S B M E])
"""
"""SENTIMENT_POLARITIES
len(PAD) +  len([postive, negative, postive and negative, No comment]) ^ len(['RESTAURANT', 'SERVICE', 'FOOD'])
"""


def stat_max_len(tree):
    max_word_len = 0
    max_sentence_len = 0
    for node in tree.iter('text'):
        # print TOKENIZER_RE.findall(node.text)
        temp_max_word_len = max(
            [len(word) for word in TOKENIZER_RE.findall(node.text)])
        temp_max_sentence_len = len(TOKENIZER_RE.findall(node.text))
        if max_word_len < temp_max_word_len:
            max_word_len = temp_max_word_len
            # print TOKENIZER_RE.findall(node.text)
        if max_sentence_len < temp_max_sentence_len:
            max_sentence_len = temp_max_sentence_len
            # print TOKENIZER_RE.findall(node.text)
    return max_word_len, max_sentence_len


def stat_characters(tree):
    characters = []
    for node in tree.iter('text'):
        # print TOKENIZER_RE.findall(node.text)
        for word in TOKENIZER_RE.findall(node.text):
            characters.extend(list(word.lower()))
        characters = list(set(characters))
    characters.sort()
    return characters


def stat_sentments(tree):
    aspectes = []
    for node in tree.iter('Opinion'):
        category = node.attrib.get('category')
        if category not in aspectes:
            aspectes.append(category)
    return aspectes


def words2labels(words, entities_with_aspects):
    entity_location = []
    entity_labels = ['1'] * len(words) + ['0'] * (MAX_SENTENCE_LEN - len(words)
                                                  )
    entities = []
    for entity in entities_with_aspects.keys():
        words_in_entity = TOKENIZER_RE.findall(entity)
        entity_len = len(words_in_entity)
        if entity_len == 1:
            for i, word in enumerate(words):
                if word == words_in_entity[0]:
                    entity_location.append((i, i))
                    entities.append(entity)
        else:
            for i, word in enumerate(words[:1 - entity_len]):
                if ''.join(words[i:i + entity_len]) == ''.join(
                        words_in_entity):
                    entity_location.append((i, i + entity_len - 1))
                    entities.append(entity)

    for i, loc in enumerate(entity_location):
        # print entities_with_aspects[entities[i]]
        entity_index = ASPECT.index(entities_with_aspects[entities[i]])
        if loc[0] == loc[1]:
            entity_labels[loc[0]] = str(entity_index * 4 + 2)
        else:
            entity_labels[loc[0]] = str(entity_index * 4 + 3)
            entity_labels[loc[1]] = str(entity_index * 4 + 5)
        if loc[1] - loc[0] > 1:
            for mid in range(loc[0] + 1, loc[1]):
                entity_labels[mid] = str(entity_index * 4 + 4)
    return entity_labels, entity_location


def generate_ner_train_line(ner_out, word_vob, words, labeli):
    nl = len(words)
    wordi = []
    chari = []
    if nl > MAX_SENTENCE_LEN:
        nl = MAX_SENTENCE_LEN
    for ti in range(nl):
        word = words[ti]
        idx = word_vob.GetWordIndex(word)
        wordi.append(str(idx))
        chars = list(word)
        nc = len(chars)
        if nc > MAX_WORD_LEN:
            lc = chars[nc - 1]
            chars[MAX_WORD_LEN - 1] = lc
            nc = MAX_WORD_LEN
        for i in range(nc):
            if chars[i] in CHARACTERS:
                idx = CHARACTERS.index(chars[i]) + 1
            else:
                idx = 1
            chari.append(str(idx))
        for i in range(nc, MAX_WORD_LEN):
            chari.append("0")
    for i in range(nl, MAX_SENTENCE_LEN):
        wordi.append("0")
        for ii in range(MAX_WORD_LEN):
            chari.append('0')
    line = " ".join(wordi)
    line += " "
    line += " ".join(chari)
    line += " "
    ner_line = line + " ".join(labeli)
    ner_out.write("%s\n" % (ner_line))
    return ner_line


def get_polarity_id(polarity):
    if len(polarity) == 2:
        j = 2
    elif 'positive' in polarity:
        j = 0
    else:
        j = 1
    return j


def generate_clfier_train_line(clfier_out, word_vob, words, entity_location,
                               entities_with_aspects, aspects_with_polarity):
    # 'RESTAURANT', 'SERVICE', 'FOOD'   label_id = r_j + s_i * (4 * s_j) + f_i * (4 * 4 * f_j) + 1
    if 'RESTAURANT' in aspects_with_polarity:
        r_j = get_polarity_id(aspects_with_polarity['RESTAURANT'])
    else:
        r_j = 3

    if 'SERVICE' in aspects_with_polarity:
        s_i = 1
        s_j = get_polarity_id(aspects_with_polarity['SERVICE'])
    else:
        s_i = 0
        s_j = 3

    if 'FOOD' in aspects_with_polarity:
        f_i = 1
        f_j = get_polarity_id(aspects_with_polarity['FOOD'])
    else:
        f_i = 0
        f_j = 3

    label_id = str(r_j + s_i * (4 * s_j) + f_i * (4 * 4 * f_j) + 1)

    vob_size = word_vob.GetTotalWord()
    wordi = []
    chari = []
    nl = len(words)
    for ti in range(nl):
        word = words[ti]
        idx = word_vob.GetWordIndex(word)
        wordi.append(str(idx))
        chari = get_chars_index(word, chari)

    if entity_location:
        aspects_id_in_vob = [
            str(
                ASPECT.index(entities_with_aspects[' '.join(words[loc[0]:loc[
                    1] + 1])]) + vob_size) for loc in entity_location
        ]
        aspects_chars = [
            ASPECT[int(id) - vob_size].lower() for id in aspects_id_in_vob
        ]
        aspects_chari = []
        for word in aspects_chars:
            aspects_chari = get_chars_index(word, aspects_chari)

        wordi, chari = entity_id_to_common(wordi, chari, entity_location,
                                           aspects_id_in_vob, aspects_chari)

    nl = len(wordi)
    if nl > MAX_SENTENCE_LEN:
        clfier_line = ' '.join(wordi[:MAX_SENTENCE_LEN]) + ' ' + ' '.join(
            chari) + ' ' + label_id
    else:
        for i in range(nl, MAX_SENTENCE_LEN):
            wordi.append('0')
            for ii in range(MAX_WORD_LEN):
                chari.append('0')
        clfier_line = ' '.join(wordi) + ' ' + ' '.join(chari) + ' ' + label_id

    clfier_out.write("%s\n" % (clfier_line))
    return clfier_line


def processLine(ner_out,
                clfier_out,
                ner_clfier_out,
                sentence_tag,
                word_vob,
                training=True):
    """Process one sentence

    :param sentence_tag:
    :return: 0: is not a valid sentence, 1: is a valid sentence.
    """
    entities_with_aspects = {}
    aspects_with_polarity = {}
    sentence = sentence_tag.find('text').text.lower()
    words = TOKENIZER_RE.findall(sentence)
    for opinion in sentence_tag.iter('Opinion'):
        entity = opinion.attrib.get('target').lower()
        words_in_entity = TOKENIZER_RE.findall(entity)
        entity = ' '.join(words_in_entity)
        aspect = ASPECT_MAPPING[opinion.attrib.get('category').split('#')[0]]
        polarity = opinion.attrib.get('polarity')
        # print 'entity: %s   aspect: %s  polarity: %s' % (entity, aspect, polarity)

        if entity != 'null':
            entities_with_aspects[entity] = aspect

        if aspect in aspects_with_polarity and polarity not in aspects_with_polarity[
                aspect]:
            aspects_with_polarity[aspect].append(polarity)
        elif aspect not in aspects_with_polarity:
            aspects_with_polarity[aspect] = [polarity]

    if not aspects_with_polarity:
        return 0
    entity_location = []
    if entities_with_aspects:
        entity_labels, entity_location = words2labels(words,
                                                      entities_with_aspects)
    else:
        entity_labels = ['1'] * len(words) + ['0'] * (MAX_SENTENCE_LEN -
                                                      len(words))
    ner_line = generate_ner_train_line(ner_out, word_vob, words, entity_labels)
    clfier_line = generate_clfier_train_line(
        clfier_out, word_vob, words, entity_location, entities_with_aspects,
        aspects_with_polarity)
    if training:
        ner_clfier_line = ner_line + ' ' + clfier_line
    else:
        ner_clfier_line = ner_line + ' ' + clfier_line[-1]
    ner_clfier_out.write("%s\n" % (ner_clfier_line))


###########################################################################################
#   if entity != 'null':
#     if entity in entities_with_aspects and aspect not in entities_with_aspects[entity]:
#       entities_with_aspects[entity].append(aspect)
#     elif entity not in entities_with_aspects:
#       entities_with_aspects[entity] = [aspect,]
#
#
#
# for entity in entities_with_aspects.keys():
#   if len(entities_with_aspects[entity]) > 1:
#     print entities_with_aspects

# for aspect in aspects_with_polarity.keys():
#   if len(aspects_with_polarity[aspect]) > 1:
#     print aspects_with_polarity
###########################################################################################


def _str_to_bool(s):
    """Convert string to bool (in argparse context)."""
    if s.lower() not in ['true', 'false']:
        raise ValueError('Argument needs to be a ' 'boolean, got {}'.format(s))
    return {'true': True, 'false': False}[s.lower()]


def main(argc, argv):
    if argc < 7:
        print(
            "Usage:%s <data> <word_vob> <ner_output> <clfier_output> <<ner_clfier_output>> <<training>>"
            % (argv[0]))
    ner_out = open(argv[3], "w")
    clfier_out = open(argv[4], "w")
    ner_clfier_out = open(argv[5], "w")
    training = _str_to_bool(argv[6])
    word_vob = w2v.Word2vecVocab()
    word_vob.Load(argv[2])
    with open(argv[1], 'rt') as f:
        tree = ElementTree.parse(f)
        print stat_sentments(tree)
        print 'max word len/max sentence len: %s' % '/'.join(
            map(str, stat_max_len(tree)))
        characters = stat_characters(tree)
        print characters
        print 'characters num: %d' % len(characters)

    for sentence in tree.findall('.//sentence'):
        processLine(ner_out, clfier_out, ner_clfier_out, sentence, word_vob,
                    training)
    ner_out.close()
    clfier_out.close()
    ner_clfier_out.close()


if __name__ == "__main__":
    main(len(sys.argv), sys.argv)

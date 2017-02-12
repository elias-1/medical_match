# coding: utf-8
#!/usr/bin/env python

########################################################################
#
# Copyright (c) 2016 www.drcubic.com, Inc. All Rights Reserved
#
########################################################################

__all__ = ["FMMSeg", "BMMSeg"]

import json
import sys
from copy import deepcopy

from trie import Trie

reload(sys)
sys.setdefaultencoding('utf8')

# In[ ]:


class FMMSeg(object):
    """A forward maximum matching Chinese word segmentor.
    """

    def __init__(self,
                 wordtrie=None,
                 train=None,
                 value=lambda x, y: 0,
                 reverse=False):
        """Construct a FMM Chinese word segmentor.

        @type train: an iterable of words
        @param train: training set
        @type wordtrie: a trie of words
        @param wordtrie: previously trained trie

        If wordtrie is provided, it's deepcopied as the initial trie,
        otherwise a new blank trie will be constructed.

        If train is provided, it's appended into the trie above.
        """
        if wordtrie:
            self._trie = deepcopy(wordtrie)
        else:
            self._trie = Trie()
        if train:
            self.add_words(train, value=value, reverse=reverse)

    def add_words(self, train, value=lambda x: 0):
        """Add train words into the trie.

        @type train: an iterable of words
        @param train: (possibly) new words
        """
        for word in train:
            self._trie[word[len(value(word)):]] = value(word)

    def entity_identify(self, sent, reverse=False):
        """Replace medical entity in sent with common words.

        @type sent: unicode string
        @param sent: the sentence to be processed.
        @param reverse: for BMMseg
        @return: sentence with common words.
        """
        entities = []
        offset = 0
        idx = self._trie.longest_prefix(sent, offset)
        while offset < len(sent):
            if idx is not None:
                entity_type = self._trie[sent[offset:idx]]
                entity = sent[offset:idx]
                offset = idx
                entity_with_type = entity + entity_type
                entities.append(entity_with_type[0:len(entity_with_type) - 1])
            else:
                offset += 1
            idx = self._trie.longest_prefix(sent, offset)
        return entities


class BMMSeg(FMMSeg):
    """A backward maximum matching Chinese word segmentor.
    """

    def get_word_list(self, entity_name_file):
        words = []
        word = {}
        with open(entity_name_file, 'r') as f:
            name_dict = json.load(f)
            for key in name_dict:
                entity_type = self.encode_entity_type(name_dict[key])
                words.append(entity_type + key)
        return words

    def add_words(self, train):
        train = [i[::-1] for i in train]
        FMMSeg.add_words(self, train, value=self.decode_entity_type)

    def entity_identify(self, sentence):
        """

        :param sentence:
        :param reverse:
        :return:
        """
        sentence = sentence[::-1]
        FMM_entities = FMMSeg.entity_identify(self, sentence)
        entities = []
        for entity in FMM_entities:
            entity = entity[::-1].split("@")

            entities.append(entity[1] + '@' + entity[0][::-1])
        return entities

    def decode_entity_type(self, word, reverse=False):
        if reverse:
            word = word[::-1]
        i = 1
        while (word[i] != '@'):
            i += 1
        return word[:i + 1]

    def encode_entity_type(self, id_list):
        entity_types = []
        for entity_id in id_list:
            if entity_id[1].isdigit():
                if entity_id[0] not in entity_types:
                    entity_types.append(entity_id[0])
            else:
                if entity_id[:2] not in entity_types:
                    entity_types.append(entity_id[:2])
        return '@' + '+'.join(sorted(entity_types)) + '@'

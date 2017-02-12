# coding: utf-8

# In[ ]:

# !/usr/bin/env python

########################################################################
#
# Copyright (c) 2016 www.drcubic.com, Inc. All Rights Reserved
#
########################################################################
# -*- coding: utf-8 -*-
# PyCi
#
# Copyright (c) 2009, The PyCi Project
# Authors: Wu Ke <ngu.kho@gmail.com>
#          Chen Xing <cxcxcxcx@gmail.com>
# URL: <http://code.google.com/p/pyci>
# For license information, see COPYING
"""Forward and Backward Maximum Matching word segmentor. Mainly used
as a baseline segmentor.
"""
__all__ = ["FMMSeg", "BMMSeg"]

import json
import sys
from copy import deepcopy

from trie import Trie, encode_word

reload(sys)
sys.setdefaultencoding('utf8')

"""Note: the sentence and token is a list of single word.
"""

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

    def add_words(self, train, value=lambda x, y: 0, reverse=False):
        """Add train words into the trie.

        @type train: an iterable of words
        @param train: (possibly) new words
        """
        for word in train:
            if reverse:
                self._trie[encode_word(word[:-1])] = value(word, reverse)
            else:
                self._trie[encode_word(word[1:])] = value(word)


    def entity_identify(self, sent):
        """Replace medical entity in sent with common words.

        @type sent: unicode string
        @param sent: the sentence to be processed.
        @param reverse: for BMMseg

        @return: sentence with common words.
        """
        offset = 0
        idx = self._trie.longest_prefix(sent, offset)
        entity_location_with_types = []
        while offset < len(sent):
            if idx is not None:
                word = encode_word(sent[offset:idx])
                entity_type = self._trie[word]
                entity_location_with_types.append([offset, idx-1, entity_type])
                offset = idx
            else:
                offset += 1
            idx = self._trie.longest_prefix(sent, offset)
        return entity_location_with_types

    def get_token_value(self, token):
        """Get the token value from self._trie
        """
        return self.__trie[encode_word(token)]


class BMMSeg(FMMSeg):
    """A backward maximum matching Chinese word segmentor.
    """

    def add_words(self, train, value=lambda x: 0):
        """Add train words into the trie.

        @type train: an iterable of words
        @param train: (possibly) new words
        """
        # just reverse everything
        train = [i[::-1] for i in train]
        FMMSeg.add_words(self, train, value=value, reverse=True)

    def entity_identify(self, sent):
        """Replace medical entity in sent with common words.

        @type sent: unicode string
        @param sent: the sentence to be processed.

        @return: sentence with common words.
        """
        sent = sent[::-1]
        entity_location_with_types = FMMSeg.entity_identify(
            self, sent)
        return entity_location_with_types

    def get_token_value(self, token):
        """Get the token value from self._trie
        """
        token = token[::-1]
        return FMMSeg.get_token_value(token)


# In[ ]:


def decode_entity_type(word, reverse=False):
    if reverse:
        word = word[::-1]
    return word[0]


def encode_entity_type(id_list):
    entity_types = []
    for entity_id in id_list:
        if entity_id[1].isdigit():
            if entity_id[0] not in entity_types:
                entity_types.append(entity_id[0])
        else:
            if entity_id[:2] not in entity_types:
                entity_types.append(entity_id[:2])

    return '@' + '-'.join(sorted(entity_types)) + '@'


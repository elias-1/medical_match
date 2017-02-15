# coding: utf-8

# In[ ]:

# !/usr/bin/env python

########################################################################
#
# Copyright (c) 2016 www.drcubic.com, Inc. All Rights Reserved
#
########################################################################
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
                self._trie[encode_word(word[0][:-1])] = value(word[1])
            else:
                self._trie[encode_word(word[0][1:])] = value(word[1])

    def entity_identify(self, sent, reverse=False):
        """Replace medical entity in sent with common words.

        @type sent: unicode string
        @param sent: the sentence to be processed.
        @param reverse: for BMMseg

        @return: [(loc[0], loc[1], entity, type), ...]
        """
        offset = 0
        idx = self._trie.longest_prefix(sent, offset)
        entity_location_with_types = []
        sent_len = len(sent)
        while offset < len(sent):
            if idx is not None:
                word = encode_word(sent[offset:idx])
                entity_with_type = self._trie[word]
                entity, type = self._get_entity_type(entity_with_type)
                if reverse:
                    entity_location_with_types.append([
                        sent_len - idx, sent_len - 1 - offset, entity, type
                    ])
                else:
                    entity_location_with_types.append([
                        offset, idx - 1, entity, type
                    ])

                offset = idx
            else:
                offset += 1
            idx = self._trie.longest_prefix(sent, offset)
        if reverse:
            entity_location_with_types = entity_location_with_types[::-1]
        return entity_location_with_types

    def get_token_value(self, token):
        """Get the token value from self._trie
        """
        return self.__trie[encode_word(token)]

    def _get_entity_type(entity_with_type):
        i = 1
        while (entity_with_type[i] != '@'):
            i += 1
        return entity_with_type[i + 1:], entity_with_type[:i + 1]


class BMMSeg(FMMSeg):
    """A backward maximum matching Chinese word segmentor.
    """

    def add_words(self, train, value=lambda x: 0):
        """Add train words into the trie.

        @type train: an iterable of words
        @param train: (possibly) new words
        """
        # just reverse everything
        train = [(i[0][::-1], i[1]) for i in train]
        FMMSeg.add_words(self, train, value=value, reverse=True)

    def entity_identify(self, sent):
        """Replace medical entity in sent with common words.

        @type sent: unicode string
        @param sent: the sentence to be processed.

        @return: sentence with common words.
        """
        sent = sent[::-1]
        entity_location_with_types = FMMSeg.entity_identify(
            self, sent, reverse=True)
        return entity_location_with_types

    def get_token_value(self, token):
        """Get the token value from self._trie
        """
        token = token[::-1]
        return FMMSeg.get_token_value(token)

# In[ ]:

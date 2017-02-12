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
import pypinyin

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

    def add_words(self, train, value=lambda x, y: 0, reverse=False):
        """Add train words into the trie.

        @type train: an iterable of words
        @param train: (possibly) new words
        """
        for word in train:
            self._trie[word['pinyin']] = value(word)

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
                entity = self._trie[sent[offset:idx]]
                entity_name = entity['key']
                entity_type = entity['type']
                offset = idx
                entity_with_type = entity_name + entity_type[0]
                entities.append(entity_with_type)
            else:
                offset += 1
            idx = self._trie.longest_prefix(sent, offset)
        return entities


class BMMSeg(FMMSeg):
    """A backward maximum matching Chinese word segmentor.
    """

    def _decode_entity_type(self, word, reverse=False):

        return word

    def _encode_entity_type(self, id_list):
        entity_types = []
        for entity_id in id_list:
            if entity_id[1].isdigit():
                if entity_id[0] not in entity_types:
                    entity_types.append(entity_id[0])
            else:
                if entity_id[:2] not in entity_types:
                    entity_types.append(entity_id[:2])

        return '@' + '+'.join(sorted(entity_types))

    def get_word_list(self, entity_name_file):
        words = []
        with open(entity_name_file, 'r') as f:
            name_dict = json.load(f)
            for key in name_dict:
                # print key
                key_with_type_and_pinyin = {}
                key_with_type_and_pinyin['key'] = key
                entity_type = [self._encode_entity_type(name_dict[key])]
                key_with_type_and_pinyin['type'] = entity_type
                list_keys = ""
                pinyinkey = pypinyin.pinyin(key, style=pypinyin.NORMAL)
                for key in pinyinkey:
                    list_keys += key[0][::]
                key_with_type_and_pinyin['pinyin'] = list_keys
                words.append(key_with_type_and_pinyin)
        return words

    def add_words(self, train):
        reversed_train = []
        for one_train in train:
            reversed_one_train = one_train
            reversed_one_train['pinyin'] = one_train['pinyin'][::-1]
            reversed_train.append(reversed_one_train)
        FMMSeg.add_words(self,
                         reversed_train,
                         value=self._decode_entity_type,
                         reverse=True)

    def entity_identify(self, question):
        """

        :param question:
        :return:
        """
        sentence = pypinyin.pinyin(question, style=pypinyin.NORMAL)
        str_sentence = ""
        for word in sentence:
            str_sentence += word[0]
        str_sentence = str_sentence[::-1]
        entities = FMMSeg.entity_identify(self, str_sentence)
        return entities

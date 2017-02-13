# -*- coding: utf-8 -*-

# PyCi
#
# Copyright (c) 2009, The PyCi Project
# Authors: Wu Ke <ngu.kho@gmail.com>
#          Chen Xing <cxcxcxcx@gmail.com>
# URL: <http://code.google.com/p/pyci>
# For license information, see COPYING
"""Trie data structure for fast dictionary look-up
"""


class Trie(object):
    """Trie is a data structure ideal for fast dictionary look-up.
    For details, see http://en.wikipedia.org/wiki/Trie
    """

    def __init__(self, keys=None, value=lambda x: 0):
        """Construct a trie from keys.

        @type keys: random accessible object with hashable elements
        @param keys: keys from which the trie is constructed
        @type value: callable
        @param value: method to get values from keys, default sets
        everything to 0
        """
        self._trie_root = TrieNode(None)
        if keys:
            for i in keys:
                self[i] = value(i)

    def __getitem__(self, key):
        key_item_list = key.split('##')
        return self._trie_root.lookup(key_item_list, 0)

    def __setitem__(self, key, value):
        key_item_list = key.split('##')
        return self._trie_root.insert(key_item_list, 0, value)

    def longest_prefix(self, seq, offset=0):
        """Find the longest prefix of seq starting at offset that is
        in the trie.

        @type seq: random accessible object with hashable elements
        @param seq: from where the longest prefix will be extracted
        @type offset: non-negative integer
        @param offset: starting index

        @return: the index of the element next to the longest prefix
        """
        return self._trie_root.longest_prefix(seq, offset)

    def __str__(self):
        return str(self._trie_root)

    def __repr__(self):
        return "<Trie: 0x%08x>" % id(self)


class TrieNode(object):
    """A node for a trie -- you should use class Trie to access data
    stored here.
    """

    def __init__(self, label):
        """Construct a trie node with label.

        @type label: hashable
        @param label: the label for this trie node
        """
        self._label = label
        self._child = {}
        self._value = None

    def insert(self, key, offset, value):
        """Insert a node with key, if the node already exists, update
        its value.

        @type key: random accessible object of hashable elements
        @param key: the key for updating
        @type offset: non-negative interge
        @param offset: starting part of actual key
        @type value: anything you like but None
        @param value: the value for the key
        """
        if value is None:
            raise ValueError
        if offset == len(key):
            self._value = value
        else:
            first = key[offset]
            if first not in self._child:
                self._child[first] = TrieNode(first)
            self._child[first].insert(key, offset + 1, value)

    def lookup(self, key, offset):
        """Lookup a node with key.

        @type key: random accessible object of hashable elements
        @param key: the key for updating
        @type offset: non-negative interge
        @param offset: starting part of actual key
        @return: the value
        """
        if offset == len(key):
            if self._value is not None:
                return self._value
            else:
                raise KeyError
        else:
            first = key[offset]
            if first not in self._child:
                raise KeyError
            return self._child[first].lookup(key, offset + 1)

    def longest_prefix(self, seq, offset):
        """Find the longest prefix of seq starting at this node.

        @seq: random accessible object with hashable elements
        @param seq: from where the longest prefix will be extracted
        @type offset: non-negative interge
        @param offset: starting part of actual key
        @return: the index if found, otherwise None
        """
        if offset == len(seq):
            if self._value is not None:
                return offset
            else:
                return None
        else:
            first = seq[offset]
            if first not in self._child:
                if self._value is not None:
                    return offset
                else:
                    return None
            else:
                return self._child[first].longest_prefix(seq, offset + 1) or \
                       (offset if self._value is not None else None)

    def __str__(self):
        if self._child:
            return "(" + self._label + " " + \
                   " ".join([str(self._child[i])
                             for i in sorted(self._child)]) + \
                   ")"
        else:
            return "(" + self._label + ")"

    def __repr__(self):
        return "<Trie: 0x%08x>" % id(self)


def encode_word(word):
    return '##'.join(word)


def demo():
    """Demo for trie
    """
    words = ["ABC", "ABD", "ABCD", "BCD"]
    trie = Trie(words)
    sent = "ABCEABABCDF"
    print words, sent

    offset = 0
    idx = trie.longest_prefix(sent, offset)
    while offset < len(sent):
        if idx is None:
            idx = offset + 1
            pref = sent[offset:idx]
            print pref, "is not in the trie"
        else:
            pref = sent[offset:idx]
            print pref,
            print trie[pref],
            trie[pref] = 1234
            print trie[pref]
        offset = idx
        idx = trie.longest_prefix(sent, offset)


if __name__ == "__main__":
    demo()

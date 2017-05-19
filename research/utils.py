#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2016 www.drcubic.com, Inc. All Rights Reserved
#
"""
File: utils.py
Author: shileicao(shileicao@stu.xjtu.edu.cn)
Date: 17-2-12 下午9:08
"""

import numpy as np

MAX_SENTENCE_LEN = 80
MAX_SENTENCE_LEN2 = 25
MAX_WORD_LEN = 6
MAX_COMMON_LEN = 5
ENTITY_TYPES = ['@d@', '@s@', '@l@', '@o@', '@m@', '@dp@', '@bp@']
"""ENTITY_TYPES
len([PAD, O]) + len(ENTITY_TYPES) * len([S B M E])
"""


def load_w2v(path, expectDim):
    fp = open(path, "r")
    print("load data from:", path)
    line = fp.readline().strip()
    ss = line.split(" ")
    total = int(ss[0])
    dim = int(ss[1])
    assert (dim == expectDim)
    ws = []
    mv = [0 for i in range(dim)]
    second = -1
    for t in range(total):
        if ss[0] == '<UNK>':
            second = t
        line = fp.readline().strip()
        ss = line.split(" ")
        assert (len(ss) == (dim + 1))
        vals = []
        for i in range(1, dim + 1):
            fv = float(ss[i])
            mv[i - 1] += fv
            vals.append(fv)
        ws.append(vals)
    for i in range(dim):
        mv[i] = mv[i] / total
    assert (second != -1)
    # if UNK don't exist, append one more token.(This assuame that the unk is not in the last line.)
    if second == -1:
        ws.append(mv)
    if second != 1:
        t = ws[1]
        ws[1] = ws[second]
        ws[second] = t
    fp.close()
    return np.asarray(ws, dtype=np.float32)


def do_load_data(path, max_sentence_len, max_chars_per_word):
    wx = []
    cx = []
    y = []
    fp = open(path, "r")
    ln = 0
    for line in fp.readlines():
        line = line.rstrip()
        ln += 1
        if not line:
            continue
        ss = line.split(" ")
        if len(ss) != (max_sentence_len * (1 + max_chars_per_word) + 1):
            print("[line:%d]len ss:%d,origin len:%d\n%s" %
                  (ln, len(ss), len(line), line))
        assert (len(ss) == (max_sentence_len * (1 + max_chars_per_word) + 1))
        lwx = []
        lcx = []
        for i in range(max_sentence_len):
            lwx.append(int(ss[i]))
            for k in range(max_chars_per_word):
                lcx.append(
                    int(ss[max_sentence_len + i * max_chars_per_word + k]))

        wx.append(lwx)
        cx.append(lcx)
        y.append(int(ss[max_sentence_len * (1 + max_chars_per_word)]))
    fp.close()
    return np.array(wx), np.array(cx), np.array(y)


def do_load_data_attend(path, max_sentence_len, max_chars_per_word):
    wx = []
    cx = []
    y = []
    entity_info = []
    fp = open(path, "r")
    ln = 0
    for line in fp.readlines():
        line = line.rstrip()
        ln += 1
        if not line:
            continue
        ss = line.split(" ")
        if len(ss) != (max_sentence_len *
                       (1 + max_chars_per_word) + 1 + MAX_COMMON_LEN):
            print("[line:%d]len ss:%d,origin len:%d\n%s" %
                  (ln, len(ss), len(line), line))
        assert (len(ss) == (max_sentence_len *
                            (1 + max_chars_per_word) + 1 + MAX_COMMON_LEN))
        lwx = []
        lcx = []
        lentity_info = []
        for i in range(max_sentence_len):
            lwx.append(int(ss[i]))
            for k in range(max_chars_per_word):
                lcx.append(
                    int(ss[max_sentence_len + i * max_chars_per_word + k]))

        len_features = max_sentence_len * (max_chars_per_word + 1)
        for i in xrange(MAX_COMMON_LEN):
            lentity_info.append(int(ss[len_features + 1 + i]))

        wx.append(lwx)
        cx.append(lcx)
        entity_info.append(lentity_info)
        y.append(int(ss[max_sentence_len * (1 + max_chars_per_word)]))
    fp.close()
    return np.array(wx), np.array(cx), np.array(y), np.array(entity_info)


def do_load_data_char_attend(path, max_sentence_len):
    cx = []
    y = []
    entity_info = []
    fp = open(path, "r")
    ln = 0
    for line in fp.readlines():
        line = line.rstrip()
        ln += 1
        if not line:
            continue
        ss = line.split(" ")
        if len(ss) != (max_sentence_len + 1 + MAX_COMMON_LEN):
            print("[line:%d]len ss:%d,origin len:%d\n%s" %
                  (ln, len(ss), len(line), line))
        assert (len(ss) == (max_sentence_len + 1 + MAX_COMMON_LEN))
        lcx = []
        lentity_info = []
        for i in range(max_sentence_len):
            lcx.append(int(ss[i]))

        len_features = max_sentence_len
        for i in xrange(MAX_COMMON_LEN):
            lentity_info.append(int(ss[len_features + 1 + i]))

        cx.append(lcx)
        entity_info.append(lentity_info)
        y.append(int(ss[max_sentence_len]))
    fp.close()
    return np.array(cx), np.array(y), np.array(entity_info)


def do_load_data_joint_attend(path, max_sentence_len):
    cx = []
    ner_y = []
    clfier_y = []
    entity_info = []
    fp = open(path, "r")
    ln = 0
    for line in fp.readlines():
        line = line.rstrip()
        ln += 1
        if not line:
            continue
        ss = line.split(" ")
        if len(ss) != (max_sentence_len * 2 + 1 + MAX_COMMON_LEN):
            print("[line:%d]len ss:%d,origin len:%d\n%s" %
                  (ln, len(ss), len(line), line))
        assert (len(ss) == (max_sentence_len + 1 + MAX_COMMON_LEN))
        lcx = []
        lentity_info = []
        lner_y = []
        for i in range(max_sentence_len):
            lcx.append(int(ss[i]))

        for i in range(max_sentence_len):
            lner_y.append(int(ss[max_sentence_len + i]))

        for i in xrange(MAX_COMMON_LEN):
            lentity_info.append(int(ss[max_sentence_len * 2 + 1 + i]))

        cx.append(lcx)
        ner_y.append(lner_y)
        entity_info.append(lentity_info)
        clfier_y.append(int(ss[max_sentence_len * 2]))
    fp.close()
    return np.array(cx), np.array(ner_y), np.array(cx), np.array(
        clfier_y), np.array(entity_info)


def do_load_data_char_common(path, max_sentence_len):
    cx = []
    y = []
    fp = open(path, "r")
    ln = 0
    for line in fp.readlines():
        line = line.rstrip()
        ln += 1
        if not line:
            continue
        ss = line.split(" ")
        if len(ss) != (max_sentence_len + 1):
            print("[line:%d]len ss:%d,origin len:%d\n%s" %
                  (ln, len(ss), len(line), line))
        assert (len(ss) == (max_sentence_len + 1))
        lcx = []
        for i in range(max_sentence_len):
            lcx.append(int(ss[i]))

        cx.append(lcx)
        y.append(int(ss[max_sentence_len]))
    fp.close()
    return np.array(cx), np.array(y)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2016 www.drcubic.com, Inc. All Rights Reserved
#
"""
File: process_medical_data.py
Author: shileicao(shileicao@stu.xjtu.edu.cn)
Date: 17-2-22 下午8:06
"""

import codecs
import os
import re
import sys

import jieba

LONG_LEN = 0
MAX_LEN = 80
LINE_NUM = 0
MIN_LEN = 6

jieba.load_userdict(os.path.join('../data', 'words.txt'))

PUNCTUATION_DUP_RM = re.compile(ur'([\?|？|\!|！|。|，|\,|\(|\)|（|）|\*|×])\1+',
                                re.IGNORECASE)


def process_line(sentence, word_output_file, char_output_file):
    global LONG_LEN
    global MAX_LEN
    global LINE_NUM

    sentence = re.sub(ur'([\?|？|\!|！|。|，|\,])\1+', r'\1', sentence)
    if len(sentence) < MIN_LEN:
        return
    LINE_NUM += 1
    char_output_file.write(' '.join(list(sentence)) + '\n')
    if len(sentence) > MAX_LEN:
        LONG_LEN += 1
    tokens = jieba.lcut(sentence, cut_all=False)
    #     print ' '.join(tokens)
    word_output_file.write(' '.join(tokens) + '\n')


def main(argv):
    argv = sys.argv
    if len(argv) < 4:
        print 'usage: %s <inputfile> <word_output_file> <char_output_file>' % argv[
            0]
        sys.exit(0)
    input_file = argv[1]
    word_output_file = argv[2]
    char_output_file = argv[3]
    count_line = 0
    input_file = open(input_file, 'r')
    word_output_file = codecs.open(word_output_file, 'w', 'utf-8')
    char_output_file = codecs.open(char_output_file, 'w', 'utf-8')
    i = 0
    for sentence in input_file:
        i += 1
        print 'procossing line %d' % i
        process_line(
            sentence.decode('utf-8'), word_output_file, char_output_file)
    word_output_file.close()
    char_output_file.close()
    input_file.close()
    print 'long_line: %d' % LONG_LEN
    print 'count_line: %d' % count_line


if __name__ == '__main__':
    main(sys.argv)

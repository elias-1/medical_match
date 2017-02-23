#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2016 www.drcubic.com, Inc. All Rights Reserved
#
"""
File: process_medical_data.py.py
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

PUNCTUATION_DUP_RM = re.compile(ur'([\?|？|\!|！|。|，|\,])\1+', re.IGNORECASE)


def process_line(sentence, output_file):
    global LONG_LEN
    global MAX_LEN
    global LINE_NUM

    sentence = ''.join(sentence.split(':')[1:])
    sentence = re.sub(ur'([\?|？|\!|！|。|，|\,])\1+', r'\1', sentence)
    if len(sentence) < MIN_LEN:
        return
    LINE_NUM += 1
    if len(sentence) > MAX_LEN:
        LONG_LEN += 1
    tokens = jieba.lcut(sentence.encode('utf-8'), cut_all=False)
    output_file.writer(' '.join(tokens) + '\n')


def main(argv):
    argv = sys.argv
    if len(argv) < 3:
        print 'usage:', argv[0], ' <inputfile> <outputfile>'
        sys.exit(0)
    input_file = argv[1]
    output_file = argv[2]
    count_line = 0
    input_file = open(input_file, 'r')
    output_file = codecs.open(output_file, 'w', 'utf-8')
    for sentence in input_file:

        process_line(sentence, output_file)
    output_file.close()
    input_file.close()
    print 'long_line:', LONG_LEN
    print 'count_line:', count_line


if __name__ == '__main__':
    main(sys.argv)

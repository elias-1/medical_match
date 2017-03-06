#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 www.drcubic.com, Inc. All Rights Reserved
#
"""
File: entity_data_clean.py
Author: shileicao(shileicao@stu.xjtu.edu.cn)
Date: 17-3-6 下午6:49
"""

import csv
import json
import sys


def processLine(data, sent2label, new_entity_out):
    error_rows = []
    know_sent = set()
    for row in data:
        new_row = [
            row[item].strip().decode('utf-8') for item in row
            if row[item].strip() != ''
        ]
        if new_row[0] in know_sent:
            continue
        else:
            know_sent.add(new_row[0])
            if new_row[0] in sent2label:
                new_row = [sent2label[new_row[0]]] + new_row
                new_entity_out.writerow(new_row)
            else:
                error_rows.append(new_row)
    return error_rows


def main(argc, argv):
    if argc < 4:
        print('Usage:%s <old_entity> <new_entity> <clfier_data>' % (argv[0]))
        exit(1)

    sent2label = {}
    know_sent = set()
    with open(argv[3], 'r') as f:
        data = json.load(f)
        for key in data.keys():
            for sent in data[key]:
                if sent in know_sent:
                    print 'sent label repeted: %s label1:%s label2:%s' % (
                        sent, key, sent2label[sent.strip()])
                else:
                    know_sent.add(sent)
                sent2label[sent.strip()] = key

    new_entity_out = csv.writer(open(argv[2], 'w'))

    with open(argv[1], 'r') as f:
        data = csv.reader(f, delimiter=',')
        error_rows = processLine(data, sent2label, new_entity_out)

    if error_rows:
        new_entity_out.writerow(['Error: No class found:'])
        new_entity_out.writerows(error_rows)
    new_entity_out.close()


if __name__ == '__main__':
    main(len(sys.argv), sys.argv)

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
reload(sys)
sys.setdefaultencoding('utf-8')


def processLine(data, sent2label, new_entity_out):
    print len(sent2label)
    error_rows = []
    know_sent = set()
    for row in data:
        new_row = [item for item in row[1:] if item.strip()]
        sent = new_row[1].decode('utf-8').strip()
        if sent in know_sent:
            continue
        else:
            know_sent.add(sent)
            if sent in sent2label:
                new_row = [sent2label[sent].encode('utf-8')] + new_row[1:]
                new_entity_out.writerow(new_row)
                sent2label.pop(sent)
            else:
                error_rows.append(new_row)
    print len(know_sent)

    with open('../data/no_clfier.json', 'w') as f:
        json.dump(sent2label, f, indent=4, ensure_ascii=False)
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


if __name__ == '__main__':
    main(len(sys.argv), sys.argv)

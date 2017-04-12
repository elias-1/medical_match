#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 www.drcubic.com, Inc. All Rights Reserved
#
"""
File: send_kg_data.py
Author: shileicao(shileicao@stu.xjtu.edu.cn)
Date: 2017/4/12 11:20
"""
import json
import re
import sys
from StringIO import StringIO

import pycurl

ENTITY_WITH_ID = re.compile('edu\/(.*?)\/(.*?)>', re.IGNORECASE)
NO_NAME = 0
COUNT = 0

useless_list = [
    "", u"", None, False, " ", "  ", "   ", "    ", "     ", "-", "--", "---"
]


def insert_record(table_name, col, row):
    url = 'http://202.117.16.221:9999/deep_serving/%s/' % table_name
    insert_data = {}
    for i, j in zip(col, row):
        if j not in useless_list:
            insert_data[i] = j
    insert_data = json.dumps(insert_data)
    storage = StringIO()
    c = pycurl.Curl()
    c.setopt(pycurl.URL, url)
    c.setopt(pycurl.HTTPHEADER, ['Content-Type: application/json'])
    c.setopt(pycurl.CUSTOMREQUEST, "POST")
    c.setopt(pycurl.POST, 1)
    c.setopt(pycurl.POSTFIELDS, insert_data)
    c.setopt(c.WRITEFUNCTION, storage.write)
    c.perform()
    c.close()
    res = storage.getvalue()
    response = json.loads(res)
    retu = response['Return']
    return retu


def extract_id_name(f):
    id2name = {}
    row_num = 1
    for line in f.readlines():
        row_num += 1
        line = line.replace('\"', '').decode('utf-8')
        row = line[:line.rindex('.')].strip().split('\t')
        entity_with_relation = ENTITY_WITH_ID.findall(row[1])
        if len(row) == 2 and entity_with_relation[0][0] == 'property':
            continue
        assert (len(row) == 3)
        if entity_with_relation[0][0] == 'property':
            entity_with_id = ENTITY_WITH_ID.findall(row[0])
            if entity_with_id[0][1] not in id2name:
                id2name[entity_with_id[0][1]] = row[2].strip()
    print('total entity names:%d' % len(id2name))
    return id2name


def insert2property(property_data):
    global COUNT
    col_name = ['entity_id', 'entity_type', 'property_name', 'property_value']
    res = insert_record('Property', col_name, property_data)
    COUNT += 1
    print('process: %d, return: %d' % (COUNT, res))


def insert2relation(relation_data):
    global COUNT
    col_name = [
        'entity_id1', 'entity_name1', 'entity_type1', 'relation', 'entity_id2',
        'entity_name2', 'entity_type2'
    ]
    res = insert_record('Entity_relation', col_name, relation_data)
    COUNT += 1
    print('process: %d, return: %d' % (COUNT, res))


def process_row(line, id2name, table_name):
    global NO_NAME
    line = line.replace('\"', '').decode('utf-8')
    row = line[:line.rindex('.')].strip().split('\t')
    entity_with_relation = ENTITY_WITH_ID.findall(row[1])
    if len(row) == 2 and entity_with_relation[0][0] == 'property':
        return
    assert (len(row) == 3)
    entity_with_id = ENTITY_WITH_ID.findall(row[0])
    entity_id1 = entity_with_id[0][1]
    entity_type1 = entity_with_id[0][0]
    relation_or_property = entity_with_relation[0][1]
    if table_name == 'property':
        if entity_with_relation[0][0] == 'property':
            property_data = (entity_id1, entity_type1, relation_or_property,
                             row[2])
            insert2property(property_data)
    elif table_name == 'entity_relation':
        if entity_with_relation[0][0] != 'property':
            entity_with_id2 = ENTITY_WITH_ID.findall(row[2])
            entity_id2 = entity_with_id2[0][1]
            entity_type2 = entity_with_id2[0][1]
            if entity_id1 not in id2name or entity_id2 not in id2name:
                NO_NAME += 1
                return
            relation_data1 = (entity_id1, id2name[entity_id1], entity_type1)
            relation_data2 = (entity_id2, id2name[entity_id2], entity_type2)
            relation_data = relation_data1 + (relation_or_property,
                                              ) + relation_data2
            insert2relation(relation_data)


def main(argc, argv):
    if argc < 2:
        print("Usage:%s <data>" % (argv[0]))

    with open(argv[1], 'rb') as f:
        id2name = extract_id_name(f)
        f.seek(0)
        row_num = 1
        for line in f.readlines():
            process_row(line, id2name, 'property')
            row_num += 1

        f.seek(0)
        row_num = 1
        for line in f.readlines():
            process_row(line, id2name, 'entity_relation')
            row_num += 1

    print('entity id no name num:%d' % NO_NAME)


if __name__ == "__main__":
    # pdb.set_trace()
    main(len(sys.argv), sys.argv)

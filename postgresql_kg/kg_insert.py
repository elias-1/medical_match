#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 www.drcubic.com, Inc. All Rights Reserved
#
"""
File: kg_insert.py
Author: shileicao(shileicao@stu.xjtu.edu.cn)
Date: 2017/3/21 17:10
"""

import sys
import csv
import re

import psycopg2

ENTITY_WITH_ID = re.compile('edu\/(.*?)\/(.*?)>', re.IGNORECASE)
conn = psycopg2.connect("dbname=kgdata user=dbuser")

def create_table(sql):
    cur = conn.cursor()
    cur.execute(sql)
    cur.close()

def create_kg_table():
    sql1 = """
    """
    sql2 = """
    """
    create_table(sql1)
    create_table(2)

def extract_id_name(csv_reader):
    id2name = {}
    for row in csv_reader:
        entity_with_relation = ENTITY_WITH_ID.findall(row[1])
        if entity_with_relation[0][0] == 'property':
            entity_with_id = ENTITY_WITH_ID.findall(row[0])
            if entity_with_id[0][1] not in id2name:
                id2name[entity_with_id[0][1]] = row[2]
    print('total entity names:%d' % len(id2name))
    return id2name


def insert2db(sql, data):
    cur = conn.cursor()
    cur.execute(sql, data)
    cur.close()

def insert2property(property_data):
    sql = ''
    insert2db(sql, property_data)

def insert2relation(relation_data):
    sql = ''
    insert2db(sql,relation_data)

def process_row(row, id2name):
    entity_with_relation = ENTITY_WITH_ID.findall(row[1])
    entity_with_id = ENTITY_WITH_ID.findall(row[0])
    entity_id1 = entity_with_id[0][1]
    entity_type1 = entity_with_id[0][1]
    relation_or_property = entity_with_relation[0][1]
    if entity_with_relation[0][0] == 'property':
        property_data = (entity_id1, entity_type1, relation_or_property, row[2])
        insert2property(property_data)
    else:
        entity_with_id2 = ENTITY_WITH_ID.findall(row[2])
        entity_id2 = entity_with_id2[0][1]
        entity_type2 = entity_with_id2[0][1]
        relation_data1 = (entity_id1, id2name[entity_id1], entity_type1)
        relation_data2 = (entity_id2, id2name[entity_id2], entity_type2)
        relation_data = relation_data1 + (relation_or_property,) + relation_data2
        insert2relation(relation_data)

def main(argc, argv):
    if argc < 2:
        print("Usage:%s <data>" % (argv[0]))

    create_kg_table()
    with open(argv[1], 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        id2name = extract_id_name(csv_reader)
        f.seek(0)
        for row in csv_reader:
            process_row(row, id2name)


if __name__ == "__main__":
    main(len(sys.argv), sys.argv)


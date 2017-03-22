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

import re
import sys

import psycopg2

ENTITY_WITH_ID = re.compile('edu\/(.*?)\/(.*?)>', re.IGNORECASE)
conn = psycopg2.connect(
    'dbname=kgdata user=dbuser password=112233 host=127.0.0.1')


def create_table(sql):
    try:
        cur = conn.cursor()
        cur.execute(sql)
        # close communication with the PostgreSQL database server
        cur.close()
        # commit the changes
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)


def create_kg_table():
    sql1 = """CREATE TABLE property (
                  entity_id varchar(20) PRIMARY KEY,
                  entity_type varchar(50) not null,
                  property_name varchar(50) not null,
                  property_value varchar(255) not null
              )
    """
    sql2 = """CREATE TABLE relation (
                  id SERIAL PRIMARY KEY,
                  entity_id1 varchar(20),
                  entity_name1 varchar(255) not null,
                  entity_type1 varchar(50) not null,
                  relation varchar(50) not null,
                  entity_id2 varchar(20) not null,
                  entity_name2 varchar(255) not null,
                  entity_type2 varchar(50) not null,
                  FOREIGN KEY (entity_id1)
                  REFERENCES property (entity_id)
                  ON UPDATE CASCADE ON DELETE CASCADE,
                  FOREIGN KEY (entity_id2)
                  REFERENCES property (entity_id)
                  ON UPDATE CASCADE ON DELETE CASCADE,
              )
    """
    create_table(sql1)
    create_table(sql2)


def extract_id_name(f):
    id2name = {}
    for line in f.readlines():
        line.replace('\"', '')
        row = line[:-1].strip().split('\t')
        assert (len(row) == 3)
        row[2] = row[2][:-1].strip()
        entity_with_relation = ENTITY_WITH_ID.findall(row[1])
        if entity_with_relation[0][0] == 'property':
            entity_with_id = ENTITY_WITH_ID.findall(row[0])
            if entity_with_id[0][1] not in id2name:
                id2name[entity_with_id[0][1]] = row[2]
    print('total entity names:%d' % len(id2name))
    return id2name


def insert2db(sql, data):
    try:
        cur = conn.cursor()
        # execute the INSERT statement
        cur.executemany(sql, data)
        # commit the changes to the database
        # close communication with the database
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)


def insert2property(property_data):
    sql = """INSERT INTO property (entity_id, entity_type, property_name, property_value)
                 VALUES (%s, %s, %s, %s);"""
    insert2db(sql, property_data)


def insert2relation(relation_data):
    sql = """INSERT INTO relation (entity_id1, entity_name1, entity_type1, relation, entity_id2, entity_name2, entity_type2)
                 VALUES (%s, %s, %s, %s %s, %s, %s);"""
    insert2db(sql, relation_data)


def process_row(line, id2name):
    line.replace('\"', '')
    row = line[:-1].strip().split('\t')
    assert (len(row) == 3)
    entity_with_relation = ENTITY_WITH_ID.findall(row[1])
    entity_with_id = ENTITY_WITH_ID.findall(row[0])
    entity_id1 = entity_with_id[0][1]
    entity_type1 = entity_with_id[0][1]
    relation_or_property = entity_with_relation[0][1]
    if entity_with_relation[0][0] == 'property':
        property_data = (entity_id1, entity_type1, relation_or_property,
                         row[2])
        insert2property(property_data)
    else:
        entity_with_id2 = ENTITY_WITH_ID.findall(row[2])
        entity_id2 = entity_with_id2[0][1]
        entity_type2 = entity_with_id2[0][1]
        relation_data1 = (entity_id1, id2name[entity_id1], entity_type1)
        relation_data2 = (entity_id2, id2name[entity_id2], entity_type2)
        relation_data = relation_data1 + (relation_or_property,
                                          ) + relation_data2
        insert2relation(relation_data)


def main(argc, argv):
    if argc < 2:
        print("Usage:%s <data>" % (argv[0]))

    create_kg_table()
    with open(argv[1], 'r') as f:
        id2name = extract_id_name(f)
        f.seek(0)
        for line in f.readlines():
            process_row(line, id2name)
    conn.commit()
    conn.close()


if __name__ == "__main__":
    main(len(sys.argv), sys.argv)

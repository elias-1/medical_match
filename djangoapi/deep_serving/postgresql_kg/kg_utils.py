#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 www.drcubic.com, Inc. All Rights Reserved
#
"""
File: kg_utils.py
Author: shileicao(shileicao@stu.xjtu.edu.cn)
Date: 2017/4/7 13:51
"""

import os

import psycopg2

from ..utils.utils import config

app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_file_dir = os.path.join(app_dir, 'config', 'config.conf')
params = config(filename=config_file_dir, section='postgresql')
conn = psycopg2.connect(**params)


def search_sql(sql):
    """ query parts from the parts table """
    try:
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        cur.close()
        return rows
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        return None


# TODO: implementation
def kg_entity_identify(sentence):
    """判断sentence是否为医疗实体，如果是，返回其概述。否则success为0，result为None
    
    Args:
        sentence: 实体短语
    Returns:
        success: 是否成功
        result: list; if success 0, None, 
        
    """
    sql = """SELECT property_value FROM property WHERE property_name = 'desc' and entity_id in ( SELECT entity_id FROM property WHERE property_name = 'name' and property_value = '%s');"""
    sql_result = search_sql(sql % (sentence))
    result_list = []
    success = 0
    if len(sql_result) > 0:
        success = 1
        for item in sql_result:
            result_list.append(item[0])
    return result_list, success
    #pass


# TODO: implementation
def kg_entity_summary(entities):
    """返回entities的概述
    
    Args:
        entities: list
    Returns:
        entitiy_summarys: list
        """
    evalues = '\',\''.join(entities)
    sql = """SELECT property_value FROM property WHERE property_name = 'desc' and entity_id in ( SELECT entity_id FROM property WHERE property_name = 'name' and property_value IN ('%s'));"""
    sql_result = search_sql(sql % (evalues))
    result_list = []
    if len(sql_result) > 0:
        success = 1
        for item in sql_result:
            result_list.append(item[0])
    return result_list, success

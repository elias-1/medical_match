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
    """
    
    Args:
        sentence: 
    Returns:
        
    """
    pass


# TODO: implementation
def kg_entity_summary(entities):
    """返回entities的概述
    
    Args:
        entities: list
    Returns:
        entitiy_summarys: list
        """
    pass

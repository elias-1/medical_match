#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 www.drcubic.com, Inc. All Rights Reserved
#
"""
File: kg_utils.py
Author: shileicao(shileicao@stu.xjtu.edu.cn)
Date: 2017/4/7 13:51

Not done!!
"""

from ..models import Property


def kg_entity_identify(sentence):
    """判断sentence是否为医疗实体，如果是，返回其概述。否则success为0，result为None
    
    Args:
        sentence: 实体短语
    Returns:
        success: 是否成功
        result_list: list; if success 0, None, 
        
    """
    sql = """SELECT distinct pid, property_value
             FROM deep_serving_property
             WHERE property_name = 'desc'
             and entity_id in (
                  SELECT entity_id 
                  FROM Property 
                  WHERE property_name = 'name' 
                  and property_value = '%s');"""

    sql_result = Property.objects.raw(sql % (sentence))
    result_list = []
    success = 0

    for item in sql_result:
        success = 1
        result_list.append(item.property_value)
    return result_list, success


def kg_entity_summary(entities):
    """返回entities的概述
    
    Args:
        entities: list
    Returns:
        entitiy_summarys: list
        success: 0/1
        """
    evalues = "','".join(entities)
    sql = """SELECT distinct a.pid, a.property_value as name, b.property_value as desc 
             FROM deep_serving_property a left join deep_serving_property b 
                on a.entity_id = b.entity_id
             WHERE a.property_name = 'name' 
                and a.property_value IN ('%s')
                and b.property_name = 'desc' ;"""
    sql_result = Property.objects.raw(sql % (evalues))
    entitiy_summarys = []
    success = 0
    summary_dict = {}

    for item in sql_result:
        success = 1
        if not summary_dict.has_key(item.name):
            summary_dict[item.name] = []
        summary_dict[item.name].append(item.desc)
    for key in summary_dict:
        re_sent = key + '的概述：' + " \n ".join(summary_dict[key])
        entitiy_summarys.append(re_sent)
    return entitiy_summarys, success


def kg_search_body_part(entities):
    """问部位

    Args:
        entities: list
    Returns:
        body_part: list
        success: 0/1
    """
    evalues = "','".join(entities)
    sql = """SELECT distinct entity_name1, entity_name2 
             FROM deep_serving_entity_relation
             WHERE relation like '%%Body%%' 
                and entity_name1 IN ('%s')
             ORDER BY entity_name1;"""
    sql_result = Property.objects.raw(sql % (evalues))
    body_part = []
    success = 0
    body_dict = {}
    if len(sql_result) > 0:
        success = 1
        for item in sql_result:
            if not body_dict.has_key(item[0]):
                body_dict[item[0]] = []
            body_dict[item[0]].append(item[1])
    for key in body_dict:
        re_sent = key + '的部位：' + " , ".join(body_dict[key])
        body_part.append(re_sent)
    return body_part, success


def kg_search_price(entities):
    """问价格

    Args:
        entities: list
    Returns:
        entity_price: str
        success: 0/1
    """
    evalues = "','".join(entities)
    sql = """SELECT distinct a.pid, a.property_value as name, b.property_value as price
             FROM deep_serving_property a left join deep_serving_property b 
                on a.entity_id = b.entity_id
             WHERE a.property_name = 'name' 
                and a.property_value IN ('%s')
                and b.property_name = 'price' ;"""
    sql_result = Property.objects.raw(sql % (evalues))
    entitiy_price = []
    success = 0
    price_dict = {}

    for item in sql_result:
        success = 1
        if not price_dict.has_key(item.name):
            price_dict[item.name] = []
        price_dict[item.name].append(item.price)
    for key in price_dict:
        re_sent = key + '的价格：' + " , ".join(price_dict[key])
        entitiy_price.append(re_sent)
    return entitiy_price, success


def kg_search_department(entities):
    """问科室

    Args:
        entities: list
    Returns:
        department_list: list
        success: 0/1
    """
    evalues = "','".join(entities)
    sql = """SELECT distinct entity_name1, entity_name2 
             FROM deep_serving_entity_relation
             WHERE relation like '%%Dep%%' 
                and entity_name1 IN ('%s')
             ORDER BY entity_name1;"""
    sql_result = Property.objects.raw(sql % (evalues))
    department_list = []
    success = 0
    department_dict = {}
    if len(sql_result) > 0:
        success = 1
        for item in sql_result:
            if not department_dict.has_key(item[0]):
                department_dict[item[0]] = []
            department_dict[item[0]].append(item[1])
    for key in department_dict:
        re_sent = key + '的科室：' + " , ".join(department_dict[key])
        department_list.append(re_sent)
    return department_list, success

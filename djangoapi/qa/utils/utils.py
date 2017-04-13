#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 www.drcubic.com, Inc. All Rights Reserved
#
"""
File: utils.py
Author: shileicao(shileicao@stu.xjtu.edu.cn)
Date: 2017/4/7 14:14
"""

from configparser import ConfigParser


def config(filename, section):
    # create a parser
    parser = ConfigParser()
    # read config file
    parser.read(filename)

    # get section
    section_info = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            section_info[param[0]] = param[1]
    else:
        raise Exception(
            'Section {0} not found in the {1} file'.format(section, filename))

    return section_info

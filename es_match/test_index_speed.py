#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 www.drcubic.com, Inc. All Rights Reserved
#
"""
File: test_index_speed.py
Author: shileicao(shileicao@stu.xjtu.edu.cn)
Date: 17-2-27 上午9:01
"""

import random

import time


def test_search_speed(search_sets, searchs):
    start = time.clock()
    count = 0
    for item in searchs:
        if item in search_sets:
            count += 1
    end = time.clock()
    return end - start, count


if __name__ == "__main__":

    list_search_sets = range(1, 1000000)
    random.shuffle(list_search_sets)

    set_search_sets = set(list_search_sets)

    cost_time, count = test_search_speed(list_search_sets, range(0, 1000))
    print('cost time:%f s, count:%d' % (cost_time, count))

    cost_time, count = test_search_speed(set_search_sets, range(0, 1000))
    print('cost time:%f s, count:%d' % (cost_time, count))

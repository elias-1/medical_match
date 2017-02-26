#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2016 www.drcubic.com, Inc. All Rights Reserved
#
"""
File: send_entity.py
Author: shileicao(shileicao@stu.xjtu.edu.cn)
Date: 17-2-25 下午2:51
"""

import json
import os
import sys
from collections import OrderedDict
from StringIO import StringIO

import pycurl
import pypinyin

reload(sys)
sys.setdefaultencoding('utf8')

useless_list = [
    "", u"", None, False, " ", "  ", "   ", "    ", "     ", "-", "--", "---"
]


def hanzi_list2pinyin(hanzi_list):
    return [
        pypinyin.pinyin(
            word, style=pypinyin.NORMAL)[0][0] for word in hanzi_list
    ]


def insert_record(col, row):
    # print dop['op']
    url = 'http://1.85.37.136:9999/medknowledge/op/'
    url = 'http://202.117.54.88:9999/medknowledge/op/'
    url = 'http://aliyun:9999/medknowledge/op/'
    insert_data = OrderedDict()
    for i, j in zip(col, row):
        if j not in useless_list:
            insert_data[i] = j

    insert_data = json.dumps(insert_data)
    storage = StringIO()
    try:
        c = pycurl.Curl()
        c.setopt(pycurl.URL, url)
        c.setopt(pycurl.HTTPHEADER, ['Content-Type: application/json'])
        # c.setopt(pycurl.CUSTOMREQUEST,dop['op'])
        # c.setopt(pycurl.CUSTOMREQUEST,'POST')
        c.setopt(pycurl.CUSTOMREQUEST, 'PUT')
        # c.setopt(pycurl.CUSTOMREQUEST,'DELETE')
        c.setopt(pycurl.POST, 1)
        c.setopt(pycurl.CONNECTTIMEOUT, 3)  #链接超时
        c.setopt(pycurl.POSTFIELDS, insert_data)
        c.setopt(c.WRITEFUNCTION, storage.write)
        c.perform()
        c.close()
    except:
        print('已断开')
        return 2
    res = storage.getvalue()
    mywritefile('temp.html', res)
    response = json.loads(res)
    print res
    retu = response['Return']
    # print res
    return retu


def send_name_search(data_path):
    x = raw_input('input first letter:\n\t').upper()
    for folder in sorted(os.listdir(data_path)):
        print folder
        if x.startswith('A'):
            pass
        elif not folder.startswith(x):
            continue
        path = os.path.join(data_path, folder)
        for index, fi in enumerate(sorted(os.listdir(path))):
            print fi
            d = myreadjson(os.path.join(path, fi))
            table = folder.split('_')[0].title()
            xid = d['Id']
            col_name = []
            urow = []
            col_name.append('Table')
            urow.append(table)
            col_name.append('Id')
            urow.append(xid)
            for l in d['Table']:
                k = l.keys()[0]
                v = l.values()[0]
                pinyin_without_tone, pinyin_first_letter = word_to_pinyin_lower(
                    v)
                col_name.append(k)
                urow.append(v)
                if k not in pureSet:
                    col_name.append(k + 'Pinyin')
                    v = pinyin_without_tone[:
                                            240] + ' ' + pinyin_first_letter[:
                                                                             14]
                    urow.append(v)
            s = mydumps(d['Content'])
            # s = d['Content']
            col_name.append('Content')
            urow.append(s)
            ret = 2
            while (ret == 2):
                # break
                ret = insert_record(col_name, urow)
            print urow[0], "Count: " + str(index) + "   Return: " + str(ret)
            # break
    pass


def send_name_auto(srcfile):
    nameNumDict = myreadjson(srcfile)
    for index, (name, num) in enumerate(nameNumDict.items()):
        print index
        pinyin_without_tone, pinyin_first_letter = word_to_pinyin_lower(name)
        dtable = OrderedDict()
        dtable['Name'] = name
        dtable['Pinyin'] = pinyin_without_tone
        dtable['PinyinFirstLetter'] = pinyin_first_letter
        dtable['Num'] = num
        col_name = []
        urow = []
        col_name.append('Type')
        urow.append('auto')
        col_name.append('Id')
        urow.append(name)
        col_name.append('Name')
        urow.append(name)
        col_name.append('Tale')
        urow.append(dtable)
        ret = 2
        while (ret == 2):
            ret = insert_record(col_name, urow)
        # print urow[0],"Count: "+str(index)+"   Return: "+str(ret)


if __name__ == '__main__':
    data_path = '../../result/es-json/name-search-json/'
    os.system('clear')
    srcfile = '../../result/es-json/name-num-dict.json'
    # send_name_auto(srcfile)
    send_name_search(data_path)

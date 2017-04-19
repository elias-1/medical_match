#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 www.drcubic.com, Inc. All Rights Reserved
#
"""
File: api_speed_test.py
Author: shileicao(shileicao@stu.xjtu.edu.cn)
Date: 2017/4/19 10:33
"""

import json
import sys
import time
import urllib
from StringIO import StringIO

import pycurl

reload(sys)
sys.setdefaultencoding('utf-8')

curl_url = """http://202.117.16.221:7777/qa/sentence_process/?q={{"sentence":"{sentence}"}}"""


def ops_api(url):
    storage = StringIO()
    try:
        nurl = url
        c = pycurl.Curl()
        c.setopt(pycurl.URL, nurl)
        c.setopt(pycurl.HTTPHEADER, ['Content-Type: application/json'])
        if url.startswith('https'):
            c.setopt(pycurl.SSL_VERIFYPEER, 0)
            c.setopt(pycurl.SSL_VERIFYHOST, 0)
        c.setopt(pycurl.CONNECTTIMEOUT, 10)
        c.setopt(c.WRITEFUNCTION, storage.write)
        c.perform()
        c.close()
    except:
        return None
    response = storage.getvalue()
    res = json.loads(response)
    return res


def process_sent(sentence):
    start = time.time()
    url = curl_url.format(sentence=urllib.quote(sentence.encode('utf-8')))
    result = ops_api(url)
    end = time.time()
    return end - start, result


def main():
    sent = u'乙肝大三阳，肝功能ALT6I,AST7O,球蛋白比1,06严重吗?'
    use_time, result = process_sent(sent)
    print('use time: %f, result: %s' % (use_time, result))


if __name__ == "__main__":
    main()

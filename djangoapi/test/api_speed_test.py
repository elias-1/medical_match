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

# url = """http://59.110.52.133:9999/qa/sentence_process/?q={%22sentence%22:%22%E4%BD%A0%E5%A5%BD%22}"""
# curl_url = """http://202.117.16.221:7777/qa/sentence_process/?q={{%22sentence%22:%22{sentence}%22}}"""

#curl_url = """http://59.110.52.133:9999/qa/sentence_process/?q={{"sentence":"{sentence}"}}"""
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


def main(argc, argv):
    if argc < 2:
        print("Usage:%s <data>" % (argv[0]))

    with open(argv[1], 'r') as f:
        data = json.load(f)
        num = 0
        total_time = 0
        for key in data.keys():
            for sent in data[key]:
                num += 1
                use_time, result = process_sent(sent)
                if result:
                    result = result.items()
                print('processing %d, use time: %f, result: %s' %
                      (num, use_time, result))
                total_time += use_time

    average_time = total_time / num
    print('processed %d sentence, total time: %f, average time %f' %
          (num, total_time, average_time))


if __name__ == "__main__":
    main(len(sys.argv), sys.argv)

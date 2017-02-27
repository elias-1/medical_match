# encoding:UTF-8

import codecs
import csv
import json
import pprint
import sys
import time

import es_match
import jieba
import jieba.posseg
import pypinyin
from core import entity_identify

if __name__ == "__main__":
    stime = time.clock()
    result = entity_identify.entity_identify(u'感冒，发骚，咳嗽吃什么药？')
    dstr = json.dumps(result, ensure_ascii=False, indent=4)
    dstr = unicode.encode(dstr, 'utf-8')
    with open('qa_result.json', 'wb') as f:
        f.write(dstr)
    etime = time.clock()
    print "read: %f s" % (etime - stime)

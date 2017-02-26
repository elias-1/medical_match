# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 16:02:34 2015

@author: qian
"""

import csv
import json
from StringIO import StringIO

import pycurl

useless_list = [
    "", u"", None, False, " ", "  ", "   ", "    ", "     ", "-", "--", "---"
]


def insert_record(col, row):
    url = 'http://117.32.155.61:8888/testapi/news/'
    insert_data = {}
    for i, j in zip(col, row):
        if j not in useless_list:
            insert_data[i] = j
    insert_data = json.dumps(insert_data)
    storage = StringIO()
    c = pycurl.Curl()
    c.setopt(pycurl.URL, url)
    c.setopt(pycurl.HTTPHEADER, ['Content-Type: application/json'])
    c.setopt(pycurl.CUSTOMREQUEST, "POST")
    c.setopt(pycurl.POST, 1)
    c.setopt(pycurl.POSTFIELDS, insert_data)
    c.setopt(c.WRITEFUNCTION, storage.write)
    c.perform()
    c.close()
    res = storage.getvalue()
    response = json.loads(res)
    retu = response['Return']
    return retu


csv_ref = csv.reader(
    open("news_sample.csv", 'rb'), delimiter='\t', quotechar='"')
col_name = csv_ref.next()
col_name = [unicode(x, 'utf-8') for x in col_name]
count = 0
for row in csv_ref:
    count = count + 1
    urow = [unicode(x, 'utf-8') for x in row]
    ret = insert_record(col_name, urow)
    print "Count: " + str(count) + "   Return: " + str(ret)

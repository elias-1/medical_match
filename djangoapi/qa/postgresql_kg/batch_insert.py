#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 www.drcubic.com, Inc. All Rights Reserved
#
"""
File: batch_insert.py
Author: shileicao(shileicao@stu.xjtu.edu.cn)
Date: 2017/4/18 16:06
"""
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Batch insert data to api"""

import json
import re
import sys
import time
from StringIO import StringIO
from threading import Condition, Thread

import pycurl
from six.moves import queue as Queue
from six.moves import xrange

QUEUE_NUM_BATCH = 1000

ENTITY_WITH_ID = re.compile('edu\/(.*?)\/(.*?)>', re.IGNORECASE)
MAX_OUTPUT_THREAD = 10
SLEEP_TIME = 20

useless_list = [
    "", u"", None, False, " ", "  ", "   ", "    ", "     ", "-", "--", "---"
]


def insert_record(table_name, col, row):
    # url = 'https://202.117.16.221:7777/qa/%s/' % table_name
    url = 'http://localhost:9999/qa/%s/' % table_name
    insert_data = {}
    for i, j in zip(col, row):
        if j not in useless_list:
            insert_data[i] = j
    insert_data = json.dumps(insert_data)
    storage = StringIO()
    c = pycurl.Curl()
    if url.startswith('https'):
        c.setopt(pycurl.SSL_VERIFYPEER, 0)
        c.setopt(pycurl.SSL_VERIFYHOST, 0)
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


class Batcher(object):
    def __init__(self, data_path):
        """Batcher constructor.

    Args:
      data_path: data path.
    """
        self._data_path = data_path
        self._id2name = {}
        self._extract_id_name()

        self._done = 0
        self._condition = Condition()

        self._input_queue = Queue.Queue(QUEUE_NUM_BATCH)

        self._input_threads = Thread(target=self._FillInputQueue)
        self._input_threads.daemon = True
        self._input_threads.start()

        self._output_threads = []
        for _ in xrange(MAX_OUTPUT_THREAD):
            self._output_threads.append(Thread(target=self._FillOuputQueue))
            self._output_threads[-1].daemon = True
            self._output_threads[-1].start()

    def _inc_done(self, res):
        with self._condition:
            self._done += 1
            print('process: %d, return: %d' % (self._done, res))

    def _extract_id_name(self):
        with open(self._data_path, 'rb') as f:
            row_num = 1
            for line in f.readlines():
                row_num += 1
                line = line.replace('\"', '').decode('utf-8')
                row = line[:line.rindex('.')].strip().split('\t')
                entity_with_relation = ENTITY_WITH_ID.findall(row[1])
                if len(row) == 2 and entity_with_relation[0][0] == 'property':
                    continue
                assert (len(row) == 3), 'row must be 3'
                if entity_with_relation[0][0] == 'property':
                    entity_with_id = ENTITY_WITH_ID.findall(row[0])
                    if entity_with_id[0][1] not in self._id2name:
                        self._id2name[entity_with_id[0][1]] = row[2].strip()
            print('total entity names:%d' % len(self._id2name))

    def _process_row(self, line, table_name):
        line = line.replace('\"', '').decode('utf-8')
        row = line[:line.rindex('.')].strip().split('\t')
        entity_with_relation = ENTITY_WITH_ID.findall(row[1])
        if len(row) == 2 and entity_with_relation[0][0] == 'property':
            return
        assert (len(row) == 3)
        entity_with_id = ENTITY_WITH_ID.findall(row[0])
        entity_id1 = entity_with_id[0][1]
        entity_type1 = entity_with_id[0][0]
        relation_or_property = entity_with_relation[0][1]
        if table_name == 'property':
            if entity_with_relation[0][0] == 'property':
                property_data = (entity_id1, entity_type1,
                                 relation_or_property, row[2])
                self._input_queue.put([table_name, property_data])
        elif table_name == 'entity_relation':
            if entity_with_relation[0][0] != 'property':
                entity_with_id2 = ENTITY_WITH_ID.findall(row[2])
                entity_id2 = entity_with_id2[0][1]
                entity_type2 = entity_with_id2[0][1]
                if entity_id1 not in self._id2name or entity_id2 not in self._id2name:
                    return
                relation_data1 = (entity_id1, self._id2name[entity_id1],
                                  entity_type1)
                relation_data2 = (entity_id2, self._id2name[entity_id2],
                                  entity_type2)
                relation_data = relation_data1 + (relation_or_property,
                                                  ) + relation_data2
                self._input_queue.put([table_name, relation_data])

    def _insert2property(self, property_data):
        col_name = [
            'entity_id', 'entity_type', 'property_name', 'property_value'
        ]
        res = insert_record('property', col_name, property_data)
        self._inc_done(res)

    def _insert2relation(self, relation_data):
        col_name = [
            'entity_id1', 'entity_name1', 'entity_type1', 'relation',
            'entity_id2', 'entity_name2', 'entity_type2'
        ]
        res = insert_record('relation', col_name, relation_data)
        self._inc_done(res)

    def _FillInputQueue(self):
        """Fill input queue with ModelInput."""
        with open(self._data_path, 'rb') as f:
            for line in f.readlines():
                self._process_row(line, 'property')
            f.seek(0)
            for line in f.readlines():
                self._process_row(line, 'entity_relation')

        while True:
            self._input_queue.put([])

    def _FillOuputQueue(self):
        """Fill bucketed batches into the bucket_input_queue."""
        while True:
            try:
                inputs = self._input_queue.get()
            except KeyError:
                # currently no urls to process
                break
            else:
                if inputs[0] == 'property':
                    self._insert2property(inputs[1])
                elif inputs[0] == 'entity_relation':
                    self._insert2relation(inputs[1])

    def processing(self):
        while self._output_threads or self._input_queue:
            for t in self._output_threads:
                if not t.is_alive():
                    self._output_threads.remove(t)

            while len(self._output_threads
                      ) < MAX_OUTPUT_THREAD and self._input_queue:
                # can start some more threads
                new_t = Thread(target=self._FillInputQueue)
                new_t.setDaemon(
                    True
                )  # set daemon so main thread can exit when receives ctrl-c
                new_t.start()
                self._output_threads.append(new_t)
            time.sleep(SLEEP_TIME)


def main(argc, argv):
    if argc < 2:
        print("Usage:%s <data>" % (argv[0]))
    batch_insert = Batcher(argv[1])
    batch_insert.processing()


if __name__ == "__main__":
    # pdb.set_trace()
    main(len(sys.argv), sys.argv)

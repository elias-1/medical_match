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
import time
from StringIO import StringIO
from threading import Thread

import pycurl
from six.moves import queue as Queue
from six.moves import xrange

QUEUE_NUM_BATCH = 1000

ENTITY_WITH_ID = re.compile('edu\/(.*?)\/(.*?)>', re.IGNORECASE)
NO_NAME = 0
COUNT = 0

useless_list = [
    "", u"", None, False, " ", "  ", "   ", "    ", "     ", "-", "--", "---"
]


def insert_record(table_name, col, row):
    url = 'https://202.117.16.221:7777/qa/%s/' % table_name
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
    def __init__(
            self,
            data_path, ):
        """Batcher constructor.

    Args:
      data_path: data path.
    """
        self._data_path = data_path
        self._input_queue = Queue.Queue(QUEUE_NUM_BATCH)
        self._input_threads = []
        for _ in xrange(1):
            self._input_threads.append(Thread(target=self._FillInputQueue))
            self._input_threads[-1].daemon = True
            self._input_threads[-1].start()
        self._output_threads = []
        for _ in xrange(4):
            self._output_threads.append(Thread(target=self._FillOuputQueue))
            self._output_threads[-1].daemon = True
            self._output_threads[-1].start()

        self._watch_thread = Thread(target=self._WatchThreads)
        self._watch_thread.daemon = True
        self._watch_thread.start()

    def _FillInputQueue(self):
        """Fill input queue with ModelInput."""

    def _FillOuputQueue(self):
        """Fill bucketed batches into the bucket_input_queue."""
        while True:
            inputs = []
            for _ in xrange(self._hps.batch_size * BUCKET_CACHE_BATCH):
                inputs.append(self._input_queue.get())
            if self._bucketing:
                inputs = sorted(inputs, key=lambda inp: inp.enc_len)

            batches = []
            for i in xrange(0, len(inputs), self._hps.batch_size):
                batches.append(inputs[i:i + self._hps.batch_size])
            shuffle(batches)
            for b in batches:
                self._bucket_input_queue.put(b)

    def _WatchThreads(self):
        """Watch the daemon input threads and restart if dead."""
        while True:
            time.sleep(60)
            input_threads = []
            for t in self._input_threads:
                if t.is_alive():
                    input_threads.append(t)
                else:
                    print('Found input thread dead.')
                    new_t = Thread(target=self._FillInputQueue)
                    input_threads.append(new_t)
                    input_threads[-1].daemon = True
                    input_threads[-1].start()
            self._input_threads = input_threads

            output_threads = []
            for t in self._output_threads:
                if t.is_alive():
                    output_threads.append(t)
                else:
                    print('Found bucketing thread dead.')
                    new_t = Thread(target=self._FillOuputQueue)
                    output_threads.append(new_t)
                    output_threads[-1].daemon = True
                    output_threads[-1].start()
            self._output_threads = output_threads

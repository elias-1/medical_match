#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 www.drcubic.com, Inc. All Rights Reserved
#
"""
File: clfier_client.py
Author: shileicao(shileicao@stu.xjtu.edu.cn)
Date: 2017/3/26 14:34

A serving_client that talks to tensorflow_model_server loaded with clfier model.

Typical usage example:
    clfier_client.py --num_tests=100 --server=localhost:9000
"""

from __future__ import print_function

import sys
import threading

import numpy as np
import tensorflow as tf
from cnn_clfier import C_MAX_SENTENCE_LEN, C_MAX_WORD_LEN, do_load_data
from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2

# This is a placeholder for a Google-internal import.

tf.app.flags.DEFINE_integer('concurrency', 1,
                            'maximum number of concurrent inference requests')
tf.app.flags.DEFINE_string('server', '', 'PredictionService host:port')
FLAGS = tf.app.flags.FLAGS


class _ResultCounter(object):
    """Counter for the prediction results."""

    def __init__(self, num_tests, concurrency):
        self._num_tests = num_tests
        self._concurrency = concurrency
        self._error = 0
        self._done = 0
        self._active = 0
        self._condition = threading.Condition()

    def inc_error(self):
        with self._condition:
            self._error += 1

    def inc_done(self):
        with self._condition:
            self._done += 1
            self._condition.notify()

    def dec_active(self):
        with self._condition:
            self._active -= 1
            self._condition.notify()

    def get_error_rate(self):
        with self._condition:
            while self._done != self._num_tests:
                self._condition.wait()
            return self._error / float(self._num_tests)

    def throttle(self):
        with self._condition:
            while self._active == self._concurrency:
                self._condition.wait()
            self._active += 1


def _create_rpc_callback(label, result_counter):
    """Creates RPC callback function.
  Args:
    label: The correct label for the predicted example.
    result_counter: Counter for the prediction result.
  Returns:
    The callback function.
  """

    def _callback(result_future):
        """Callback function.
    Calculates the statistics for the prediction result.
    Args:
      result_future: Result future of the RPC.
    """
        exception = result_future.exception()
        if exception:
            result_counter.inc_error()
            print(exception)
        else:
            sys.stdout.write('.')
            sys.stdout.flush()
            response = result_future.result().outputs['classes'].string_val
            prediction = int(response[0])
            if label != prediction:
                result_counter.inc_error()
        result_counter.inc_done()
        result_counter.dec_active()

    return _callback


def do_inference(hostport, concurrency):
    """Tests PredictionService with concurrent requests.
  Args:
    hostport: Host:port address of the PredictionService.
    concurrency: Maximum number of concurrent requests.
  Returns:
    The classification error rate.
  Raises:
    IOError: An error occurred processing test data set.
  """

    clfier_tX, clfier_tcX, clfier_tY = do_load_data(
        FLAGS.test_data_path, FLAGS.max_sentence_len, FLAGS.max_chars_per_word)
    num_tests = clfier_tX.shape[0]
    host, port = hostport.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    result_counter = _ResultCounter(num_tests, concurrency)
    for i in xrange(num_tests):
        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'clfier'
        request.model_spec.signature_name = 'predict_sentence'
        label = clfier_tY[i]
        request.inputs['words'].CopyFrom(
            tf.contrib.util.make_tensor_proto(
                clfier_tX[i], shape=[1, C_MAX_SENTENCE_LEN]))
        request.inputs['chars'].CopyFrom(
            tf.contrib.util.make_tensor_proto(
                clfier_tcX[i], shape=[1, C_MAX_SENTENCE_LEN * C_MAX_WORD_LEN]))
        result_counter.throttle()
        result_future = stub.Predict.future(request, 5.0)  # 5 seconds
        result_future.add_done_callback(
            _create_rpc_callback(label, result_counter))
    return result_counter.get_error_rate()


def main(_):

    if not FLAGS.server:
        print('please specify server host:port')
        return
    error_rate = do_inference(FLAGS.server, FLAGS.concurrency)
    print('\nInference error rate: %s%%' % (error_rate * 100))


if __name__ == '__main__':
    tf.app.run()

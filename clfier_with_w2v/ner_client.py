#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 www.drcubic.com, Inc. All Rights Reserved
#
"""
File: ner_client.py
Author: shileicao(shileicao@stu.xjtu.edu.cn)
Date: 2017/3/27 20:52
"""

from __future__ import print_function

import sys
import threading
import time

import numpy as np
import tensorflow as tf
from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2
from train_ner import FLAGS, NER_MAX_SENTENCE_LEN, Model, do_load_data

# This is a placeholder for a Google-internal import.

tf.app.flags.DEFINE_integer('concurrency', 1,
                            'maximum number of concurrent inference requests')
tf.app.flags.DEFINE_string('server', '', 'PredictionService host:port')
FLAGS = tf.app.flags.FLAGS

trainsMatrix = np.load('transition.npy')


def viterbi_decode(score, transition_params):
    """Decode the highest scoring sequence of tags outside of TensorFlow.
  This should only be used at test time.
  Args:
    score: A [seq_len, num_tags] matrix of unary potentials.
    transition_params: A [num_tags, num_tags] matrix of binary potentials.
  Returns:
    viterbi: A [seq_len] list of integers containing the highest scoring tag
        indicies.
    viterbi_score: A float containing the score for the Viterbi sequence.
  """
    trellis = np.zeros_like(score)
    backpointers = np.zeros_like(score, dtype=np.int32)
    trellis[0] = score[0]

    for t in range(1, score.shape[0]):
        v = np.expand_dims(trellis[t - 1], 1) + transition_params
        trellis[t] = score[t] + np.max(v, 0)
        backpointers[t] = np.argmax(v, 0)

    viterbi = [np.argmax(trellis[-1])]
    for bp in reversed(backpointers[1:]):
        viterbi.append(bp[viterbi[-1]])
    viterbi.reverse()

    viterbi_score = np.max(trellis[-1])
    return viterbi, viterbi_score


def decode_entity_location(self, entity_info):
    entity_location = []
    types_id = []
    last_type_id = -1
    loc = -1
    begin = -1
    for word_tag in entity_info:
        loc += 1
        if word_tag < 2:
            last_type_id = -1
            begin = -1
            continue
        type_id = (word_tag - 2) / 4
        entity_tag = (word_tag - 2) % 4
        if entity_tag == 0:
            entity_location.append((loc, loc))
            types_id.append(type_id)
            last_type_id = type_id
            begin = -1
        elif type_id != last_type_id:
            if begin != -1:
                entity_location.append((begin, loc - 1))
                types_id.append(last_type_id)
                last_type_id = type_id
                begin = -1
            else:
                last_type_id = type_id
                begin = loc
        else:
            if begin != -1:
                if entity_tag == 3:
                    entity_location.append((begin, loc))
                    types_id.append(type_id)
                    last_type_id = type_id
                    begin = -1
                else:
                    last_type_id = type_id
            else:
                last_type_id = type_id
                begin = loc
    return entity_location, types_id


class _ResultCounter(object):
    """Counter for the prediction results."""

    def __init__(self, num_tests, concurrency):
        self._num_tests = num_tests
        self._concurrency = concurrency
        self._error = 0
        self._total = 0
        self._done = 0
        self._active = 0
        self._condition = threading.Condition()

    def inc_error_and_total(self, error_num, total_num):
        with self._condition:
            self._error += error_num
            self._total += total_num

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
            return self._error / float(self._total)

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
            result_counter.inc_error_and_total(0, 1)
            print(exception)
        else:
            sys.stdout.write('.')
            sys.stdout.flush()
            result = result_future.result()
            unary_score = np.array(result.outputs['scores'].float_val)

            # shape = list(result.outputs['scores'].tensor_shape.dim)
            # shape = [int(dim[0]) for dim in shape]
            # print(shape)

            unary_score = np.reshape(unary_score, (NER_MAX_SENTENCE_LEN, -1))
            seq_len = int(result.outputs['sequence_length'].int_val[0])

            tf_unary_scores_ = unary_score[:seq_len]

            viterbi_sequence, _ = viterbi_decode(tf_unary_scores_,
                                                 trainsMatrix)

            correct_labels = np.sum(
                np.equal(viterbi_sequence, label[:seq_len]))

            result_counter.inc_error_and_total(seq_len - correct_labels,
                                               seq_len)
            # entity_location, types_id = decode_entity_location(
            #     viterbi_sequence)
            # entity_result = []
            # for loc, type_id in zip(entity_location, types_id):
            #     entity = sentence[loc[0]:loc[1] + 1]
            #     entity = entity.replace(',', '，')
            #     entities = entity.strip('，').split('，')
            #     entity_result.extend(entities)

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

    ner_tX, ner_tY = do_load_data(FLAGS.ner_test_data_path)
    num_tests = ner_tX.shape[0]
    host, port = hostport.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    result_counter = _ResultCounter(num_tests, concurrency)
    for i in xrange(num_tests):
        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'ner'
        request.model_spec.signature_name = 'predict_sentence'
        label = ner_tY[i]
        request.inputs['words'].CopyFrom(
            tf.contrib.util.make_tensor_proto(
                ner_tX[i], shape=[1, NER_MAX_SENTENCE_LEN]))

        result_counter.throttle()
        result_future = stub.Predict.future(request, 5.0)  # 5 seconds
        result_future.add_done_callback(
            _create_rpc_callback(label, result_counter))
    return result_counter.get_error_rate()


def main(_):
    start = time.clock()
    if not FLAGS.server:
        print('please specify server host:port')
        return
    error_rate = do_inference(FLAGS.server, FLAGS.concurrency)
    print('\nInference error rate: %s%%' % (error_rate * 100))
    end = time.clock()


if __name__ == '__main__':
    tf.app.run()

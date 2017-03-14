#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 www.drcubic.com, Inc. All Rights Reserved
#
"""
File: ner.py
Author: shileicao(shileicao@stu.xjtu.edu.cn)
Date: 17-3-14 下午3:05
"""
import os
import sys

import jieba
import numpy as np
import tensorflow as tf

from .train_ner import FLAGS, Model, do_load_data, test_evaluate
from .utils import ENTITY_TYPES

reload(sys)
sys.setdefaultencoding('utf8')
# Eval Parameters
MAX_SENTENCE_LEN = 80
tf.flags.DEFINE_string("run_dir", "ner_logs_v2",
                       "Dir of training run using for ckpt")

tf.flags.DEFINE_string("exec_dir", ".", "execute env dir")
UNK = '<UNK>'


class Ner:
    def __init__(self):
        #         graph = tf.Graph()
        #         self.sess = tf.Session(graph=graph)
        self.sess = tf.Session()

        run_dir = os.path.join(FLAGS.exec_dir, FLAGS.run_dir)
        char2vec_path = os.path.join(FLAGS.exec_dir, FLAGS.word2vec_path)
        self.char_vob = self.get_vob(char2vec_path)
        self.model = Model(char2vec_path)
        checkpoint_file = tf.train.latest_checkpoint(run_dir)
        saver = tf.train.Saver()
        saver.restore(self.sess, checkpoint_file)
        self.test_unary_score, self.test_sequence_length = self.model.test_unary_score(
        )
        print(tf.global_variables())

    def get_vob(self, vob_path):
        vob = []
        with open(vob_path, 'r') as f:
            f.readline()
            for row in f.readlines():
                vob.append(row.split()[0].decode('utf-8'))
        return vob

    def process_line(self, x_text):
        nl = len(x_text)
        chari = []
        if nl > MAX_SENTENCE_LEN:
            nl = MAX_SENTENCE_LEN
        for ti in range(nl):
            char = x_text[ti]
            idx = self.char_vob.GetWordIndex(char)
            chari.append(str(idx))
        for i in range(nl, MAX_SENTENCE_LEN):
            chari.append("0")

        return chari

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

    def __call__(self, sentence):
        chari = self.process_line(sentence)
        chari = map(int, chari)

        feed_dict = {self.model.inp_w: np.array([chari]), }
        unary_score_val, test_sequence_length_val = self.sess.run(
            [self.test_unary_score, self.test_sequence_length], feed_dict)

        tf_unary_scores_ = unary_score_val[0][:test_sequence_length_val[0]]

        viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(
            tf_unary_scores_, self.model.transition_params)

        entity_location, types_id = self.decode_entity_location(
            viterbi_sequence)
        entity_with_types = []
        for loc, type_id in zip(entity_location, types_id):
            entity = sentence[loc[0]:loc[1] + 1]
            type = ENTITY_TYPES[type_id]
            entity_with_types.append(entity + '/' + type)

        print('  ||  '.join(entity_with_types))
        return entity_with_types

    def test(self):
        twX, tY = do_load_data(FLAGS.test_data_path)

        test_evaluate(self.sess, self.test_unary_score,
                      self.test_sequence_length, self.model.transition_params,
                      self.model.inp_w, twX, tY)


def main(argv=None):
    ner = Ner()
    # print sentence_clfier(u'得了糖尿病，该吃什么药')
    ner.test()


if __name__ == '__main__':
    tf.app.run()

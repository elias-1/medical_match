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

import numpy as np
import tensorflow as tf

from train_ner import (FLAGS, NER_MAX_SENTENCE_LEN, Model, do_load_data,
                       test_evaluate)

ENTITY_TYPES = ['@d@', '@s@', '@l@', '@o@', '@m@', '@dp@', '@bp@']

reload(sys)
sys.setdefaultencoding('utf8')
# Eval Parameters
tf.flags.DEFINE_string("ner_run_dir", "ner_logs_v2/1489569777",
                       "Dir of training run using for ckpt")

tf.flags.DEFINE_string("ner_exec_dir",
                       "/home/elias/code/medical_match/clfier_with_w2v",
                       "execute env dir")
UNK = '<UNK>'


class Ner:
    def __init__(self):
        self.sess = tf.Session()

        char2vec_path = os.path.join(FLAGS.ner_exec_dir,
                                     FLAGS.ner_word2vec_path)
        self.char_vob = self.get_vob(char2vec_path)

        self.model = Model(FLAGS.ner_num_tags, char2vec_path,
                           FLAGS.ner_num_hidden)
        self.test_unary_score, self.test_sequence_length = self.model.inference(
            self.model.inp_w, trainMode=False)

        ner_run_dir = os.path.join(FLAGS.ner_exec_dir, FLAGS.ner_run_dir)
        checkpoint_file = tf.train.latest_checkpoint(ner_run_dir)

        tf.reset_default_graph()
        with self.sess.graph.as_default():
            saver = tf.train.Saver()
            saver.restore(self.sess, checkpoint_file)

        # saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        # saver.restore(self.sess, checkpoint_file)

        # vnames = [v.name for v in tf.global_variables()]
        # print vnames

        # self.inp_w = self.sess.graph.get_operation_by_name(
        #     "input_words").outputs[0]
        #
        # self.test_sequence_length = self.sess.graph.get_operation_by_name(
        #     "length").outputs[0]
        #
        # self.test_unary_score = self.sess.graph.get_operation_by_name(
        #     "unary_scores").outputs[0]
        transition_fpath = os.path.join(FLAGS.ner_exec_dir, 'transition.npy')
        self.trainsMatrix = np.load(transition_fpath)

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
        if nl > NER_MAX_SENTENCE_LEN:
            nl = NER_MAX_SENTENCE_LEN
        for ti in range(nl):
            char = x_text[ti]
            try:
                idx = self.char_vob.index(char)
            except ValueError:
                idx = self.char_vob.index(UNK)
            chari.append(str(idx))
        for i in range(nl, NER_MAX_SENTENCE_LEN):
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

        tf.reset_default_graph()
        with self.sess.graph.as_default():
            feed_dict = {
                self.model.inp_w: np.array([chari]),
            }
            unary_score_val, test_sequence_length_val = self.sess.run(
                [self.test_unary_score, self.test_sequence_length], feed_dict)

            tf_unary_scores_ = unary_score_val[0][:test_sequence_length_val[0]]

            viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(
                tf_unary_scores_, self.trainsMatrix)

            entity_location, types_id = self.decode_entity_location(
                viterbi_sequence)
            entity_with_types = []
            for loc, type_id in zip(entity_location, types_id):
                entity = sentence[loc[0]:loc[1] + 1]
                #type = ENTITY_TYPES[type_id]
                #entity_with_type = entity + '/' + type
                #entity_with_types.append(entity_with_type.encode('utf-8'))
                entity = entity.replace(',', '，')
                ens = entity.strip('，').split('，')
                entity_with_types.extend(ens)

            #print('  ||  '.join(entity_with_types))
            return entity_with_types

    def test(self):
        tf.reset_default_graph()
        with self.sess.graph.as_default():
            twX, tY = do_load_data(FLAGS.ner_test_data_path)

            test_evaluate(self.sess, self.test_unary_score,
                          self.test_sequence_length, self.trainsMatrix,
                          self.model.inp_w, twX, tY)


def main(argv=None):
    # import pdb
    # pdb.set_trace()
    ner = Ner()
    # print ner(u'得了糖尿病，该吃什么药')
    print ''.join(ner(u'今天天气怎么样？'))
    print ''.join(ner(u'下雨了怎么办？'))
    # ner.test()


if __name__ == '__main__':
    tf.app.run()

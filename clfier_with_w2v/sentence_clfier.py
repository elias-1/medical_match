#!/usr/bin/env python
# -*- coding: utf-8 -*-

########################################################################
#
# Copyright (c) 2016 www.drcubic.com, Inc. All Rights Reserved
#
########################################################################
"""
File: sentence_clfier.py
Author: shileicao(shileicao@stu.xjtu.edu.cn)
Date: 2016/12/28 16:49:22
"""

import sys

import numpy as np
import tensorflow as tf
import w2v
from cnn_clfier import FLAGS, TextCNN
from prepare_data import process_line

reload(sys)
sys.setdefaultencoding('utf8')

# Eval Parameters
tf.flags.DEFINE_string("run_dir", "cnn_clfier_logs/1488122912",
                       "Dir of training run")


class SentenceClfier:
    def __init__(self):
        graph = tf.Graph()
        self.sess = tf.Session(graph=graph)
        self.model = TextCNN(FLAGS.word2vec_path, FLAGS.char2vec_path)
        checkpoint_file = tf.train.latest_checkpoint(FLAGS.run_dir)
        saver = tf.train.Saver()
        saver.restore(self.sess, checkpoint_file)
        self.word_vob = w2v.Word2vecVocab()
        self.word_vob.Load(FLAGS.word2vec_path)
        self.char_vob = w2v.Word2vecVocab()
        self.char_vob.Load(FLAGS.char2vec_path)

        self.test_clfier_score = self.model.test_clfier_score()

    def __call__(self, sentence):
        wordi, chari = process_line(sentence, self.word_vob, self.char_vob)
        wordi = map(int, wordi)
        chari = map(int, chari)

        feed_dict = {
            self.model.inp_w: [np.array(wordi)],
            self.model.inp_c: [np.array(chari)]
        }
        clfier_score_val = self.sess.run([self.test_clfier_score], feed_dict)
        predictions = np.argmax(clfier_score_val[0], 1)

        return predictions[0][0] + 1


def main(argv=None):
    sentence_clfier = SentenceClfier()
    print sentence_clfier(u'得了糖尿病应该吃什么药？')


if __name__ == '__main__':
    tf.app.run()

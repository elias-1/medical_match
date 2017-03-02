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

import os
import sys

import jieba
import numpy as np
import tensorflow as tf
from cnn_clfier import FLAGS, TextCNN, do_load_data, test_evaluate

reload(sys)
sys.setdefaultencoding('utf8')
MAX_SENTENCE_LEN = 30
MAX_WORD_LEN = 6

# Eval Parameters
tf.flags.DEFINE_string("run_dir", "cnn_clfier_logs/1488196736",
                       "Dir of training run using for ckpt")

tf.flags.DEFINE_string("exec_dir", ".", "execute env dir")
UNK = '<UNK>'


def tokenizer(sentence):
    return jieba.lcut(sentence, cut_all=False)


class SentenceClfier:
    def __init__(self):

        word2vec_path = os.path.join(FLAGS.exec_dir, FLAGS.word2vec_path)
        char2vec_path = os.path.join(FLAGS.exec_dir, FLAGS.char2vec_path)
        run_dir = os.path.join(FLAGS.exec_dir, FLAGS.run_dir)

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.model = TextCNN(word2vec_path, char2vec_path)
            self.test_clfier_score = self.model.test_clfier_score()

            sv = tf.train.Supervisor(graph=self.graph, logdir=run_dir)

            with sv.managed_session(master='') as sess:

                checkpoint_file = tf.train.latest_checkpoint(run_dir)
                saver = tf.train.Saver()
                saver.restore(sess, checkpoint_file)

                clfier_tX, clfier_tcX, clfier_tY = do_load_data(
                    FLAGS.test_data_path, FLAGS.max_sentence_len,
                    FLAGS.max_chars_per_word)
                test_evaluate(sess, self.test_clfier_score, self.model.inp_w,
                              self.model.inp_c, clfier_tX, clfier_tcX,
                              clfier_tY)

                self.word_vob = self.get_vob(word2vec_path)
                self.char_vob = self.get_vob(char2vec_path)

    def get_vob(self, vob_path):
        vob = []
        with open(vob_path, 'r') as f:
            f.readline()
            for row in f.readlines():
                vob.append(row.split()[0].decode('utf-8'))
        return vob

    def process_line(self, x_text):
        words = tokenizer(x_text)
        nl = len(words)
        wordi = []
        chari = []
        if nl > MAX_SENTENCE_LEN:
            nl = MAX_SENTENCE_LEN
        for ti in range(nl):
            word = words[ti]
            try:
                word_idx = self.word_vob.index(word)
            except ValueError:
                word_idx = self.word_vob.index(UNK)

            wordi.append(str(word_idx))
            chars = list(word)
            nc = len(chars)
            if nc > MAX_WORD_LEN:
                lc = chars[nc - 1]
                chars[MAX_WORD_LEN - 1] = lc
                nc = MAX_WORD_LEN
            for i in range(nc):
                try:
                    char_idx = self.char_vob.index(chars[i])
                except ValueError:
                    char_idx = self.char_vob.index(UNK)
                chari.append(str(char_idx))
            for i in range(nc, MAX_WORD_LEN):
                chari.append("0")
        for i in range(nl, MAX_SENTENCE_LEN):
            wordi.append("0")
            for ii in range(MAX_WORD_LEN):
                chari.append('0')
        return wordi, chari

    def __call__(self, sentence):
        wordi, chari = self.process_line(sentence)
        wordi = map(int, wordi)
        chari = map(int, chari)

        feed_dict = {
            self.model.inp_w: np.array([wordi]),
            self.model.inp_c: np.array([chari])
        }
        clfier_score_val = self.sess.run([self.test_clfier_score], feed_dict)
        predictions = np.argmax(clfier_score_val[0], 1)

        return predictions[0] + 1

    def test(self):
        with self.graph.as_default():
            clfier_tX, clfier_tcX, clfier_tY = do_load_data(
                FLAGS.test_data_path, FLAGS.max_sentence_len,
                FLAGS.max_chars_per_word)
            test_evaluate(self.sess, self.test_clfier_score, self.model.inp_w,
                          self.model.inp_c, clfier_tX, clfier_tcX, clfier_tY)


def main(argv=None):
    sentence_clfier = SentenceClfier()
    # print sentence_clfier(u'得了糖尿病，该吃什么药')
    # sentence_clfier.test()


if __name__ == '__main__':
    tf.app.run()

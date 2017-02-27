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

import jieba
import numpy as np
import tensorflow as tf
from cnn_clfier import FLAGS, TextCNN

reload(sys)
sys.setdefaultencoding('utf8')
MAX_SENTENCE_LEN = 30
MAX_WORD_LEN = 6

# Eval Parameters
tf.flags.DEFINE_string("run_dir", "cnn_clfier_logs/1488196736",
                       "Dir of training run")

UNK = '<UNK>'


def tokenizer(sentence):
    return jieba.lcut(sentence, cut_all=False)


class SentenceClfier:
    def __init__(self):
        #         graph = tf.Graph()
        #         self.sess = tf.Session(graph=graph)
        self.sess = tf.Session()
        self.model = TextCNN(FLAGS.word2vec_path, FLAGS.char2vec_path)
        checkpoint_file = tf.train.latest_checkpoint(FLAGS.run_dir)
        saver = tf.train.Saver()
        saver.restore(self.sess, checkpoint_file)
        self.word_vob = self.get_vob(FLAGS.word2vec_path)
        self.char_vob = self.get_vob(FLAGS.char2vec_path)

        self.test_clfier_score = self.model.test_clfier_score()

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


def main(argv=None):
    sentence_clfier = SentenceClfier()
    print sentence_clfier(u'得了糖尿病，该吃什么药')


if __name__ == '__main__':
    tf.app.run()

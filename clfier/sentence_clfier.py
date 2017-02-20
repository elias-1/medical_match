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

import json
import os
import sys

import jieba
import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn

reload(sys)
sys.setdefaultencoding('utf8')

# Data Parameters
tf.flags.DEFINE_string("data", "data", "Data source for the data.")

# Eval Parameters
tf.flags.DEFINE_string("run_dir", "clfier/runs/1487590765",
                       "Dir of training run")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True,
                        "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False,
                        "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

# In[ ]:

jieba.load_userdict(os.path.join(FLAGS.data, 'words.txt'))


def tokenizer(iterator):
    """Tokenizer generator.
    Args:
        iterator: Input iterator with strings.
    Yields:
        array of tokens per each value in the input.
    """
    for value in iterator:
        yield jieba.lcut(value, cut_all=False)


def sentence_clfier(sentence):

    # Map data into vocabulary
    vocab_path = os.path.join(FLAGS.run_dir, "vocab")
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(
        vocab_path)
    x_test = np.array(list(vocab_processor.transform([sentence])))

    # Evaluation
    # ==================================================
    checkpoint_file = tf.train.latest_checkpoint(os.path.join(FLAGS.run_dir,
                                                              'checkpoints'))
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(
                checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]

            dropout_keep_prob = graph.get_operation_by_name(
                "dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            prediction_op = graph.get_operation_by_name(
                "output/predictions").outputs[0]

            feed_dict = {input_x: x_test, dropout_keep_prob: 1.0}
            predictions = sess.run([prediction_op], feed_dict)

            label_info_dir = os.path.join(FLAGS.run_dir, 'label_to_int.json')
            with open(label_info_dir, 'r') as f:
                label_to_int = json.load(f)

            int_to_label = {
                label_to_int[key]: key
                for key in label_to_int.keys()
            }
            return int_to_label[predictions[0][0]]


def main(argv=None):
    sentence_clfier(u'得了糖尿病应该吃什么药？')


if __name__ == '__main__':
    tf.app.run()

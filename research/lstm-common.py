#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2016 www.drcubic.com, Inc. All Rights Reserved
#
"""
File: lstm.py
Author: shileicao(shileicao@stu.xjtu.edu.cn)
Date: 2017-5-18 09:54:54
"""

from __future__ import absolute_import, division, print_function

import os

import numpy as np
import tensorflow as tf
from utils import (ENTITY_TYPES, MAX_SENTENCE_LEN2, MAX_WORD_LEN, do_load_data,
                   load_w2v)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_data_path', "clfier_common_train.txt",
                           'Training data dir')
tf.app.flags.DEFINE_string('test_data_path', "clfier_common_test.txt",
                           'Test data dir')
tf.app.flags.DEFINE_string('clfier_log_dir', "lstm_common_logs",
                           'The log  dir')
tf.app.flags.DEFINE_string("word2vec_path",
                           "../clfier_with_w2v/words_vec_100.txt",
                           "the word2vec data path")
tf.app.flags.DEFINE_string("char2vec_path",
                           "../clfier_with_w2v/chars_vec_50.txt",
                           "the char2vec data path")
tf.app.flags.DEFINE_integer("max_sentence_len", MAX_SENTENCE_LEN2,
                            "max num of tokens per query")
tf.app.flags.DEFINE_integer("embedding_word_size", 100, "embedding size")
tf.app.flags.DEFINE_integer("embedding_char_size", 50, "second embedding size")
tf.app.flags.DEFINE_integer("char_window_size", 2,
                            "the window size of char convolution")
tf.app.flags.DEFINE_integer("max_chars_per_word", MAX_WORD_LEN,
                            "max number of characters per word ")
tf.app.flags.DEFINE_integer("num_hidden", 100, "hidden unit number")
tf.app.flags.DEFINE_integer("batch_size", 64, "num example per mini batch")
tf.app.flags.DEFINE_integer("train_steps", 2000, "trainning steps")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "learning rate")

tf.app.flags.DEFINE_float("filter_size", 5, "The clfier's conv fileter size")
tf.app.flags.DEFINE_float("num_filters", 128, "Number of filters")
tf.app.flags.DEFINE_float("num_classes", 7, "Number of classes to classify")
tf.app.flags.DEFINE_float('dropout_keep_prob', 0.5,
                          'Dropout keep probability (default: 0.5)')

tf.flags.DEFINE_float('l2_reg_lambda', 0,
                      'L2 regularization lambda (default: 0.0)')


class Model:
    def __init__(self, w2vPath, c2vPath, numHidden):
        self.numHidden = numHidden
        self.w2v = load_w2v(w2vPath, FLAGS.embedding_word_size)
        self.words = tf.Variable(self.w2v, name="words")
        self.common_id_embedding = tf.Variable(
            tf.random_uniform([len(ENTITY_TYPES), FLAGS.embedding_word_size],
                              -1.0, 1.0),
            name="common_id_embedding")
        self.words_emb = tf.concat(
            [self.words, self.common_id_embedding], 0, name='concat')
        self.c2v = load_w2v(c2vPath, FLAGS.embedding_char_size)
        self.chars = tf.Variable(self.c2v, name="chars")

        with tf.variable_scope('CNN_Layer') as scope:
            self.char_filter = tf.get_variable(
                "char_filter",
                shape=[
                    FLAGS.char_window_size, FLAGS.embedding_char_size, 1,
                    FLAGS.embedding_char_size
                ],
                regularizer=tf.contrib.layers.l2_regularizer(0.0001),
                initializer=tf.truncated_normal_initializer(stddev=0.01),
                dtype=tf.float32)

        with tf.variable_scope('Clfier_output') as scope:
            self.clfier_softmax_W = tf.get_variable(
                "clfier_W",
                shape=[numHidden * 2, FLAGS.num_classes],
                regularizer=tf.contrib.layers.l2_regularizer(0.0001),
                initializer=tf.truncated_normal_initializer(stddev=0.01),
                dtype=tf.float32)

            self.clfier_softmax_b = tf.get_variable(
                "clfier_softmax_b",
                shape=[FLAGS.num_classes],
                regularizer=tf.contrib.layers.l2_regularizer(0.0001),
                initializer=tf.truncated_normal_initializer(stddev=0.01),
                dtype=tf.float32)

        self.inp_w = tf.placeholder(
            tf.int32, shape=[None, FLAGS.max_sentence_len], name="input_words")
        self.inp_c = tf.placeholder(
            tf.int32,
            shape=[None, FLAGS.max_sentence_len * FLAGS.max_chars_per_word],
            name="input_chars")

    def length(self, data):
        used = tf.sign(tf.abs(data))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    def char_convolution(self, vecs):
        conv1 = tf.nn.conv2d(
            vecs,
            self.char_filter, [1, 1, FLAGS.embedding_char_size, 1],
            padding='VALID')
        conv1 = tf.nn.relu(conv1)
        pool1 = tf.nn.max_pool(
            conv1,
            ksize=[
                1, FLAGS.max_chars_per_word - FLAGS.char_window_size + 1, 1, 1
            ],
            strides=[
                1, FLAGS.max_chars_per_word - FLAGS.char_window_size + 1, 1, 1
            ],
            padding='SAME')
        pool1 = tf.squeeze(pool1, [1, 2])
        return pool1

    def inference(self, clfier_X, clfier_cX, reuse=None, trainMode=True):
        word_vectors = tf.nn.embedding_lookup(self.words, clfier_X)
        char_vectors = tf.nn.embedding_lookup(self.chars, clfier_cX)
        char_vectors = tf.reshape(char_vectors, [
            -1, FLAGS.max_sentence_len, FLAGS.embedding_char_size,
            FLAGS.max_chars_per_word
        ])
        char_vectors = tf.transpose(char_vectors, perm=[1, 0, 3, 2])
        char_vectors = tf.expand_dims(char_vectors, -1)

        length = self.length(clfier_X)
        length_64 = tf.cast(length, tf.int64)

        do_char_conv = lambda x: self.char_convolution(x)
        char_vectors_x = tf.map_fn(do_char_conv, char_vectors)
        char_vectors_x = tf.transpose(char_vectors_x, perm=[1, 0, 2])
        word_vectors = tf.concat([word_vectors, char_vectors_x], 2)

        #if trainMode:
        #  word_vectors = tf.nn.dropout(word_vectors, FLAGS.dropout_keep_prob)
        with tf.variable_scope("rnn_fwbw", reuse=reuse) as scope:
            forward_output, _ = tf.nn.dynamic_rnn(
                tf.contrib.rnn.LSTMCell(self.numHidden),
                word_vectors,
                dtype=tf.float32,
                sequence_length=length,
                scope="RNN_forward")
            backward_output_, _ = tf.nn.dynamic_rnn(
                tf.contrib.rnn.LSTMCell(self.numHidden),
                inputs=tf.reverse_sequence(word_vectors, length_64, seq_dim=1),
                dtype=tf.float32,
                sequence_length=length,
                scope="RNN_backword")

        backward_output = tf.reverse_sequence(
            backward_output_, length_64, seq_dim=1)

        output = tf.concat([forward_output, backward_output], 2)
        if trainMode:
            output = tf.nn.dropout(output, FLAGS.dropout_keep_prob)

        ds = tf.reduce_sum(output, axis=1)

        scores = tf.nn.xw_plus_b(ds, self.clfier_softmax_W,
                                 self.clfier_softmax_b)
        return scores

    def loss(self, clfier_X, clfier_cX, clfier_Y):
        self.scores = self.inference(clfier_X, clfier_cX)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.scores, labels=clfier_Y)
        loss = tf.reduce_mean(cross_entropy, name='cross_entropy')
        regularization_loss = tf.add_n(
            tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        final_loss = loss + regularization_loss * FLAGS.l2_reg_lambda
        return final_loss

    def test_clfier_score(self):
        scores = self.inference(
            self.inp_w, self.inp_c, reuse=True, trainMode=False)
        return scores


def read_csv(batch_size, file_name):
    filename_queue = tf.train.string_input_producer([file_name])
    reader = tf.TextLineReader(skip_header_lines=0)
    key, value = reader.read(filename_queue)
    # decode_csv will convert a Tensor from type string (the text line) in
    # a tuple of tensor columns with the specified defaults, which also
    # sets the data type for each column
    decoded = tf.decode_csv(
        value,
        field_delim=' ',
        record_defaults=[[0]
                         for i in range(FLAGS.max_sentence_len * (
                             FLAGS.max_chars_per_word + 1) + 1)])

    # batch actually reads the file and loads "batch_size" rows in a single tensor
    return tf.train.shuffle_batch(
        decoded,
        batch_size=batch_size,
        capacity=batch_size * 4,
        min_after_dequeue=batch_size)


def inputs(path):
    whole = read_csv(FLAGS.batch_size, path)
    features = tf.transpose(tf.stack(whole[0:FLAGS.max_sentence_len]))
    char_features = tf.transpose(
        tf.stack(whole[FLAGS.max_sentence_len:(FLAGS.max_chars_per_word + 1) *
                       FLAGS.max_sentence_len]))
    len_features = FLAGS.max_sentence_len * (FLAGS.max_chars_per_word + 1)
    label = tf.transpose(tf.concat(whole[len_features:len_features + 1], 0))
    return features, char_features, label


def train(total_loss):
    return tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(total_loss)


def test_evaluate(sess, test_clfier_score, inp_w, inp_c, entity_info,
                  clfier_tX, clfier_tcX, clfier_tY, tentity_info):
    batchSize = FLAGS.batch_size
    totalLen = clfier_tX.shape[0]
    numBatch = int((totalLen - 1) / batchSize) + 1
    correct_clfier_labels = 0
    for i in range(numBatch):
        endOff = (i + 1) * batchSize
        if endOff > totalLen:
            endOff = totalLen
        y = clfier_tY[i * batchSize:endOff]
        feed_dict = {
            inp_w: clfier_tX[i * batchSize:endOff],
            inp_c: clfier_tcX[i * batchSize:endOff],
            entity_info: tentity_info[i * batchSize:endOff]
        }
        clfier_score_val = sess.run([test_clfier_score], feed_dict)
        predictions = np.argmax(clfier_score_val[0], 1)
        correct_clfier_labels += np.sum(np.equal(predictions, y))

    accuracy = 100.0 * correct_clfier_labels / float(totalLen)
    print("Accuracy: %.3f%%" % accuracy)


def main(unused_argv):
    trainDataPath = tf.app.flags.FLAGS.train_data_path
    graph = tf.Graph()
    # ner_checkpoint_file = tf.train.latest_checkpoint(FLAGS.ner_log_dir)
    with graph.as_default():
        model = Model(FLAGS.word2vec_path, FLAGS.char2vec_path,
                      FLAGS.num_hidden)
        print("train data path:", trainDataPath)
        clfier_X, clfier_cX, clfier_Y = inputs(trainDataPath)
        clfier_tX, clfier_tcX, clfier_tY = do_load_data(
            FLAGS.test_data_path, FLAGS.max_sentence_len,
            FLAGS.max_chars_per_word)
        total_loss = model.loss(clfier_X, clfier_cX, clfier_Y)
        train_op = train(total_loss)
        test_clfier_score = model.test_clfier_score()

        clfier_saver = tf.train.Saver(tf.global_variables())
        if tf.gfile.Exists(FLAGS.clfier_log_dir):
            tf.gfile.DeleteRecursively(FLAGS.clfier_log_dir)
        tf.gfile.MakeDirs(FLAGS.clfier_log_dir)

        clfier_checkpoint_path = os.path.join(FLAGS.clfier_log_dir,
                                              'model.ckpt')

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
        with tf.Session(config=tf.ConfigProto(
                gpu_options=gpu_options)) as sess:

            init_op = tf.variables_initializer(tf.global_variables())
            sess.run(init_op)
            tf.train.start_queue_runners(sess=sess)

            # actual training loop
            training_steps = FLAGS.train_steps
            for step in range(training_steps):
                #                 if sv.should_stop():
                #                     break
                try:
                    _ = sess.run([train_op])
                    # for debugging and learning purposes, see how the loss gets decremented thru training steps
                    if (step + 1) % 10 == 0:
                        print("[%d] loss: [%r]" %
                              (step + 1, sess.run(total_loss)))
                    if (step + 1) % 20 == 0:
                        test_evaluate(sess, test_clfier_score, model.inp_w,
                                      model.inp_c, clfier_tX, clfier_tcX,
                                      clfier_tY)
                except KeyboardInterrupt, e:
                    #     sv.saver.save(
                    #         sess,
                    #         FLAGS.clfier_log_dir + '/model',
                    #         global_step=(step + 1))
                    #     raise e
                    # sv.saver.save(sess, FLAGS.clfier_log_dir + '/finnal-model')
                    clfier_saver.save(
                        sess, clfier_checkpoint_path, global_step=(step + 1))
                    raise e
            clfier_saver.save(sess, clfier_checkpoint_path)


if __name__ == '__main__':
    tf.app.run()

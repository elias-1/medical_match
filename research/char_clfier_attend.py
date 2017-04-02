#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2016 www.drcubic.com, Inc. All Rights Reserved
#
"""
File: char_clfier_attend.py
Author: shileicao(shileicao@stu.xjtu.edu.cn)
Date: 2017-04-02 16:11:46
"""

from __future__ import absolute_import, division, print_function

import os

import numpy as np
import tensorflow as tf
from utils import (ENTITY_TYPES, MAX_COMMON_LEN, MAX_SENTENCE_LEN,
                   do_load_data_char_attend, load_w2v)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_data_path', "char_clfier_train_attend.txt",
                           'Training data dir')
tf.app.flags.DEFINE_string('test_data_path', "char_clfier_test_attend.txt",
                           'Test data dir')
tf.app.flags.DEFINE_string('clfier_log_dir', "char_clfier_attend_logs",
                           'The log  dir')
tf.app.flags.DEFINE_string('ner_log_dir', "ner_logs", 'The log  dir')

tf.app.flags.DEFINE_string("char2vec_path",
                           "../clfier_with_w2v/chars_vec_100.txt",
                           "the char2vec data path")
tf.app.flags.DEFINE_integer("max_sentence_len", MAX_SENTENCE_LEN,
                            "max num of tokens per query")
tf.app.flags.DEFINE_integer("embedding_char_size", 100,
                            "second embedding size")
tf.app.flags.DEFINE_integer("num_hidden", 100, "hidden unit number")
tf.app.flags.DEFINE_integer("batch_size", 64, "num example per mini batch")
tf.app.flags.DEFINE_integer("train_steps", 2000, "trainning steps")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "learning rate")

tf.app.flags.DEFINE_float("filter_size", 20, "The clfier's conv fileter size")
tf.app.flags.DEFINE_float("num_filters", 128, "Number of filters")
tf.app.flags.DEFINE_float("num_classes", 7, "Number of classes to classify")
tf.app.flags.DEFINE_float('dropout_keep_prob', 0.5,
                          'Dropout keep probability (default: 0.5)')

tf.flags.DEFINE_float('l2_reg_lambda', 0,
                      'L2 regularization lambda (default: 0.0)')

tf.flags.DEFINE_float('matrix_norm', 1, 'frobieums norm (default: 1)')


def linear(args, output_size, bias, bias_start=0.0, scope=None, reuse=None):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.
    scope: VariableScope for the created subgraph; defaults to "Linear".

  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
    if args is None or (isinstance(args, (list, tuple)) and not args):
        raise ValueError('`args` must be specified')
    if not isinstance(args, (list, tuple)):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError('Linear is expecting 2D arguments: %s' %
                             str(shapes))
        if not shape[1]:
            raise ValueError('Linear expects shape[1] of arguments: %s' %
                             str(shapes))
        else:
            total_arg_size += shape[1]

    # Now the computation.
    with tf.variable_scope(scope or 'Linear', reuse=reuse):
        matrix = tf.get_variable('Matrix', [total_arg_size, output_size])
        if len(args) == 1:
            res = tf.matmul(args[0], matrix)
        else:
            res = tf.matmul(tf.concat(axis=1, values=args), matrix)
        if not bias:
            return res
        bias_term = tf.get_variable(
            'Bias', [output_size],
            initializer=tf.constant_initializer(bias_start))
    return res + bias_term


class Model:
    def __init__(self, c2vPath, numHidden):
        self.numHidden = numHidden
        self.c2v = load_w2v(c2vPath, FLAGS.embedding_char_size)
        self.chars = tf.Variable(self.c2v, name="chars")

        self.common_id_embedding_pad = tf.constant(
            0.0, shape=[1, numHidden * 2], name="common_id_embedding_pad")

        self.common_id_embedding = tf.Variable(
            tf.random_uniform([len(ENTITY_TYPES), numHidden * 2], -1.0, 1.0),
            name="common_id_embedding")

        self.common_embedding = tf.concat(
            0, [self.common_id_embedding_pad, self.common_id_embedding],
            name='common_embedding')

        with tf.variable_scope('Attention') as scope:
            self.attend_W = tf.get_variable(
                "attend_W",
                shape=[1, 1, self.numHidden * 2, self.numHidden * 2],
                regularizer=tf.contrib.layers.l2_regularizer(0.0001),
                initializer=tf.truncated_normal_initializer(stddev=0.01),
                dtype=tf.float32)

            self.attend_V = tf.get_variable(
                "attend_V",
                shape=[self.numHidden * 2],
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

        self.inp_c = tf.placeholder(
            tf.int32, shape=[None, FLAGS.max_sentence_len], name="input_words")

        self.entity_info = tf.placeholder(
            tf.int32, shape=[None, MAX_COMMON_LEN], name="entity_info")

    def length(self, data):
        used = tf.sign(tf.abs(data))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    def inference(self, clfier_cX, entity_info, reuse=None, trainMode=True):

        char_vectors = tf.nn.embedding_lookup(self.chars, clfier_cX)
        length = self.length(clfier_cX)
        length_64 = tf.cast(length, tf.int64)

        #if trainMode:
        #  char_vectors = tf.nn.dropout(char_vectors, FLAGS.dropout_keep_prob)
        with tf.variable_scope("rnn_fwbw", reuse=reuse) as scope:
            forward_output, _ = tf.nn.dynamic_rnn(
                tf.contrib.rnn.LSTMCell(self.numHidden),
                char_vectors,
                dtype=tf.float32,
                sequence_length=length,
                scope="RNN_forward")
            backward_output_, _ = tf.nn.dynamic_rnn(
                tf.contrib.rnn.LSTMCell(self.numHidden),
                inputs=tf.reverse_sequence(char_vectors, length_64, seq_dim=1),
                dtype=tf.float32,
                sequence_length=length,
                scope="RNN_backword")

        backward_output = tf.reverse_sequence(
            backward_output_, length_64, seq_dim=1)

        output = tf.concat([forward_output, backward_output], 2)
        if trainMode:
            output = tf.nn.dropout(output, FLAGS.dropout_keep_prob)

        entity_emb = tf.nn.embedding_lookup(self.common_embedding, entity_info)

        hidden = tf.reshape(
            output, [-1, FLAGS.max_sentence_len, 1, self.numHidden * 2])
        hidden_feature = tf.nn.conv2d(hidden, self.attend_W, [1, 1, 1, 1],
                                      "SAME")
        query = tf.reduce_sum(entity_emb, axis=1)
        y = linear(query, self.numHidden * 2, True, reuse=reuse)
        y = tf.reshape(y, [-1, 1, 1, self.numHidden * 2])
        # Attention mask is a softmax of v^T * tanh(...).
        s = tf.reduce_sum(self.attend_V * tf.tanh(hidden_feature + y), [2, 3])
        a = tf.nn.softmax(s)
        # Now calculate the attention-weighted vector d.
        d = tf.reduce_sum(
            tf.reshape(a, [-1, FLAGS.max_sentence_len, 1, 1]) * hidden, [1, 2])
        ds = tf.reshape(d, [-1, self.numHidden * 2])

        scores = tf.nn.xw_plus_b(ds, self.clfier_softmax_W,
                                 self.clfier_softmax_b)
        return scores

    def loss(self, clfier_X, clfier_cX, clfier_Y, entity_info):
        self.scores = self.inference(clfier_X, clfier_cX, entity_info)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.scores, labels=clfier_Y)
        loss = tf.reduce_mean(cross_entropy, name='cross_entropy')
        regularization_loss = tf.add_n(
            tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        normed_embedding = tf.nn.l2_normalize(self.common_id_embedding, dim=1)
        similarity_matrix = tf.matmul(normed_embedding,
                                      tf.transpose(normed_embedding, [1, 0]))
        fro_norm = tf.reduce_sum(tf.nn.l2_loss(similarity_matrix))
        final_loss = loss + regularization_loss * FLAGS.l2_reg_lambda + fro_norm * FLAGS.matrix_norm
        return final_loss

    def test_clfier_score(self):
        scores = self.inference(
            self.inp_c, self.entity_info, reuse=True, trainMode=False)
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
        record_defaults=[
            [0] for i in range(FLAGS.max_sentence_len * +1 + MAX_COMMON_LEN)
        ])

    # batch actually reads the file and loads "batch_size" rows in a single tensor
    return tf.train.shuffle_batch(
        decoded,
        batch_size=batch_size,
        capacity=batch_size * 4,
        min_after_dequeue=batch_size)


def inputs(path):
    whole = read_csv(FLAGS.batch_size, path)
    features = tf.transpose(tf.stack(whole[0:FLAGS.max_sentence_len]))
    len_features = FLAGS.max_sentence_len
    label = tf.transpose(tf.concat(whole[len_features:len_features + 1], 0))
    entity_info = tf.transpose(tf.stack(whole[len_features + 1:]))
    return features, label, entity_info


def train(total_loss):
    return tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(total_loss)


def test_evaluate(sess, test_clfier_score, inp_c, entity_info, clfier_tcX,
                  clfier_tY, tentity_info):
    batchSize = FLAGS.batch_size
    totalLen = clfier_tcX.shape[0]
    numBatch = int((totalLen - 1) / batchSize) + 1
    correct_clfier_labels = 0
    for i in range(numBatch):
        endOff = (i + 1) * batchSize
        if endOff > totalLen:
            endOff = totalLen
        y = clfier_tY[i * batchSize:endOff]
        feed_dict = {
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
        model = Model(FLAGS.char2vec_path, FLAGS.num_hidden)
        print("train data path:", trainDataPath)
        clfier_cX, clfier_Y, entity_info = inputs(trainDataPath)
        clfier_tcX, clfier_tY, tentity_info = do_load_data_char_attend(
            FLAGS.test_data_path, FLAGS.max_sentence_len)
        total_loss = model.loss(clfier_cX, clfier_Y, entity_info)
        train_op = train(total_loss)
        test_clfier_score = model.test_clfier_score()

        # ner_variables = [
        #     v for v in tf.global_variables() if 'rnn_fwbw' in v.name
        # ] + [model.words, model.chars, model.char_filter]

        # ner_saver = tf.train.Saver(ner_variables)

        clfier_saver = tf.train.Saver(tf.global_variables())
        if tf.gfile.Exists(FLAGS.clfier_log_dir):
            tf.gfile.DeleteRecursively(FLAGS.clfier_log_dir)
        tf.gfile.MakeDirs(FLAGS.clfier_log_dir)

        clfier_checkpoint_path = os.path.join(FLAGS.clfier_log_dir,
                                              'model.ckpt')

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
        with tf.Session(config=tf.ConfigProto(
                gpu_options=gpu_options)) as sess:
            # sv = tf.train.Supervisor(graph=graph, logdir=FLAGS.clfier_log_dir)
            # with sv.managed_session(
            #         master='',
            #         config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

            # ner_saver.restore(sess, ner_checkpoint_file)
            #
            # other_variables = [
            #     v for v in tf.global_variables() if v not in ner_variables
            # ]
            #
            # init_op = tf.variables_initializer(other_variables)
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
                        test_evaluate(sess, test_clfier_score, model.inp_c,
                                      model.entity_info, clfier_tcX, clfier_tY,
                                      tentity_info)
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

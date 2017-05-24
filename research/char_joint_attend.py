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
                   do_load_data_joint_attend, load_w2v)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_data_path', "joint_train.txt",
                           'Training data dir')
tf.app.flags.DEFINE_string('test_data_path', "joint_test.txt", 'Test data dir')
tf.app.flags.DEFINE_string('joint_log_dir', "joint_logs", 'The log  dir')

tf.app.flags.DEFINE_string("char2vec_path",
                           "../clfier_with_w2v/chars_vec_100.txt",
                           "the char2vec data path")
tf.app.flags.DEFINE_integer("max_sentence_len", MAX_SENTENCE_LEN,
                            "max num of tokens per query")
tf.app.flags.DEFINE_integer("embedding_char_size", 100,
                            "second embedding size")
tf.app.flags.DEFINE_integer("num_tags", 30, "num ner tags")
tf.app.flags.DEFINE_integer("num_hidden", 100, "hidden unit number")
tf.app.flags.DEFINE_integer("batch_size", 64, "num example per mini batch")
tf.app.flags.DEFINE_integer("train_steps", 2000, "trainning steps")
tf.app.flags.DEFINE_integer("joint_steps", 100, "trainning steps")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "learning rate")

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
    def __init__(self, distinctTagNum, c2vPath, numHidden):
        self.distinctTagNum = distinctTagNum
        self.numHidden = numHidden
        self.c2v = load_w2v(c2vPath, FLAGS.embedding_char_size)
        self.chars = tf.Variable(self.c2v, name="chars")

        self.common_id_embedding_pad = tf.constant(
            0.0, shape=[1, numHidden * 2], name="common_id_embedding_pad")

        self.common_id_embedding = tf.Variable(
            tf.random_uniform([len(ENTITY_TYPES), numHidden * 2], -1.0, 1.0),
            name="common_id_embedding")

        self.common_embedding = tf.concat(
            [self.common_id_embedding_pad, self.common_id_embedding],
            0,
            name='common_embedding')

        with tf.variable_scope('Ner_output') as scope:
            self.W = tf.get_variable(
                shape=[numHidden * 2, distinctTagNum],
                initializer=tf.truncated_normal_initializer(stddev=0.01),
                name="weights",
                regularizer=tf.contrib.layers.l2_regularizer(0.001))
            self.b = tf.Variable(tf.zeros([distinctTagNum], name="bias"))

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

    def inference(self,
                  clfier_cX,
                  model='ner',
                  entity_info=None,
                  rnn_reuse=None,
                  linear_resue=None,
                  trainMode=True):

        char_vectors = tf.nn.embedding_lookup(self.chars, clfier_cX)
        length = self.length(clfier_cX)
        length_64 = tf.cast(length, tf.int64)

        # if trainMode:
        #  char_vectors = tf.nn.dropout(char_vectors, FLAGS.dropout_keep_prob)
        with tf.variable_scope("rnn_fwbw", reuse=rnn_reuse) as scope:
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

        if model == 'ner':
            output = tf.reshape(output, [-1, self.numHidden * 2])
            matricized_unary_scores = tf.matmul(output, self.W) + self.b
            # matricized_unary_scores = tf.nn.log_softmax(matricized_unary_scores)
            unary_scores = tf.reshape(
                matricized_unary_scores,
                [-1, FLAGS.max_sentence_len, self.distinctTagNum])

            return unary_scores, length
        elif model == 'clfier':
            entity_emb = tf.nn.embedding_lookup(self.common_embedding,
                                                entity_info)

            hidden = tf.reshape(
                output, [-1, FLAGS.max_sentence_len, 1, self.numHidden * 2])
            hidden_feature = tf.nn.conv2d(hidden, self.attend_W, [1, 1, 1, 1],
                                          "SAME")
            query = tf.reduce_sum(entity_emb, axis=1)
            y = linear(query, self.numHidden * 2, True, reuse=linear_resue)
            y = tf.reshape(y, [-1, 1, 1, self.numHidden * 2])
            # Attention mask is a softmax of v^T * tanh(...).
            s = tf.reduce_sum(self.attend_V *
                              tf.tanh(hidden_feature + 0.1 * y), [2, 3])
            a = tf.nn.softmax(s)
            # Now calculate the attention-weighted vector d.
            d = tf.reduce_sum(
                tf.reshape(a, [-1, FLAGS.max_sentence_len, 1, 1]) * hidden,
                [1, 2])
            ds = tf.reshape(d, [-1, self.numHidden * 2])

            scores = tf.nn.xw_plus_b(ds, self.clfier_softmax_W,
                                     self.clfier_softmax_b)
            return scores, length
        else:
            raise ValueError('model must either be clfier or ner')

    def ner_loss(self, ner_cX, ner_Y):
        P, sequence_length = self.inference(ner_cX, model='ner')
        log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
            P, ner_Y, sequence_length)
        loss = tf.reduce_mean(-log_likelihood)
        regularization_loss = tf.add_n(
            tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        return loss + regularization_loss * FLAGS.l2_reg_lambda

    def clfier_loss(self, clfier_cX, clfier_Y, entity_info):
        self.scores, _ = self.inference(
            clfier_cX, model='clfier', entity_info=entity_info, rnn_reuse=True)
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

    def test_unary_score(self):
        P, sequence_length = self.inference(
            self.inp_c, model='ner', rnn_reuse=True, trainMode=False)
        return P, sequence_length

    def test_clfier_score(self):
        scores, _ = self.inference(
            self.inp_c,
            model='clfier',
            entity_info=self.entity_info,
            rnn_reuse=True,
            linear_resue=True,
            trainMode=False)
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
            [0] for i in range(FLAGS.max_sentence_len * 2 + 1 + MAX_COMMON_LEN)
        ])

    # batch actually reads the file and loads "batch_size" rows in a single tensor
    return tf.train.shuffle_batch(
        decoded,
        batch_size=batch_size,
        capacity=batch_size * 4,
        min_after_dequeue=batch_size)


def inputs(path):
    whole = read_csv(FLAGS.batch_size, path)
    ner_train_len = FLAGS.max_sentence_len * 2
    ner_features = clfier_features = tf.transpose(
        tf.stack(whole[0:FLAGS.max_sentence_len]))

    ner_label = tf.transpose(
        tf.stack(whole[FLAGS.max_sentence_len:2 * FLAGS.max_sentence_len]))

    clfier_label = tf.transpose(
        tf.concat(whole[ner_train_len:ner_train_len + 1], 0))
    entity_info = tf.transpose(tf.stack(whole[ner_train_len + 1:]))
    return ner_features, ner_label, clfier_features, clfier_label, entity_info


def train(total_loss, var_list=None):
    return tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
        total_loss, var_list=var_list)


def ner_test_evaluate(sess, unary_score, test_sequence_length, transMatrix,
                      inp_ner_c, ner_cX, ner_Y):
    batchSize = FLAGS.batch_size
    totalLen = ner_cX.shape[0]
    numBatch = int((ner_cX.shape[0] - 1) / batchSize) + 1
    correct_labels = 0
    total_labels = 0
    entity_infos = []
    for i in range(numBatch):
        endOff = (i + 1) * batchSize
        if endOff > totalLen:
            endOff = totalLen
        y = ner_Y[i * batchSize:endOff]
        feed_dict = {inp_ner_c: ner_cX[i * batchSize:endOff]}
        unary_score_val, test_sequence_length_val = sess.run(
            [unary_score, test_sequence_length], feed_dict)
        for tf_unary_scores_, y_, sequence_length_ in zip(
                unary_score_val, y, test_sequence_length_val):
            # print("seg len:%d" % (sequence_length_))
            tf_unary_scores_ = tf_unary_scores_[:sequence_length_]
            y_ = y_[:sequence_length_]
            viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(
                tf_unary_scores_, transMatrix)
            # Evaluate word-level accuracy.
            correct_labels += np.sum(np.equal(viterbi_sequence, y_))
            total_labels += sequence_length_
            entity_infos.append(viterbi_sequence)

    accuracy = 100.0 * correct_labels / float(total_labels)
    print("NER Accuracy: %.3f%%" % accuracy)
    return entity_infos


def clfier_test_evaluate(sess, test_clfier_score, inp_c, entity_info,
                         clfier_tcX, clfier_tY, tentity_info):
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
    print("Clfier Accuracy: %.3f%%" % accuracy)


def decode_entity_location(entity_info):
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
    types_id = map(lambda x: int(x) + 1, types_id)
    return entity_location, types_id


def entity_to_common(entity_infos):
    tentity_info = []
    for i in xrange(len(entity_infos)):
        entity_location, types_id = decode_entity_location(entity_infos[i])
        nl = len(types_id)

        for i in range(nl, MAX_COMMON_LEN):
            types_id.append('0')
        tentity_info.append(types_id[:MAX_COMMON_LEN])

    return np.array(tentity_info)


def main(unused_argv):
    trainDataPath = FLAGS.train_data_path
    graph = tf.Graph()
    with graph.as_default():
        model = Model(FLAGS.num_tags, FLAGS.char2vec_path, FLAGS.num_hidden)
        print("train data path:", os.path.realpath(trainDataPath))
        ner_cX, ner_Y, clfier_cX, clfier_Y, entity_info = inputs(trainDataPath)
        ner_tcX, ner_tY, clfier_tcX, clfier_tY, aa = do_load_data_joint_attend(
            FLAGS.test_data_path, FLAGS.max_sentence_len)

        ner_total_loss = model.ner_loss(ner_cX, ner_Y)
        ner_var_list = [
            v for v in tf.global_variables()
            if 'Attention' not in v.name and 'Clfier_output' not in v.name and
            'Linear' not in v.name
        ]
        print('ner var list:')
        print([v.name for v in ner_var_list])

        ner_train_op = train(ner_total_loss, var_list=ner_var_list)
        ner_test_unary_score, ner_test_sequence_length = model.test_unary_score(
        )
        clfier_total_loss = model.clfier_loss(clfier_cX, clfier_Y, entity_info)
        clfier_var_list = [
            v for v in tf.global_variables()
            if 'Ner_output' not in v.name and 'transitions' not in v.name and
            'rnn_fwbw' not in v.name
        ]
        print('clfier var list:')
        print([v.name for v in clfier_var_list])

        clfier_train_op = train(clfier_total_loss, var_list=clfier_var_list)
        test_clfier_score = model.test_clfier_score()

        ner_seperate_list = [
            v for v in tf.global_variables()
            if 'Ner_output' in v.name or 'transition' in v.name
        ]
        ner_seperate_op = train(ner_total_loss, var_list=ner_seperate_list)

        clfier_seperate_list = [
            v for v in tf.global_variables()
            if 'Attention' in v.name or 'Clfier_output' in v.name or 'Linear'
            in v.name
        ]
        clfier_seperate_op = train(
            ner_total_loss, var_list=clfier_seperate_list)

        sv = tf.train.Supervisor(graph=graph, logdir=FLAGS.joint_log_dir)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
        with sv.managed_session(
                master='',
                config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            # actual training loop
            training_steps = FLAGS.train_steps
            for step in range(training_steps):
                if sv.should_stop():
                    break
                try:
                    if step < FLAGS.joint_steps:
                        _, trainsMatrix = sess.run(
                            [ner_train_op, model.transition_params])
#                     else:
#                         _, trainsMatrix = sess.run(
#                            [ner_seperate_op, model.transition_params])
# for debugging and learning purposes, see how the loss gets decremented thru training steps
                    if (step + 1) % 10 == 0:
                        print(
                            "[%d] NER loss: [%r]    Classification loss: [%r]"
                            % (step + 1, sess.run(ner_total_loss),
                               sess.run(clfier_total_loss)))
                    if (step + 1) % 20 == 0:
                        entity_infos = ner_test_evaluate(
                            sess, ner_test_unary_score,
                            ner_test_sequence_length, trainsMatrix,
                            model.inp_c, ner_tcX, ner_tY)
                        tentity_info = entity_to_common(entity_infos)
                        clfier_test_evaluate(sess, test_clfier_score,
                                             model.inp_c, model.entity_info,
                                             clfier_tcX, clfier_tY,
                                             tentity_info)
                    _ = sess.run([clfier_train_op])

#                     if step < FLAGS.joint_steps:
#                         if step > 200:
#                             _ = sess.run([clfier_train_op])
#                     else:
#                         _ = sess.run([clfier_seperate_op])

                except KeyboardInterrupt, e:
                    sv.saver.save(
                        sess,
                        FLAGS.joint_log_dir + '/model',
                        global_step=(step + 1))
                    raise e
            sv.saver.save(sess, FLAGS.joint_log_dir + '/finnal-model')

if __name__ == '__main__':
    tf.app.run()

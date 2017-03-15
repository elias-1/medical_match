# -*- coding: utf-8 -*-
# @Author: Koth
# @Date:   2017-01-24 16:13:14
# @Last Modified by:   Elias
# @Last Modified time: 2017-02-06 15:23:12

# python train_ner.py --word_ner_word2vec_path data/glove.6B.100d.txt --ner_train_data_path /home/elias/code/deep-drcubic/NER_with_sentence_clfier/train.txt --ner_test_data_path test.txt --ner_learning_rate 0.001

from __future__ import absolute_import, division, print_function

import os
import time

import numpy as np
import tensorflow as tf
from .utils import load_w2v

NER_MAX_SENTENCE_LEN = 80
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'ner_train_data_path',
    "/home/elias/code/medical_match/clfier_with_w2v/ner_train_v2.txt",
    'Training data dir')
tf.app.flags.DEFINE_string('ner_test_data_path', "ner_test_v2.txt",
                           'Test data dir')
tf.app.flags.DEFINE_string('ner_log_dir', "ner_logs_v2", 'The log  dir')
tf.app.flags.DEFINE_string("ner_word2vec_path", "chars_vec_100.txt",
                           "the word2vec data path")
tf.app.flags.DEFINE_integer("ner_max_sentence_len", NER_MAX_SENTENCE_LEN,
                            "max num of tokens per query")
tf.app.flags.DEFINE_integer("ner_embedding_word_size", 100, "embedding size")
tf.app.flags.DEFINE_integer("ner_num_tags", 30, "num pos tags")
tf.app.flags.DEFINE_integer("ner_num_hidden", 100, "hidden unit number")
tf.app.flags.DEFINE_integer("ner_batch_size", 64, "num example per mini batch")
tf.app.flags.DEFINE_integer("ner_train_steps", 1500, "trainning steps")
tf.app.flags.DEFINE_float("ner_learning_rate", 0.001, "learning rate")
tf.app.flags.DEFINE_float('ner_dropout_keep_prob', 0.5,
                          'Dropout keep probability (default: 0.5)')

tf.flags.DEFINE_float('ner_l2_reg_lambda', 1,
                      'L2 regularization lambda (default: 0.0)')

TIMESTAMP = str(int(time.time()))


def do_load_data(path):
    wx = []
    y = []
    fp = open(path, "r")
    ln = 0
    for line in fp.readlines():
        line = line.rstrip()
        ln += 1
        if not line:
            continue
        ss = line.split(" ")
        if len(ss) != (FLAGS.ner_max_sentence_len * 2):
            print("[line:%d]len ss:%d,origin len:%d\n%s" %
                  (ln, len(ss), len(line), line))
        assert (len(ss) == (FLAGS.ner_max_sentence_len * 2))
        lwx = []
        ly = []
        for i in range(FLAGS.ner_max_sentence_len):
            lwx.append(int(ss[i]))
            ly.append(int(ss[i + FLAGS.ner_max_sentence_len]))
        wx.append(lwx)
        y.append(ly)
    fp.close()
    return np.array(wx), np.array(y)


class Model:
    def __init__(self, distinctTagNum, w2vPath, numHidden):
        self.distinctTagNum = distinctTagNum
        self.numHidden = numHidden
        self.w2v = load_w2v(w2vPath, FLAGS.ner_embedding_word_size)
        self.words = tf.Variable(self.w2v, name="words")

        with tf.variable_scope('Ner_output') as scope:
            self.W = tf.get_variable(
                shape=[numHidden * 2, distinctTagNum],
                initializer=tf.truncated_normal_initializer(stddev=0.01),
                name="weights",
                regularizer=tf.contrib.layers.l2_regularizer(0.001))
            self.b = tf.Variable(tf.zeros([distinctTagNum], name="bias"))
        self.inp_w = tf.placeholder(
            tf.int32,
            shape=[None, FLAGS.ner_max_sentence_len],
            name="input_words")

    def length(self, data):
        used = tf.sign(tf.abs(data))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    def inference(self, wX, reuse=None, trainMode=True):
        word_vectors = tf.nn.embedding_lookup(self.words, wX)

        length = self.length(wX)
        length_64 = tf.cast(length, tf.int64, name='length')

        #if trainMode:
        #  word_vectors = tf.nn.dropout(word_vectors, FLAGS.ner_dropout_keep_prob)
        with tf.variable_scope("rnn_fwbw", reuse=reuse) as scope:
            forward_output, _ = tf.nn.dynamic_rnn(
                tf.nn.rnn_cell.BasicLSTMCell(self.numHidden),
                word_vectors,
                dtype=tf.float32,
                sequence_length=length,
                scope="RNN_forward")
            backward_output_, _ = tf.nn.dynamic_rnn(
                tf.nn.rnn_cell.BasicLSTMCell(self.numHidden),
                inputs=tf.reverse_sequence(
                    word_vectors, length_64, seq_dim=1),
                dtype=tf.float32,
                sequence_length=length,
                scope="RNN_backword")

        backward_output = tf.reverse_sequence(
            backward_output_, length_64, seq_dim=1)

        output = tf.concat(2, [forward_output, backward_output])
        output = tf.reshape(output, [-1, self.numHidden * 2])
        if trainMode:
            output = tf.nn.dropout(output, FLAGS.ner_dropout_keep_prob)

        matricized_unary_scores = tf.matmul(output, self.W) + self.b
        # matricized_unary_scores = tf.nn.log_softmax(matricized_unary_scores)
        unary_scores = tf.reshape(
            matricized_unary_scores,
            [-1, FLAGS.ner_max_sentence_len, self.distinctTagNum],
            name='unary_scores')

        return unary_scores, length

    def loss(self, wX, Y):
        P, sequence_length = self.inference(wX)
        log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
            P, Y, sequence_length)
        loss = tf.reduce_mean(-log_likelihood)
        regularization_loss = tf.add_n(
            tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        return loss + regularization_loss * FLAGS.ner_l2_reg_lambda

    def test_unary_score(self):
        P, sequence_length = self.inference(
            self.inp_w, reuse=True, trainMode=False)
        return P, sequence_length


def read_csv(ner_batch_size, file_name):
    filename_queue = tf.train.string_input_producer([file_name])
    reader = tf.TextLineReader(skip_header_lines=0)
    key, value = reader.read(filename_queue)
    # decode_csv will convert a Tensor from type string (the text line) in
    # a tuple of tensor columns with the specified defaults, which also
    # sets the data type for each column
    decoded = tf.decode_csv(
        value,
        field_delim=' ',
        record_defaults=[[0] for i in range(FLAGS.ner_max_sentence_len * 2)])

    # batch actually reads the file and loads "ner_batch_size" rows in a single tensor
    return tf.train.shuffle_batch(
        decoded,
        ner_batch_size=ner_batch_size,
        capacity=ner_batch_size * 4,
        min_after_dequeue=ner_batch_size)


def inputs(path):
    whole = read_csv(FLAGS.ner_batch_size, path)
    features = tf.transpose(tf.stack(whole[0:FLAGS.ner_max_sentence_len]))
    label = tf.transpose(tf.stack(whole[FLAGS.ner_max_sentence_len:]))
    return features, label


def train(total_loss):
    return tf.train.AdamOptimizer(FLAGS.ner_learning_rate).minimize(total_loss)


def test_evaluate(sess, unary_score, test_sequence_length, transMatrix, inp_w,
                  twX, tY):
    batchSize = FLAGS.ner_batch_size
    totalLen = twX.shape[0]
    numBatch = int((twX.shape[0] - 1) / batchSize) + 1
    correct_labels = 0
    total_labels = 0
    for i in range(numBatch):
        endOff = (i + 1) * batchSize
        if endOff > totalLen:
            endOff = totalLen
        y = tY[i * batchSize:endOff]
        feed_dict = {inp_w: twX[i * batchSize:endOff], }
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
    accuracy = 100.0 * correct_labels / float(total_labels)
    print("Accuracy: %.3f%%" % accuracy)


def main(unused_argv):
    # curdir = os.path.dirname(os.path.realpath(__file__))
    trainDataPath = FLAGS.ner_train_data_path
    # if not trainDataPath.startswith("/"):
    #     trainDataPath = curdir + "/../../" + trainDataPath
    graph = tf.Graph()
    with graph.as_default():
        model = Model(FLAGS.ner_num_tags, FLAGS.ner_word2vec_path,
                      FLAGS.ner_num_hidden)
        print("train data path:", trainDataPath)
        wX, Y = inputs(trainDataPath)
        twX, tY = do_load_data(FLAGS.ner_test_data_path)
        total_loss = model.loss(wX, Y)
        train_op = train(total_loss)
        test_unary_score, test_sequence_length = model.test_unary_score()

        logdir = os.path.join(FLAGS.ner_log_dir, TIMESTAMP)
        sv = tf.train.Supervisor(graph=graph, logdir=logdir)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
        with sv.managed_session(
                master='',
                config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            # actual training loop
            training_steps = FLAGS.ner_train_steps
            for step in range(training_steps):
                if sv.should_stop():
                    break
                try:
                    _, trainsMatrix = sess.run(
                        [train_op, model.transition_params])
                    # for debugging and learning purposes, see how the loss gets decremented thru training steps
                    if (step + 1) % 10 == 0:
                        print("[%d] loss: [%r]" %
                              (step + 1, sess.run(total_loss)))
                    if (step + 1) % 20 == 0:
                        test_evaluate(sess, test_unary_score,
                                      test_sequence_length, trainsMatrix,
                                      model.inp_w, twX, tY)
                except KeyboardInterrupt, e:
                    sv.saver.save(
                        sess, logdir + '/model', global_step=(step + 1))
                    raise e
            sv.saver.save(sess, logdir + '/finnal-model')
            np.save('transition.npy', trainsMatrix)


if __name__ == '__main__':
    tf.app.run()

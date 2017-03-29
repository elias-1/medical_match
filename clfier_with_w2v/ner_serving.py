#!/usr/bin/env python
# -*- coding: utf-8 -*-

########################################################################
#
# Copyright (c) 2016 www.drcubic.com, Inc. All Rights Reserved
#
########################################################################
"""
File: ner_serving.py
Author: shileicao(shileicao@stu.xjtu.edu.cn)
Date: 2016/12/28 16:49:22
"""
"""Export cnn_clfier model given existing training checkpoints.
The model is exported as SavedModel with proper signatures that can be loaded by
standard tensorflow_model_server.
"""

import os.path

import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import (
    signature_constants, signature_def_utils, tag_constants, utils)
from tensorflow.python.util import compat
from train_ner import FLAGS, NER_MAX_SENTENCE_LEN, Model

# This is a placeholder for a Google-internal import.

tf.app.flags.DEFINE_string('checkpoint_dir', 'ner_logs_v2/1490617207',
                           """Directory where to read training checkpoints.""")
tf.app.flags.DEFINE_string('output_dir', '/tmp/ner_output',
                           """Directory where to export inference model.""")
tf.app.flags.DEFINE_integer('model_version', 1,
                            """Version number of the model.""")

WORKING_DIR = os.path.dirname(os.path.realpath(__file__))


def export():

    with tf.Graph().as_default():
        # Build inference model.
        # Please refer to Tensorflow inception model for details.

        # Input transformation.
        serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
        feature_configs = {
            'words/encoded':
            tf.FixedLenFeature(shape=[NER_MAX_SENTENCE_LEN], dtype=tf.int64),
        }
        tf_example = tf.parse_example(serialized_tf_example, feature_configs)
        features = tf_example['words/encoded']

        # Run inference.
        model = Model(FLAGS.ner_num_tags, FLAGS.ner_word2vec_path,
                      FLAGS.ner_num_hidden)
        test_unary_score, test_sequence_length = model.inference(
            features, trainMode=False)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            # Restore variables from training checkpoints.
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                # Assuming model_checkpoint_path looks something like:
                #   /my-favorite-path/imagenet_train/model.ckpt-0,
                # extract global_step from it.
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split(
                    '-')[-1]
                print 'Successfully loaded model from %s at step=%s.' % (
                    ckpt.model_checkpoint_path, global_step)
            else:
                print 'No checkpoint file found at %s' % FLAGS.checkpoint_dir
                return

            # Export inference model.
            output_path = os.path.join(
                compat.as_bytes(FLAGS.output_dir),
                compat.as_bytes(str(FLAGS.model_version)))
            print 'Exporting trained model to', output_path
            builder = saved_model_builder.SavedModelBuilder(output_path)

            # Build the signature_def_map.
            default_inputs_tensor_info = utils.build_tensor_info(
                serialized_tf_example)

            scores_output_tensor_info = utils.build_tensor_info(
                test_unary_score)

            scores_seq_len_tensor_info = utils.build_tensor_info(
                test_sequence_length)

            default_signature = signature_def_utils.build_signature_def(
                inputs={
                    signature_constants.CLASSIFY_INPUTS:
                    default_inputs_tensor_info
                },
                outputs={
                    signature_constants.CLASSIFY_OUTPUT_SCORES:
                    scores_output_tensor_info,
                    'sequence_length':
                    scores_seq_len_tensor_info
                },
                method_name=signature_constants.CLASSIFY_METHOD_NAME)

            predict_words_tensor_info = utils.build_tensor_info(features)
            prediction_signature = signature_def_utils.build_signature_def(
                inputs={
                    'words': predict_words_tensor_info,
                },
                outputs={
                    'scores': scores_output_tensor_info,
                    'sequence_length': scores_seq_len_tensor_info
                },
                method_name=signature_constants.PREDICT_METHOD_NAME)

            legacy_init_op = tf.group(
                tf.initialize_all_tables(), name='legacy_init_op')
            builder.add_meta_graph_and_variables(
                sess, [tag_constants.SERVING],
                signature_def_map={
                    'predict_sentence':
                    prediction_signature,
                    signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    default_signature,
                },
                legacy_init_op=legacy_init_op)

            builder.save()
            print 'Successfully exported model to %s' % FLAGS.output_dir


def main(unused_argv=None):
    export()


if __name__ == '__main__':
    tf.app.run()

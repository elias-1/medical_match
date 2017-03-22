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
"""Export cnn_clfier model given existing training checkpoints.
The model is exported as SavedModel with proper signatures that can be loaded by
standard tensorflow_model_server.
"""

import os.path

import tensorflow as tf
from cnn_clfier import C_MAX_SENTENCE_LEN, C_MAX_WORD_LEN, FLAGS, TextCNN
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import (
    signature_constants, signature_def_utils, tag_constants, utils)
from tensorflow.python.util import compat

# This is a placeholder for a Google-internal import.

tf.app.flags.DEFINE_string('checkpoint_dir', 'cnn_clfier_logs/1489571611',
                           """Directory where to read training checkpoints.""")
tf.app.flags.DEFINE_string('output_dir', '/tmp/clfier_output',
                           """Directory where to export inference model.""")
tf.app.flags.DEFINE_integer('model_version', 1,
                            """Version number of the model.""")
tf.app.flags.DEFINE_integer('image_size', 299,
                            """Needs to provide same value as in training.""")

WORKING_DIR = os.path.dirname(os.path.realpath(__file__))


def export():

    with tf.Graph().as_default():
        # Build inference model.
        # Please refer to Tensorflow inception model for details.

        input_shape = C_MAX_SENTENCE_LEN * (1 + C_MAX_WORD_LEN)

        # Input transformation.
        serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
        feature_configs = {
            'sentence/encoded':
            tf.FixedLenFeature(shape=[input_shape], dtype=tf.string),
        }
        tf_example = tf.parse_example(serialized_tf_example, feature_configs)
        whole = tf_example['sentence/encoded']
        features = tf.transpose(tf.stack(whole[0:FLAGS.max_sentence_len]))
        char_features = tf.transpose(
            tf.stack(whole[FLAGS.max_sentence_len:(
                FLAGS.max_chars_per_word + 1) * FLAGS.max_sentence_len]))

        # Run inference.
        model = TextCNN(FLAGS.word2vec_path, FLAGS.char2vec_path)
        scores = model.inference(features, char_features)

        values, indices = tf.nn.top_k(scores, FLAGS.num_classes)
        prediction_classes = tf.contrib.lookup.index_to_string(
            tf.to_int64(indices),
            mapping=tf.constant([str(i) for i in xrange(FLAGS.num_classes)]))

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
            classify_inputs_tensor_info = utils.build_tensor_info(
                serialized_tf_example)
            classes_output_tensor_info = utils.build_tensor_info(
                prediction_classes)
            scores_output_tensor_info = utils.build_tensor_info(values)

            classification_signature = signature_def_utils.build_signature_def(
                inputs={
                    signature_constants.CLASSIFY_INPUTS:
                    classify_inputs_tensor_info
                },
                outputs={
                    signature_constants.CLASSIFY_OUTPUT_CLASSES:
                    classes_output_tensor_info,
                    signature_constants.CLASSIFY_OUTPUT_SCORES:
                    scores_output_tensor_info
                },
                method_name=signature_constants.CLASSIFY_METHOD_NAME)

            predict_features_tensor_info = utils.build_tensor_info(features)
            predict_char_features_tensor_info = utils.build_tensor_info(
                char_features)
            prediction_signature = signature_def_utils.build_signature_def(
                inputs={
                    'features': predict_features_tensor_info,
                    'char_features': predict_char_features_tensor_info
                },
                outputs={
                    'classes': classes_output_tensor_info,
                    'scores': scores_output_tensor_info
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
                    classification_signature,
                },
                legacy_init_op=legacy_init_op)

            builder.save()
            print 'Successfully exported model to %s' % FLAGS.output_dir


def main(unused_argv=None):
    export()


if __name__ == '__main__':
    tf.app.run()

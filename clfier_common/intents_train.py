#!/usr/bin/env python

########################################################################
#
# Copyright (c) 2016 www.drcubic.com, Inc. All Rights Reserved
#
########################################################################
"""
File: intents_train.py
Author: shileicao(shileicao@stu.xjtu.edu.cn)
Date: 2016/12/28 16:49:22

A modified version of the original train.py, which was used to classfy
polarity datasets.

This modified version was used to cope with the intents classification 
task in luis-like application.
"""

# In[ ]:

from __future__ import print_function

import datetime
import json
import os
import random
import time
from copy import deepcopy

import jieba
import numpy as np
import tensorflow as tf
from mm import medical_to_common
from tensorflow.contrib import learn
from text_cnn import TextCNN

# In[ ]:

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_string('split_rate', '7,2,1',
                       'Percentage of the training data to use for validation')
tf.flags.DEFINE_string('data', 'data', 'Data source')
tf.flags.DEFINE_integer('min_examples_per_class', 100,
                        'Minimal examples numbers per class')

# Model Hyperparameters
tf.flags.DEFINE_integer('embedding_dim', 128,
                        'Dimensionality of character embedding (default: 128)')
tf.flags.DEFINE_string('filter_sizes', '3,4,5',
                       'Comma-separated filter sizes (default: \'3,4,5\')')
tf.flags.DEFINE_integer('num_filters', 128,
                        'Number of filters per filter size (default: 128)')
tf.flags.DEFINE_float('dropout_keep_prob', 0.5,
                      'Dropout keep probability (default: 0.5)')
tf.flags.DEFINE_float('l2_reg_lambda', 0.0,
                      'L2 regularization lambda (default: 0.0)')

# Training parameters
tf.flags.DEFINE_integer('batch_size', 64, 'Batch Size (default: 64)')
tf.flags.DEFINE_string('train', 'runs', 'Folder for training')
tf.flags.DEFINE_integer('num_epochs', 200,
                        'Number of training epochs (default: 200)')
tf.flags.DEFINE_integer(
    'evaluate_every', 100,
    'Evaluate model on dev set after this many steps (default: 100)')
tf.flags.DEFINE_integer('checkpoint_every', 100,
                        'Save model after this many steps (default: 100)')
# Misc Parameters
tf.flags.DEFINE_boolean('allow_soft_placement', True,
                        'Allow device soft device placement')
tf.flags.DEFINE_boolean('log_device_placement', False,
                        'Log placement of ops on devices')

FLAGS = tf.flags.FLAGS

# In[ ]:

# get_ipython().magic(u'pinfo jieba.add_word')

# In[ ]:

TIMESTAMP = str(int(time.time()))

# jieba.load_userdict(os.path.join(FLAGS.data,'words.txt'))


def tokenizer(iterator):
    """Tokenizer generator.
    Args:
        iterator: Input iterator with strings.
    Yields:
        array of tokens per each value in the input.
    """
    for value in iterator:
        yield jieba.lcut(value, cut_all=False)


# In[ ]:


def data_shuffle(x, y):
    # Randomly shuffle data
    x_temp = deepcopy(x)
    y_temp = deepcopy(y)
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    return x_temp[shuffle_indices], y_temp[shuffle_indices]


def _load_data_and_labels():
    """Loads data from files, splits the data into
    words and generates labels. Returns split sentences and labels.
    
    Note: The returned x_text and labels are aranged labels by labels. 
    
    Returns:
        x_text: Input.
        labels: Output.
    """
    # Load data from file
    x_text = []
    y = []
    with open(os.path.join(FLAGS.data, 'qa.json'), 'r') as f:
        data = json.load(f)

    num_classes = 0
    label_to_int = {}
    for key in data.keys():
        one_label_size = len(data[key])
        if one_label_size > FLAGS.min_examples_per_class:
            slim_index = int(one_label_size * 1)
            x_text.extend(random.sample(data[key], slim_index))
            y.extend([num_classes] * slim_index)
            label_to_int[key] = num_classes
            num_classes += 1

    run_dir = os.path.join(FLAGS.train, TIMESTAMP)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    with open(os.path.join(run_dir, 'label_to_int.json'), 'w+') as f:
        json.dump(label_to_int, f)

    labels = np.array(y, dtype=np.int32)

    labels = (np.arange(num_classes) == labels[:, None]).astype(np.float32)

    return [x_text, labels]


def _batch_iter(x_train, y_train, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data_size = len(y_train)
    num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
    for epoch in xrange(num_epochs):
        # Shuffle the data at each epoch
        print('\nEpoch: %d' % epoch)
        if shuffle:
            x_train, y_train = data_shuffle(x_train, y_train)

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield (x_train[start_index:end_index],
                   y_train[start_index:end_index])


# In[ ]:


def data_preparation():
    """Get train/dev data from file.

    Returns:
        x_train: A list. Each item is also a list corresponding a sentence.
        y_train: A list. Each item is a label. 
        x_dev: Like x_train.
        y_dev: Like y_train.
        vocab_size: Vocabulary size.
    """
    # Load data
    print('Loading data...')
    x_text, y = _load_data_and_labels()

    print('\n'.join(x_text[:10]))
    print()

    x_text, medical_entity_types = medical_to_common(x_text)

    print('\n'.join(x_text[:10]))

    for token in medical_entity_types:
        jieba.add_word(word=token, freq=10000)

    with open(
            os.path.join(FLAGS.train, TIMESTAMP, 'medical_entity_types.json'),
            'w+') as f:
        json.dump(medical_entity_types, f)

    # Build vocabulary
    max_document_length = max([len(tokens) for tokens in tokenizer(x_text)])
    print('Max document length: %d' % max_document_length)
    vocab_processor = learn.preprocessing.VocabularyProcessor(
        max_document_length=max_document_length, tokenizer_fn=tokenizer)

    x = np.array(list(vocab_processor.fit_transform(x_text)))

    # Write vocabulary
    vocab_size = len(vocab_processor.vocabulary_)

    vocab_processor.save(os.path.join(FLAGS.train, TIMESTAMP, 'vocab'))

    split_rate = list(map(int, FLAGS.split_rate.split(',')))

    split_rate = [float(value) / sum(split_rate) for value in split_rate]

    def one_label_data_split(x, y):
        split_first_index = int(split_rate[0] * float(len(y)))
        split_second_index = int(
            (split_rate[1] + split_rate[0]) * float(len(y)))

        one_label_x_train = x[:split_first_index]
        one_label_y_train = y[:split_first_index]
        one_label_x_validate = x[split_first_index:split_second_index]
        one_label_y_validate = y[split_first_index:split_second_index]
        one_label_x_test = x[split_second_index:]
        one_label_y_test = y[split_second_index:]
        return one_label_x_train, one_label_y_train, one_label_x_validate, one_label_y_validate, one_label_x_test, one_label_y_test

    def train_dev_test_split(x, y):
        x_train = np.array([])
        y_train = np.array([])
        x_validate = np.array([])
        y_validate = np.array([])
        x_test = np.array([])
        y_test = np.array([])

        begin = 0
        last_label = None
        for i in xrange(len(y)):
            if np.argmax(y[i]) != last_label and last_label is not None:
                one_label_x = x[begin:i]
                one_label_y = y[begin:i]

                one_label_x_train, one_label_y_train, one_label_x_validate, one_label_y_validate, one_label_x_test, one_label_y_test = one_label_data_split(
                    one_label_x, one_label_y)

                x_train = np.vstack(
                    [x_train,
                     one_label_x_train]) if x_train.size else one_label_x_train

                y_train = np.vstack(
                    [y_train,
                     one_label_y_train]) if y_train.size else one_label_y_train

                x_validate = np.vstack([
                    x_validate, one_label_x_validate
                ]) if x_validate.size else one_label_x_validate

                y_validate = np.vstack([
                    y_validate, one_label_y_validate
                ]) if y_validate.size else one_label_y_validate

                x_test = np.vstack(
                    [x_test,
                     one_label_x_test]) if x_test.size else one_label_x_test

                y_test = np.vstack(
                    [y_test,
                     one_label_y_test]) if y_test.size else one_label_y_test

                begin = i
            last_label = np.argmax(y[i])
        return x_train, y_train, x_validate, y_validate, x_test, y_test

    x_train, y_train, x_validate, y_validate, x_test, y_test = train_dev_test_split(
        x, y)

    x_train, y_train = data_shuffle(x_train, y_train)
    x_validate, y_validate = data_shuffle(x_validate, y_validate)
    x_test, y_test = data_shuffle(x_test, y_test)

    #     def slimming(x, y, rate):
    #         size = len(x)
    #         return x[:int(size * rate),:],y[:int(size * rate),:],

    #     x_train, y_train = slimming(x_train, y_train, 0.1)
    #     x_validate, y_validate = slimming(x_validate, y_validate, 0.1)
    #     x_test, y_test = slimming(x_test, y_test, 0.1)

    print('Vocabulary Size: {:d}'.format(vocab_size))
    print('Train/Validate/Test split: {:d}/{:d}/{:d}'.format(
        len(y_train), len(y_validate), len(y_test)))

    return x_train, y_train, x_validate, y_validate, x_test, y_test, vocab_size


# In[ ]:

# get_ipython().magic(u'pinfo csv.reader')
# get_ipython().magic(u'pinfo map')

# In[ ]:

# get_ipython().magic(u'pdb')

# In[ ]:

# Training
# ==================================================


def training():
    """Train the model.
    """
    x_train, y_train, x_validate, y_validate, x_test, y_test, vocab_size = data_preparation(
    )

    print('---------------------------------------------')
    print(x_train.shape)
    print(y_train.shape)
    print(x_validate.shape)
    print(y_validate.shape)
    print(x_test.shape)
    print(y_test.shape)
    print('---------------------------------------------')

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=vocab_size,
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(','))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name='global_step', trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(
                grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram(
                        '{}/grad/hist'.format(v.name), g)
                    sparsity_summary = tf.summary.scalar(
                        '{}/grad/sparsity'.format(v.name),
                        tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            out_dir = os.path.abspath(
                os.path.join(os.path.curdir, FLAGS.train, TIMESTAMP))
            print('Writing to {}\n'.format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar('loss', cnn.loss)
            acc_summary = tf.summary.scalar('accuracy', cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge(
                [loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, 'summaries', 'train')
            train_summary_writer = tf.train.SummaryWriter(train_summary_dir,
                                                          sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, 'summaries', 'dev')
            dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir,
                                                        sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists
            # so we need to create it
            checkpoint_dir = os.path.abspath(
                os.path.join(out_dir, 'checkpoints'))
            checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables())

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                '''
                A single training step
                '''
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run([
                    train_op, global_step, train_summary_op, cnn.loss,
                    cnn.accuracy
                ], feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print(
                    '%s: step %8d, loss %2.6f, acc %2.6f\r' %
                    (time_str, step, loss, accuracy),
                    end='')

            def validate_step(x, y, writer=None):
                '''
                Evaluates model on a dev set
                '''
                size = len(y)
                if size < FLAGS.batch_size:
                    raise ValueError(
                        "batch size for evals larger than dataset: %d" % size)

                accuracies = []
                for begin in xrange(0, size, FLAGS.batch_size):
                    end = begin + FLAGS.batch_size
                    if end < size:
                        feed_dict = {
                            cnn.input_x: x[begin:end, :],
                            cnn.input_y: y[begin:end, :],
                            cnn.dropout_keep_prob: 1.0
                        }
                    else:
                        feed_dict = {
                            cnn.input_x: x[-FLAGS.batch_size:, :],
                            cnn.input_y: y[-FLAGS.batch_size:, :],
                            cnn.dropout_keep_prob: 1.0
                        }

                    summaries, step, accuracy = sess.run(
                        [dev_summary_op, global_step, cnn.accuracy], feed_dict)

                    accuracies.append(accuracy)
                    if writer:
                        writer.add_summary(summaries, step)

                accuracy = sum(accuracies) / len(accuracies)
                time_str = datetime.datetime.now().isoformat()
                print('%s: step %8d, acc %2.6f' % (time_str, step, accuracy))

            def class_accuracy_stat(predictions, labels, all_data=False):

                predictions = predictions.tolist()
                labels = labels.tolist()

                with open(
                        os.path.join(FLAGS.train, TIMESTAMP,
                                     'label_to_int.json'), 'r') as f:
                    label_to_int = json.load(f)

                int_to_label = {
                    label_to_int[key]: key
                    for key in label_to_int.keys()
                }

                labels_stat = {}

                for i in xrange(len(labels)):
                    try:
                        labels_stat[int_to_label[labels[i]]][
                            'num'] = labels_stat[int_to_label[labels[i]]][
                                'num'] + 1 if labels_stat[int_to_label[labels[
                                    i]]]['num'] else 1
                    except:
                        labels_stat[int_to_label[labels[i]]] = {}
                        labels_stat[int_to_label[labels[i]]][
                            'num'] = labels_stat[int_to_label[labels[i]]][
                                'num'] + 1 if labels_stat[int_to_label[labels[
                                    i]]].has_key('num') else 1

                    if labels[i] == predictions[i]:
                        labels_stat[int_to_label[labels[i]]][
                            'predictions'] = labels_stat[int_to_label[labels[
                                i]]]['predictions'] + 1 if labels_stat[
                                    int_to_label[labels[i]]].has_key(
                                        'predictions') else 1

                for label in labels_stat:
                    if not labels_stat[label].has_key('predictions'):
                        labels_stat[label]['predictions'] = 0

                for label in labels_stat:
                    labels_stat[label]['accuracy'] = float(labels_stat[label][
                        'predictions']) / labels_stat[label]['num']

                if all_data:
                    with open(
                            os.path.join(FLAGS.train, TIMESTAMP,
                                         'all_labels_stat.json'), 'w+') as f:
                        json.dump(labels_stat, f)
                else:
                    with open(
                            os.path.join(FLAGS.train, TIMESTAMP,
                                         'test_labels_stat.json'), 'w+') as f:
                        json.dump(labels_stat, f)

            def test_step(x, y, writer=None, all_data=False):
                '''
                Evaluates model on a dev set
                '''
                size = len(y)

                accuracies = []
                all_predictions = []
                for begin in xrange(0, size, FLAGS.batch_size):
                    end = begin + FLAGS.batch_size
                    if end < size:
                        feed_dict = {
                            cnn.input_x: x[begin:end, :],
                            cnn.input_y: y[begin:end, :],
                            cnn.dropout_keep_prob: 1.0
                        }
                    else:
                        feed_dict = {
                            cnn.input_x: x[-FLAGS.batch_size:, :],
                            cnn.input_y: y[-FLAGS.batch_size:, :],
                            cnn.dropout_keep_prob: 1.0
                        }

                    summaries, step, accuracy, batch_predictions = sess.run([
                        dev_summary_op, global_step, cnn.accuracy,
                        cnn.predictions
                    ], feed_dict)

                    all_predictions = np.concatenate(
                        [all_predictions, batch_predictions])
                    accuracies.append(accuracy)
                    if writer:
                        writer.add_summary(summaries, step)

                class_accuracy_stat(
                    all_predictions, np.argmax(
                        y, axis=1), all_data)
                accuracy = sum(accuracies) / len(accuracies)
                time_str = datetime.datetime.now().isoformat()
                print('%s: step %8d, acc %2.6f' % (time_str, step, accuracy))

            # Generate batches
            batches = _batch_iter(x_train, y_train, FLAGS.batch_size,
                                  FLAGS.num_epochs)
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = batch
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print('\nValidation:')
                    validate_step(
                        x_validate, y_validate, writer=dev_summary_writer)
                    print('')
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(
                        sess, checkpoint_prefix, global_step=current_step)
                    print('Saved model checkpoint to {}\n'.format(path))

            print('\nTest:')
            test_step(x_test, y_test, writer=dev_summary_writer)
            test_step(
                np.concatenate(
                    (x_train, x_validate, x_test), axis=0),
                np.concatenate(
                    (y_train, y_validate, y_test), axis=0),
                writer=dev_summary_writer,
                all_data=True)
            print('')


# In[ ]:


def main(argv=None):
    training()


# In[ ]:

if __name__ == '__main__':
    tf.app.run()

# In[ ]:

# get_ipython().magic(u'pinfo np.concatenate')

# In[ ]:

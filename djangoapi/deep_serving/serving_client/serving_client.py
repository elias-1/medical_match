#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 www.drcubic.com, Inc. All Rights Reserved
#
"""
File: serving_client.py
Author: shileicao(shileicao@stu.xjtu.edu.cn)
Date: 2017/3/28 9:42
"""

import os
import sys
import jieba
import numpy as np
import predict_pb2
import prediction_service_pb2
import tensor_pb2
import tensor_shape_pb2
from grpc.beta import implementations
from ..utils.utils import config

reload(sys)
sys.setdefaultencoding('utf8')

app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_file_dir = os.path.join(app_dir, 'config', 'config.conf')
serving_params = config(filename=config_file_dir, section='deep_serving')

data_dir = os.path.join(app_dir, 'data')

NER_MAX_SENTENCE_LEN = serving_params['NER_MAX_SENTENCE_LEN']
trainsMatrix = np.load(os.path.join(data_dir, serving_params['trainsMatrix']))
ner_server = serving_params['ner_server']
clfier_server = serving_params['clfier_server']
ner_char2vec_path = os.path.join(data_dir, serving_params['ner_char2vec_path'])
UNK = serving_params['UNK']
ENTITY_TYPES = serving_params['ENTITY_TYPES'].split(',')

clfier_word2vec_path = os.path.join(data_dir,
                                    serving_params['clfier_word2vec_path'])
clfier_char2vec_path = os.path.join(data_dir,
                                    serving_params['clfier_char2vec_path'])
C_MAX_SENTENCE_LEN = serving_params['C_MAX_SENTENCE_LEN']
C_MAX_WORD_LEN = serving_params['C_MAX_WORD_LEN']


def _tokenizer(sentence):
    return jieba.lcut(sentence, cut_all=False)


def _get_vob(vob_path):
    vob = []
    with open(vob_path, 'r') as f:
        f.readline()
        for row in f.readlines():
            vob.append(row.split()[0].decode('utf-8'))
    return vob


def _predict_tensor_proto(chari, predict_shape):
    # dtype 7 is String
    tensor_proto = tensor_pb2.TensorProto(
        dtype=9, tensor_shape=predict_shape())
    tensor_proto.int64_val.extend(chari)

    return tensor_proto


class Ner(object):
    def __init__(self,
                 tfserving_host=ner_server.split(':')[0],
                 tfserving_port=int(ner_server.split(':')[1])):
        self.tfserving_host = tfserving_host
        self.tfserving_port = tfserving_port

        # setup grcp channel
        self.channel = implementations.insecure_channel(self.tfserving_host,
                                                        self.tfserving_port)

        # setup grpc prediction stub for tfserving
        self.stub = prediction_service_pb2.beta_create_PredictionService_stub(
            self.channel)
        self.char_vob = _get_vob(ner_char2vec_path)

    def _predict_shape(self):
        return tensor_shape_pb2.TensorShapeProto(dim=[
            tensor_shape_pb2.TensorShapeProto.Dim(size=1),
            tensor_shape_pb2.TensorShapeProto.Dim(size=NER_MAX_SENTENCE_LEN)
        ])

    def _predict_request(self, chari):
        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'ner'
        request.model_spec.signature_name = 'predict_sentence'
        request.inputs['words'].CopyFrom(
            _predict_tensor_proto(chari, self._predict_shape))
        return request

    def _viterbi_decode(self, score, transition_params):
        """Decode the highest scoring sequence of tags outside of TensorFlow.
        This should only be used at test time.
        Args:
            score: A [seq_len, num_tags] matrix of unary potentials.
            transition_params: A [num_tags, num_tags] matrix of binary potentials.
        Returns:
            viterbi: A [seq_len] list of integers containing the highest scoring tag
                indicies.
            viterbi_score: A float containing the score for the Viterbi sequence.
        """
        trellis = np.zeros_like(score)
        backpointers = np.zeros_like(score, dtype=np.int32)
        trellis[0] = score[0]

        for t in range(1, score.shape[0]):
            v = np.expand_dims(trellis[t - 1], 1) + transition_params
            trellis[t] = score[t] + np.max(v, 0)
            backpointers[t] = np.argmax(v, 0)

        viterbi = [np.argmax(trellis[-1])]
        for bp in reversed(backpointers[1:]):
            viterbi.append(bp[viterbi[-1]])
        viterbi.reverse()

        viterbi_score = np.max(trellis[-1])
        return viterbi, viterbi_score

    def _decode_entity_location(self, entity_info):
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
        return entity_location, types_id

    def _process_line(self, x_text):
        nl = len(x_text)
        chari = []
        if nl > NER_MAX_SENTENCE_LEN:
            nl = NER_MAX_SENTENCE_LEN
        for ti in range(nl):
            char = x_text[ti]
            try:
                idx = self.char_vob.index(char)
            except ValueError:
                idx = self.char_vob.index(UNK)
            chari.append(str(idx))
        for i in range(nl, NER_MAX_SENTENCE_LEN):
            chari.append("0")

        return chari

    def __call__(self, sentence):
        chari = self._process_line(sentence)
        chari = map(int, np.array(chari, np.int64))
        # make a prediction request
        request = self._predict_request(chari)
        # send it to the server, with a 60 second timeout
        result = self.stub.Predict(request, 10.0)

        unary_score = np.array(result.outputs['scores'].float_val)
        shape = list(result.outputs['scores'].tensor_shape.dim)
        unary_score = np.reshape(unary_score,
                                 (int(shape[1].size), int(shape[2].size)))
        seq_len = int(result.outputs['sequence_length'].int_val[0])

        tf_unary_scores_ = unary_score[:seq_len]

        viterbi_sequence, _ = self._viterbi_decode(tf_unary_scores_,
                                                   trainsMatrix)

        entity_location, types_id = self._decode_entity_location(
            viterbi_sequence)
        entity_result = []
        type_result = []
        for loc, type_id in zip(entity_location, types_id):
            entity = sentence[loc[0]:loc[1] + 1]
            entity = entity.replace(',', '，')
            entities = entity.strip('，').split('，')
            entity_result.extend(entities)
            type_result.extend([ENTITY_TYPES[type_id]] * len(entities))

        return entity_result, type_result


class Clfier(object):
    def __init__(self,
                 tfserving_host=clfier_server.split(':')[0],
                 tfserving_port=int(clfier_server.split(':')[1])):
        self.tfserving_host = tfserving_host
        self.tfserving_port = tfserving_port

        # setup grcp channel
        self.channel = implementations.insecure_channel(self.tfserving_host,
                                                        self.tfserving_port)

        # setup grpc prediction stub for tfserving
        self.stub = prediction_service_pb2.beta_create_PredictionService_stub(
            self.channel)

        self.word_vob = _get_vob(clfier_word2vec_path)
        self.char_vob = _get_vob(clfier_char2vec_path)

    def _create_predict_shape(self, shape):
        def _predict_shape():
            dim = []
            for s in shape:
                dim.append(tensor_shape_pb2.TensorShapeProto.Dim(size=s))
            return tensor_shape_pb2.TensorShapeProto(dim=dim)

        return _predict_shape

    def _predict_request(self, wordi, chari):
        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'clfier'
        request.model_spec.signature_name = 'predict_sentence'
        predict_shape_word = [1, C_MAX_SENTENCE_LEN]
        predict_shape_char = [1, C_MAX_SENTENCE_LEN * C_MAX_WORD_LEN]
        request.inputs['words'].CopyFrom(
            _predict_tensor_proto(
                wordi, self._create_predict_shape(predict_shape_word)))
        request.inputs['chars'].CopyFrom(
            _predict_tensor_proto(
                chari, self._create_predict_shape(predict_shape_char)))
        return request

    def _process_line(self, x_text):
        words = _tokenizer(x_text)
        nl = len(words)
        wordi = []
        chari = []
        if nl > C_MAX_SENTENCE_LEN:
            nl = C_MAX_SENTENCE_LEN
        for ti in range(nl):
            word = words[ti]
            try:
                word_idx = self.word_vob.index(word)
            except ValueError:
                word_idx = self.word_vob.index(UNK)

            wordi.append(str(word_idx))
            chars = list(word)
            nc = len(chars)
            if nc > C_MAX_WORD_LEN:
                lc = chars[nc - 1]
                chars[C_MAX_WORD_LEN - 1] = lc
                nc = C_MAX_WORD_LEN
            for i in range(nc):
                try:
                    char_idx = self.char_vob.index(chars[i])
                except ValueError:
                    char_idx = self.char_vob.index(UNK)
                chari.append(str(char_idx))
            for i in range(nc, C_MAX_WORD_LEN):
                chari.append("0")
        for i in range(nl, C_MAX_SENTENCE_LEN):
            wordi.append("0")
            for ii in range(C_MAX_WORD_LEN):
                chari.append('0')
        return wordi, chari

    def __call__(self, sentence):
        wordi, chari = self._process_line(sentence)
        wordi = map(int, wordi)
        chari = map(int, chari)

        # make a prediction request
        request = self._predict_request(
            np.array(wordi, np.int64), np.array(chari, np.int64))
        # send it to the server, with a 60 second timeout
        result = self.stub.Predict(request, 10.0)

        response = result.outputs['classes'].string_val
        prediction = int(response[0])

        return prediction + 1


def main():
    ner = Ner()
    clfier = Clfier()
    sentence_list = [
        u'感冒吃什么药', u'头部挺疼的，是怎么了', u'大便很稀，赖床咋办啊？', u'头痛，喉咙痒是不是发烧了？'
    ]
    for sentence in sentence_list:
        prediction = clfier(sentence)
        entity_result, type_result = ner(sentence)
        entity_with_type = map(lambda x, y: x + '/' + y, entity_result,
                               type_result)
        print('sentence: %s, class: %d, entity_with_type: %s' %
              (sentence.encode('utf-8'), prediction,
               ' '.join(entity_with_type).encode('utf-8')))


if __name__ == '__main__':
    main()

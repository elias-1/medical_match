import json
import traceback

from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt

from .es_match.entity_refine import entity_refine
from .postgresql_kg.kg_utils import *
from .serving_client.serving_client import Clfier, Ner

ner = Ner()
clfier = Clfier()


@csrf_exempt
def sentence_clfier(request):
    if request.method == "GET":
        json_out = {}
        try:
            input_dict = json.loads(request.GET["q"])
            sentence = input_dict['sentence']
            prediction = clfier(sentence)
            json_out['class'] = prediction
            json_out["Return"] = 0
        except:
            traceback.print_exc()
            json_out["Return"] = 1
        return HttpResponse(
            json.dumps(json_out), content_type="application/json")


@csrf_exempt
def sentence_ner(request):
    if request.method == "GET":
        json_out = {}
        try:
            input_dict = json.loads(request.GET["q"])
            sentence = input_dict['sentence']
            entity_result, type_result = ner(sentence)
            json_out['entities'] = entity_result
            json_out['types'] = type_result
            json_out["Return"] = 0
        except:
            traceback.print_exc()
            json_out["Return"] = 1
        return HttpResponse(
            json.dumps(json_out), content_type="application/json")


@csrf_exempt
def sentence_clfier_ner(request):
    if request.method == "GET":
        json_out = {}
        try:
            input_dict = json.loads(request.GET["q"])
            sentence = input_dict['sentence']

            prediction = clfier(sentence)
            json_out['class'] = prediction

            entity_result, type_result = ner(sentence)
            json_out['entities'] = entity_result
            json_out['types'] = type_result
            json_out["Return"] = 0
        except:
            traceback.print_exc()
            json_out["Return"] = 1
        return HttpResponse(
            json.dumps(json_out), content_type="application/json")


@csrf_exempt
def sentence_ner_es(request):
    if request.method == "GET":
        json_out = {}
        try:
            input_dict = json.loads(request.GET["q"])
            sentence = input_dict['sentence']
            entity_result, type_result = ner(sentence)
            json_out['entities'], json_out['types'] = entity_refine(
                entity_result, type_result)
            json_out["Return"] = 0
        except:
            traceback.print_exc()
            json_out["Return"] = 1
        return HttpResponse(
            json.dumps(json_out), content_type="application/json")


@csrf_exempt
def sentence_process(request):
    """ 处理前端传过来的句子
    Args:
        request: request
    Return:
        json_dict: 
            {
                result: result,
                flag: 0(白话) 1 (正常返回) 2(疾病/症状问药打架式) 3(症状问诊及相应科室) 4 (概述)
            }
        
    """
    if request.method == "GET":
        json_out = {}
        try:
            input_dict = json.loads(request.GET["q"])
            sentence = input_dict['sentence']

            if len(sentence) <= 4:
                identify_result = kg_entity_identify(sentence)
                if identify_result['success']:
                    json['result'] = identify_result['result']
                    json['flag'] = 4
                    json_out["Return"] = 0
                    return HttpResponse(
                        json.dumps(json_out), content_type="application/json")
                else:
                    json['flag'] = 0
                    json_out["Return"] = 0
                    return HttpResponse(
                        json.dumps(json_out), content_type="application/json")

            entity_result, type_result = ner(sentence)
            entities, types = entity_refine(entity_result, type_result)
            if not entity_result:
                json['flag'] = 0
                json_out["Return"] = 0
                return HttpResponse(
                    json.dumps(json_out), content_type="application/json")

            prediction = clfier(sentence)
            if prediction == 0:
                json['flag'] = 2
                json_out["Return"] = 0
                return HttpResponse(
                    json.dumps(json_out), content_type="application/json")
            elif prediction == 8 or prediction == 3:
                json['flag'] = 3
                json_out["Return"] = 0
                return HttpResponse(
                    json.dumps(json_out), content_type="application/json")
            elif prediction == '7':
                json['result'] = kg_entity_summary(entities)
                json['flag'] = 4
                json_out["Return"] = 0
                return HttpResponse(
                    json.dumps(json_out), content_type="application/json")

            # TODO: add kg support
            # json['result'] =
            json['flag'] = 1
            json_out["Return"] = 0
        except:
            traceback.print_exc()
            json_out["Return"] = 1
        return HttpResponse(
            json.dumps(json_out), content_type="application/json")

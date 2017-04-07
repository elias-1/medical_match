import json
import traceback

from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt

from .es_match.entity_refine import entity_refine
from .postgresql_kg.kg_utils import kg_entity_identify
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
            json_out['entities'] = entity_refine(entity_result)
            json_out['types'] = type_result
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
                flag: 0(白话) 1 (概述) 2() 3()
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
                    json['flag'] = 1
                    json_out["Return"] = 0
                    return HttpResponse(
                        json.dumps(json_out), content_type="application/json")
                else:
                    json['flag'] = 0
                    json_out["Return"] = 0
                    return HttpResponse(
                        json.dumps(json_out), content_type="application/json")

            entity_result, type_result = ner(sentence)
            if not entity_result:
                json['flag'] = 0
                json_out["Return"] = 0
                return HttpResponse(
                    json.dumps(json_out), content_type="application/json")

            json_out['entities'] = entity_refine(entity_result)
            json_out['types'] = type_result
            json_out["Return"] = 0
        except:
            traceback.print_exc()
            json_out["Return"] = 1
        return HttpResponse(
            json.dumps(json_out), content_type="application/json")

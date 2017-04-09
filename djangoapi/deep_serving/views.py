import json
import traceback

from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt

from .es_match.entity_refine import entity_refine
from .postgresql_kg.kg_utils import *
from .serving_client.serving_client import Clfier, Ner
from .simple_qa.simple_query import simple_qa

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
        flag: 0 白话 
              1 疾病/症状问药打架式 
              2 药品对比、药品配伍  
                    额外返回result: 
              3 疾病相关疾病、药品适应证、检查相关疾病、手术相关疾病   症状问诊打架式
                    额外返回result: 
              4 疾病的相关症状、药品适应症状、症状的相关症状、检查的相关症状
                    额外返回result: 
              5 疾病的相关检查、症状的相关检查
                    额外返回result: 
              6 疾病的相关手术
                    额外返回result:
              7 手术部位、检查的部位、症状的部位、疾病的部位
                    额外返回result: 
              8 症状的科室、疾病的科室 
                    额外返回result: 
              9 症状、疾病、药品、检查、手术的概述
                    额外返回result: list
              10 疾病、检查、手术的价格 
                    额外返回result:
              11 药品用量
                    额外返回result: 
        
    """
    if request.method == "GET":
        json_out = {}
        try:
            input_dict = json.loads(request.GET["q"])
            sentence = input_dict['sentence']
            json_out["Return"] = 0
            if len(sentence) <= 4:
                identify_result = kg_entity_identify(sentence)
                if identify_result['success']:
                    json_out['result'] = identify_result['result']
                    json_out['flag'] = 9
                else:
                    json_out['flag'] = 0
                return HttpResponse(
                    json.dumps(json_out), content_type="application/json")

            entity_result, type_result = ner(sentence)
            entities, types = entity_refine(entity_result, type_result)
            if not entity_result:
                json_out['flag'] = 0
                return HttpResponse(
                    json.dumps(json_out), content_type="application/json")

            prediction = clfier(sentence)
            json_out['flag'] = prediction
            if prediction == 1 or prediction == 3:
                pass

            elif prediction == 7:
                # TODO implementation
                body_part, success = kg_search_body_part(entities)
                if success:
                    json_out['result'] = body_part
                else:
                    json_out['Return'] = 2
                    json_out['result'] = [u'没能找到%s的相应部位' % u'、'.join(entities)]

            elif prediction == 8:
                # TODO implementation
                department, success = kg_search_department(entities)
                if success:
                    json_out['result'] = department
                else:
                    json_out['Return'] = 2
                    json_out['result'] = [u'没能找到%s的相应科室' % u'、'.join(entities)]

            elif prediction == 9:
                entitiy_summarys, success = kg_entity_summary(entities)
                if success:
                    json_out['result'] = entitiy_summarys
                else:
                    json_out['Return'] = 2
                    json_out['result'] = [u'没能找到%s的概述' % u'、'.join(entities)]
                return HttpResponse(
                    json.dumps(json_out), content_type="application/json")

            elif prediction == 10:
                # TODO implementation
                price, success = kg_search_price(entities)
                if success:
                    json_out['result'] = price
                else:
                    json_out['Return'] = 2
                    json_out['result'] = u'没能找到%s的价格' % u'、'.join(entities)

            else:
                json_result = simple_qa(entities, prediction)
                json_out['result'] = json_result['content']
                if json_result['return'] == 1:
                    json_out['Return'] = 2
            return HttpResponse(
                json.dumps(json_out), content_type="application/json")
        except:
            traceback.print_exc()
            json_out["Return"] = 1
        return HttpResponse(
            json.dumps(json_out), content_type="application/json")

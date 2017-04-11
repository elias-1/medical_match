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
import json
import traceback
from collections import OrderedDict

from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from elasticsearch import Elasticsearch
from fuzzywuzzy import fuzz

from .es_match.entity_refine import entity_refine
from .interactive_query.interactive_query import *
from .postgresql_kg import kg_utils
from .serving_client.serving_client import Clfier, Ner
from .simple_qa import simple_query

ner = Ner()
clfier = Clfier()

app_dir = os.path.dirname(os.path.abspath(__file__))
config_file_dir = os.path.join(app_dir, 'config', 'config.conf')
params = config(filename=config_file_dir, section='elasticsearch')
es = Elasticsearch(**params)


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
                identify_result = kg_utils.kg_entity_identify(sentence)
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
                body_part, success = kg_utils.kg_search_body_part(entities)
                if success:
                    json_out['result'] = body_part
                else:
                    json_out['Return'] = 2
                    json_out['result'] = [u'没能找到%s的相应部位' % u'、'.join(entities)]

            elif prediction == 8:
                department, success = kg_utils.kg_search_department(entities)
                if success:
                    json_out['result'] = department
                else:
                    json_out['Return'] = 2
                    json_out['result'] = [u'没能找到%s的相应科室' % u'、'.join(entities)]

            elif prediction == 9:
                entitiy_summarys, success = kg_utils.kg_entity_summary(
                    entities)
                if success:
                    json_out['result'] = entitiy_summarys
                else:
                    json_out['Return'] = 2
                    json_out['result'] = [u'没能找到%s的概述' % u'、'.join(entities)]
                return HttpResponse(
                    json.dumps(json_out), content_type="application/json")

            elif prediction == 10:
                price, success = kg_utils.kg_search_price(entities)
                if success:
                    json_out['result'] = price
                else:
                    json_out['Return'] = 2
                    json_out['result'] = u'没能找到%s的价格' % u'、'.join(entities)

            else:
                json_result = simple_query.simple_qa(entities, prediction)
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


# ycc
@csrf_exempt
def get_symptom_id(request):  # get symptom id
    """
    功能:
        返回症状名字对应的ID，采用ES模糊匹配

    示例:
        http://yiliao1:9999/kgInterface/getSymId/?q={%22Name%22:%22%E5%A4%B4%E7%97%9B%22}

    输入:
        Name:症状名,字符串
    返回:
        Results: list:
            Id:  症状ID
            Name: 症状名字
    """
    if request.method == "GET":
        json_out = {}
        try:
            input_dict = json.loads(request.GET["q"])
            sname = input_dict['Name']
            query_size = 200
            es = Elasticsearch()
            res = es.search(
                index='medknowledge',
                doc_type='search',
                body={
                    'size': query_size,
                    'query': {
                        "query_string": {
                            'fields': ['Name', 'Ename', 'Oname'],
                            "query": sname
                        }
                    }
                })
            answers = res['hits']['hits']
            results = []
            for i, answer in enumerate(answers):
                d = answer['_source']
                try:
                    xid = d['Sid']
                    xname = d['Name']
                    if len(xid):
                        results.append({'Id': xid, 'Name': xname})
                except:
                    continue
            json_out["Return"] = 0
            json_out["Results"] = results[:20]
        except:
            traceback.print_exc()
            json_out["Return"] = 1
        return HttpResponse(
            json.dumps(json_out), content_type="application/json")


# ycc
@csrf_exempt
def get_symptom_disease(request):  # get possible disease of symptom 
    """
    功能:
        返回症状的可能疾病,使用json文件存储的数据,应该改成rdf3x查询(速度太慢)

    示例:
        http://yiliao1:9999/kgInterface/getSymDis/?q={%22Ids%22:[%22s15670%22],%22NotIds%22:[],%22UnknownIds%22:[]}

    输入:
        Ids: 患者已有的症状
        NotIds: 患者没有的症状
        UnknowIds: 患者不确定是否含有的症状
    返回:
        Results: dict:
            PosDep: 可能科室,list:
                disease: 该科室对应的疾病ID
                Id: 科室ID
                Name: 科室名字
            PosDis: 可能疾病,list:
                department: 该疾病对应的科室ID
                Id: 疾病ID
                Name: 疾病名字
            PosSym: 是否还有如下症状,list:
                Id: 症状ID
                Name: 症状名字
    """
    if request.method == "GET":
        json_out = {}
        try:
            input_dict = json.loads(request.GET["q"])
            fids = input_dict['Ids']
            sids = set(fids)
            not_sids = set(input_dict['NotIds'])
            unknown_sids = set(input_dict['UnknownIds'])
            nodetype1 = 'symptom'
            nodetype2 = 'disease'
            json_out["Results"] = {}
            posDis = get_fids_to_nodetype2list(fids, nodetype1, nodetype2)
            posSym, didname_dict = get_disease_symptom(posDis, sids, not_sids,
                                                       unknown_sids)

            posdisset = set(didname_dict)
            if len(fids) > 1:
                for fid in fids:
                    otherDis = get_fids_to_nodetype2list(
                        set(fids) - set([fid]), nodetype1, nodetype2)
                    for d in otherDis:
                        xid = d['Id']
                        name = d['Name']
                        if xid not in didname_dict:
                            didname_dict[xid] = '________' + name

            posdislist = sorted(
                posdisset, key=lambda s: id_degree_dict[s], reverse=True)
            posDis = get_dis_list(posdislist)

            moredisset = set(didname_dict) - set(posdisset)
            moredislist = sorted(
                moredisset, key=lambda s: id_degree_dict[s], reverse=True)
            moreDis = get_dis_list(moredislist, '-----')

            json_out["Results"]['PosSym'] = posSym
            json_out["Results"]['PosDis'] = posDis + moreDis
            json_out["Results"]['PosDep'] = get_dep_dis_list(posDis + moreDis)
            json_out["Return"] = 0
        except:
            traceback.print_exc()
            json_out["Return"] = 1
        return HttpResponse(
            json.dumps(json_out), content_type="application/json")


# ycc
@csrf_exempt
def get_symptom_id_2(request):  # get symptom id to find medicine
    """
    功能:
        根据症状/疾病,询问药品
        返回症状/疾病名字对应的ID，读取json文件,采用fuzz做模糊匹配

    示例:
        http://yiliao1:9999/kgInterface/getSymId2/?q={%22Name%22:%22%E5%A4%B4%E7%97%9B%22}

    输入:
        Name:症状名,字符串
    返回:
        Results: list:
            Id:  症状/疾病ID
            Name: 症状/疾病名字
    """
    if request.method == "GET":
        json_out = {}
        try:
            input_dict = json.loads(request.GET["q"])
            sname = input_dict['Name']
            name_score_dict = {
                name: fuzz.partial_ratio(name, sname)
                for name in symptom_name_id_dict
            }
            name_list = sorted(
                name_score_dict.keys(),
                key=lambda name: name_score_dict[name],
                reverse=True)
            results = [{
                'Name': name,
                'Id': symptom_name_id_dict[name]
            } for name in name_list
                       if symptom_name_id_dict[name] in sid_midlist_dict]
            json_out["Return"] = 0
            json_out["num"] = len(symptom_name_id_dict)
            json_out["Results"] = results[:20]
        except:
            traceback.print_exc()
            json_out["Return"] = 1
        return HttpResponse(
            json.dumps(json_out), content_type="application/json")


# ycc
@csrf_exempt
def get_symptom_medcine(request):  # get possible disease of symptom 
    """
    功能:
        返回症状/疾病的推荐药品,使用json文件存储的数据,应该改成rdf3x查询(速度太慢)

    示例:
        http://1.85.37.136:9999/kgInterface/getSymMed/?q={%22Sids%22:[%22s12097%22,%22s15063%22],%22NotSids%22:[],%22Tids%22:[],%22NotTids%22:[],%22Age%22:%22%22}

    输入:
        Sids: 患者已有的症状
        NotSids: 患者没有的症状
        Tids: 患者已有的禁忌症
        NotTids: 患者没有的禁忌症
        Age: 患者年龄
    返回:
        Results: dict:
            PosMed: 可能疾病,list:
                Id: 药品ID
                Name: 药品名字
            PosSym: 是否还有如下症状,list:
                Id: 症状ID
                Name: 症状名字
            PosTaboo: 是否还有如下禁忌症,list:
                Id: 禁忌症ID
                Name: 禁忌症名字
    """
    if request.method == "GET":
        json_out = {}
        try:
            input_dict = json.loads(request.GET["q"])
            sids = input_dict['Sids']
            sids_not = input_dict['NotSids']
            tids = input_dict['Tids']
            tids_not = input_dict['NotTids']
            age = input_dict['Age']

            midset = set()
            for sid in sids:
                for mid in sid_midlist_dict[sid]:
                    if len(set(mid_tidlist_dict[mid]) & set(tids)) == 0:
                        midset.add(mid)
            posMed = []
            for mid in midset:
                mname = medicine_id_name_dict[mid]
                posMed.append({'Id': mid, 'Name': mname})

            tid_num_dict = {}
            for med in posMed:
                mid = med['Id']
                for tid in mid_tidlist_dict[mid]:
                    try:
                        tid_num_dict[tid] += 1
                    except:
                        tid_num_dict[tid] = 1
            tid_list = sorted(
                tid_num_dict.keys(),
                key=lambda tid: abs(tid_num_dict[tid] * 2 - len(posMed)))
            tid_list = [
                tid for tid in tid_list if tid_num_dict[tid] != len(posMed)
            ]
            posTaboo = [{
                'Id': tid,
                'Name': taboo_id_name_dict[tid]
            } for tid in tid_list]

            sidset = set()
            for med in posMed:
                mid = med['Id']
                for sid in mid_sidlist_dict[mid]:
                    sidset.add(sid)
            sidset = sidset - set(sids)
            posSym = [{
                'Id': sid,
                'Name': symptom_id_name_dict[sid]
            } for sid in sidset]

            json_out["Results"] = OrderedDict()
            num = 20

            posMedNew = []
            mednameset = set()
            for med in posMed:
                medname = med['Name']
                if medname not in mednameset:
                    mednameset.add(medname)
                    posMedNew.append(med)
            json_out["Results"]['PosMed'] = posMedNew
            json_out["Results"]['PosSym'] = posSym[:10]
            json_out["Results"]['PosTaboo'] = posTaboo[:10]
            json_out["Return"] = 0
        except:
            traceback.print_exc()
            json_out["Return"] = 1
        return HttpResponse(
            json.dumps(json_out), content_type="application/json")

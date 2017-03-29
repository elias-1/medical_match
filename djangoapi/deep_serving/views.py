import json
import traceback

from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt

from .serving_client.serving_client import Clfier, Ner

ner = Ner()
clfier = Clfier()


@csrf_exempt
def sentence_clfier(request):
    if request.method == "GET":
        json_out = {}
        try:
            input_dict = json.loads(request.GET["q"])
            sentence = input_dict('sentence')
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
            sentence = input_dict('sentence')
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
            sentence = input_dict('sentence')

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

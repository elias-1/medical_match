import json
import traceback
from datetime import datetime

from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt

# from StringIO import StringIO


@csrf_exempt
def sentence_clfier(request):
    if request.method == "GET":
        json_out = {}
        try:
            input_dict = json.loads(request.GET["q"])

        except:
            traceback.print_exc()
            json_out["Return"] = 1
        return HttpResponse(
            json.dumps(json_out), content_type="application/json")


@csrf_exempt
def news_latest(request):
    if request.method == "GET":
        try:
            news_count = News.objects.count()
            news_info = News.objects.order_by("-UpdateTime")[0:min(news_count,
                                                                   100)]
            json_out = []
            for temp_news in news_info:
                if temp_news.Content:
                    json_out = json_out + split_text(
                        w.get_file(temp_news.Content))
            items2remove = []
            for i in json_out:
                if len(i) < 4 or i.isdigit():
                    items2remove.append(i)
            for j in items2remove:
                json_out.remove(j)
        except:
            json_out = {}
            traceback.print_exc()
            json_out["Return"] = 1
        return HttpResponse(
            json.dumps(json_out), content_type="application/json")


def news_op(request):
    if request.method == "GET":
        json_out = {}
        try:
            input_dict = json.loads(request.GET["q"])
            nid = int(input_dict["NewsID"])
            news_info = News.objects.filter(NewsID=nid)
            if news_info:
                news_info = news_info.values()[0]
                for key in news_info.keys():
                    if news_info[key] not in useless_list:
                        if key in datetime_list:
                            json_out[key] = news_info[key].isoformat()
                        elif key == "Content":
                            if news_info[key]:
                                json_out[key] = w.get_file(news_info[key])
                        else:
                            json_out[key] = news_info[key]
                json_out["Return"] = 0
            else:
                json_out["Return"] = 1
        except:
            traceback.print_exc()
            json_out["Return"] = 1
        return HttpResponse(
            json.dumps(json_out), content_type="application/json")

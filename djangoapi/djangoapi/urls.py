"""djangoapi URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.8/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Add an import:  from blog import urls as blog_urls
    2. Add a URL to urlpatterns:  url(r'^blog/', include(blog_urls))
"""
from django.conf.urls import include, url
# from django.contrib import admin

urlpatterns = [
    # url(r'^admin/', include(admin.site.urls)),
    url(r'^qa/property/$', 'qa.views.property_op'),
    url(r'^qa/relation/$', 'qa.views.relation_op'),
    url(r'^qa/sentence_clfier/$', 'qa.views.sentence_clfier'),
    url(r'^qa/sentence_ner/$', 'qa.views.sentence_ner'),
    url(r'^qa/sentence_clfier_ner/$', 'qa.views.sentence_clfier_ner'),
    url(r'^qa/sentence_ner_es/$', 'qa.views.sentence_ner_es'),
    url(r'^qa/sentence_process/$', 'qa.views.sentence_process'),
    url(r'^qa/getSymId/$', 'qa.views.get_symptom_id'),
    url(r'^qa/getSymDis/$', 'qa.views.get_symptom_disease'),
    url(r'^qa/getSymId2/$', 'qa.views.get_symptom_id_2'),
    url(r'^qa/getSymMed/$', 'qa.views.get_symptom_medcine'),
]

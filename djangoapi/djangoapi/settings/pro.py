#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 www.drcubic.com, Inc. All Rights Reserved
#
"""
File: pro.py
Author: shileicao(shileicao@stu.xjtu.edu.cn)
Date: 2017/4/12 19:26
python manage.py migrate –-settings=djangoapi.settings.pro
"""
from .base import *

DEBUG = False

# ADMINS = (
#     ('Antonio M', 'antonio.mele@zenxit.com'),
# )

ALLOWED_HOSTS = ['*']

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': 'kgdata',
        'USER': 'dbuser',
        'PASSWORD': '112233',
    }
}

#SECURE_SSL_REDIRECT = True
#CSRF_COOKIE_SECURE = True

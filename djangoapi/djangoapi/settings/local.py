#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 www.drcubic.com, Inc. All Rights Reserved
#
"""
File: local.py
Author: shileicao(shileicao@stu.xjtu.edu.cn)
Date: 2017/4/12 19:24
"""

from .base import *

DEBUG = True

# DATABASES = {
#     'default': {
#         'ENGINE': 'django.db.backends.sqlite3',
#         'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),
#     }
# }

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': 'kgdata',
        'USER': 'dbuser',
        'PASSWORD': '112233',
    },
    # 'postgresql': {
    #     'ENGINE': 'django.db.backends.postgresql_psycopg2',
    #     'NAME': 'kgdata',
    #     'USER': 'dbuser',
    #     'PASSWORD': '112233',
    # },
}

DATABASE_ROUTERS = ['djangoapi.database_router.DatabaseAppsRouter']
DATABASE_APPS_MAPPING = {
    'qa': 'postgresql',
}

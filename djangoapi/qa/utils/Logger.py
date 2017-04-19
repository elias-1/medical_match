#!/usr/bin/python
# -*- coding: utf-8 -*-

import logging.config
import os
import traceback

import Singleton

app_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(app_dir, 'log')
logger_config_dir = os.path.join(app_dir, 'config', 'logger.conf')

if not os.path.exists(log_dir):
    os.makedirs(log_dir)


class Logger(Singleton.Singleton):
    logger = None

    def __init__(self):
        if self.logger is None:
            try:
                logging.config.fileConfig(logger_config_dir)
                self.logger = logging.getLogger('exportLog')
            except:
                traceback.print_exc()

    def getLogger(self):
        return self.logger

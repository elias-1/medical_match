#!/usr/bin/python
# -*- coding: utf-8 -*-

import logging.config
import os
import traceback

import Singleton

app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
logger_config_dir = os.path.join(app_dir, 'config', 'logger.conf')


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

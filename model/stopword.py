#!/usr/bin/env python
# -*- coding:utf8 -*-
# Created by jinfen @ Dec 27, 2017
# Modified by jinfen @ Dec 27, 2017
import os
import codecs

from config import path_config

STOPWORD_DIR = os.path.join(path_config.DATA_DIR, 'wordbag','stopwords.txt')


def stopword():
    file = codecs.open(STOPWORD_DIR,encoding='utf-8')
    stopwords = []
    for line in file:
        stopwords.append(line.rstrip())
    return stopwords
#!/usr/bin/env python
# -*- coding:utf8 -*-
# Created by jinfen @ Dec 27, 2017
# Modified by jinfen @ Dec 28, 2017

import sys
import os
import logging
import numpy as np
import pandas as pd

from utils import misc
from config import path_config

FEEDBACK_DIR = os.path.join(os.path.join(path_config.FEEDBACK_DIR,'%s_%s_predict.npz' % (sys.argv[1], sys.argv[2])))

def main():
    result = np.load(FEEDBACK_DIR)
    total_data = result['total_data']
    accuracy = result['accuracy']
    score = result['score']
    columnname = ['Unnamed: 0', 'Description', 'Downloads', 'Languages', 'Name',
                  'NeedNet', 'Package', 'SuggestScore', 'Type', 'UpdateDescription',
                  'appid', 'cate_type', 'developers', 'evaluat_desc', 'game_feature',
                  'game_id', 'game_key', 'game_play_way', 'grade', 'is_enable', 'main_play',
                  'mix_server', 'pingce_score', 'platform', 'play_key', 'recommend', 'shortDesc',
                  'sociaty_attribute', 'subject', 'zhcn_name', 'pred']
    total_data = pd.DataFrame(total_data, columns=columnname)
    print("accuracy", accuracy)
    print("score(grade_,precision, recallscore, f1score)", score)
    #print("totaldata", total_data)
    return accuracy, score, total_data


if __name__ == "__main__":
    misc.SetupLogger()

    script_name = sys.argv[0]
    logging.info("%s starts" % script_name)
    main()
    logging.info("%s completes" % script_name)

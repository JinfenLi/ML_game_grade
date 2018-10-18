#!/usr/bin/env python
# -*- coding:utf8 -*-
# Created by jinfen @ Dec 27, 2017
# Modified by jinfen @ Dec 27, 2017

import sys
import os
import cPickle
import datetime
import warnings
import logging

import numpy as np
import pandas as pd
import sklearn.exceptions
from sklearn import metrics
from sklearn.metrics import classification_report

from model import lda
from model import jieba_bigram as jb
from model import senti
from model import dataprocessing as dp
from config import path_config

from utils import misc

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

TEST_DATA_DIR = os.path.join(path_config.TEST_DATA_DIR, "test_%s_%s.csv" % (sys.argv[1], sys.argv[2]))
FEEDBACK_DIR = os.path.join(path_config.FEEDBACK_DIR, '%s_%s_predict.npz' % (sys.argv[1], sys.argv[2]))
MODEL_DIR = os.path.join(path_config.MODEL_DIR, 'clf.pkl')
FAULT_DIR = os.path.join(path_config.FAULT_DIR, '%s_%s_fault.csv' % (sys.argv[1], sys.argv[2]))


def loadmodel():
    with open(MODEL_DIR, "rb") as f:
        model = cPickle.load(f)
    return model


# params:testdataset
# return: predict result, every classification's precision, accuracy, f1score
def predict(test_data, rawdata):
    rawdata = pd.DataFrame(rawdata)
    XX = test_data.drop(
            ["grade", "evaluat_desc", 'game_id', "Description", "shortDesc", "UpdateDescription", "subject",
             "game_tags", "Type", "game_feature", "game_key", "main_play", "play_key", "game_play_way",
             "gamefeature", "playway", "grade_", "gametype"], axis=1)
    feature = XX.columns
    clf = loadmodel()
    y_pred = clf.predict(test_data[feature])
    accuracy = metrics.accuracy_score(test_data["grade_"], y_pred)
    print("accuracy", accuracy)

    print('*' * 10 + 'confusion matrix' + '*' * 10)
    print(metrics.confusion_matrix(test_data["grade_"], y_pred))
    print('*' * 10 + 'performance measure' + '*' * 10)
    print(classification_report(test_data["grade_"], y_pred))
    gradeset = list(set(test_data["grade_"]))
    precision = metrics.precision_score(test_data["grade_"], y_pred, average=None)
    print("precision", precision)

    recallscore = metrics.recall_score(test_data["grade_"], y_pred, average=None)
    print("recall score", recallscore)

    f1score = metrics.f1_score(test_data["grade_"], y_pred, average=None)
    print("f1score", f1score)
    print(list(zip(precision, recallscore, f1score)))

    test_data["pred"] = y_pred
    feedback_dataset = pd.concat([rawdata, test_data["pred"]], axis=1)
    # print [column for column in feedback_dataset]
    fault = test_data[test_data.pred != test_data.grade_]
    fault.to_csv(FAULT_DIR, index=True, header=True)
    feedback_dataset = np.array(feedback_dataset)
    accuracy = np.array(accuracy)
    score = np.array(list(zip(gradeset, precision, recallscore, f1score)))
    print("score", score)
    np.savez(FEEDBACK_DIR, accuracy=accuracy, score=score, total_data=feedback_dataset)

    return test_data


def main():
    starttime = datetime.datetime.now()
    rawdata = dp.loadtestdata(TEST_DATA_DIR)
    predata = dp.gen_rawdata(rawdata)
    test = dp.datapreprocessing(predata)
    test = test.reset_index(drop=True)

    feature_dict = {"evaluat_desc": 500, "Description": 300}
    for key in feature_dict:
        X = jb.testdataset(key, feature_dict[key], test)
        test = pd.concat([test, X], axis=1)

    # lda feature training
    lda_feature_dict = {"playway": 3, "gamefeature": 3, "gametype": 3,
                        "subject": 3, "evaluat_desc": 2}
    test = test.reset_index(drop=True)

    for key in lda_feature_dict:
        X = lda.testdatalda(key, lda_feature_dict[key], test)
        column_name = key + "_topic"
        test = pd.concat([X[column_name], test], axis=1)

    test = senti.senti(test)
    predict(test, rawdata)

    endtime = datetime.datetime.now()
    logging.info("predict time: %.2f" % (endtime - starttime).seconds)


if __name__ == "__main__":
    misc.SetupLogger(level=logging.DEBUG)

    script_name = sys.argv[0]
    logging.info("%s starts" % script_name)
    main()
    logging.info("%s completes" % script_name)

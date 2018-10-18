# !/usr/bin/env python
# -*- coding:utf8 -*-
# Created by jinfen @ Dec 27, 2017
# Modified by jinfen @ Dec 27, 2017

import sys
import os
import cPickle
import logging

import pandas as pd
import numpy as np
from imblearn.datasets import make_imbalance
from sklearn.ensemble import RandomForestClassifier

from model import lda
from model import jieba_bigram as jb
from model import senti
from model import dataprocessing as dp
from config import path_config

from utils import misc


#MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath("")), 'data', 'modelfile')


def training(train_dataset):
    logging.debug("func called training")
    XX = train_dataset.drop(
            ["grade", "evaluat_desc", 'game_id', "Description", "shortDesc", "UpdateDescription", "subject",
             "game_tags", "Type", "game_feature", "game_key", "main_play", "play_key", "game_play_way",
             "playway", "gamefeature", "grade_", "gametype"], axis=1)

    feature = XX.columns

    X_train, y_train = train_dataset[feature], train_dataset["grade_"]

    logging.debug('balancing dataset')
    def ratio_data(grade, n):

        return int(round(len(y_train[y_train == grade]) * n))

    ratio = {}
    #ratio = {3: ratio_data(3, 1), 4: ratio_data(4, 1), 1: ratio_data(1, 1), 2: ratio_data(2, 0.9)}
    for i in range(1,5):
        if i in list(set(y_train)) and i!=2:
            ratio[i] = ratio_data(i, 1)
        if i in list(set(y_train)) and i == 2:
            ratio[i] = ratio_data(i, 0.9)
    X_train, y_train = make_imbalance(XX, train_dataset.grade_, ratio=ratio)

    logging.debug('feature importance & feature selection')
    clf = RandomForestClassifier(criterion='entropy', n_estimators=100, random_state=1, n_jobs=2)
    clf.fit(X_train, y_train)
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    for f in range(X_train.shape[1]):
        print(XX.columns[f], importances[indices[f]])

    logging.debug('save params to pkl file')
    with open(os.path.join(path_config.MODEL_DIR, 'clf.pkl'), "wb") as f:
        cPickle.dump(clf, f)

    return clf


def main():
    logging.info("data preprocessing")
    rawdata = dp.loadtraindata(train_dir)
    predata = dp.gen_rawdata(rawdata)
    train = dp.datapreprocessing(predata)
    train = train[train.grade != 0]
    train = train[(train.evaluat_desc != '0')]
    train = train.reset_index(drop=True)
    feature_dict = {"evaluat_desc": 500, "Description": 300}

    logging.info("extracting features with jieba")
    for key in feature_dict:
        X = jb.traindataset(key, feature_dict[key], train)
        train = pd.concat([train, X], axis=1)

    logging.info('lda feature training')
    lda_feature_dict = {"playway": 3, "gamefeature": 3, "gametype": 3,
                        "subject": 3,
                        "evaluat_desc": 2}
    train = train.reset_index(drop=True)

    for key in lda_feature_dict:
        X = lda.traindatalda(key, lda_feature_dict[key], train)

        column_name = key + "_topic"
        train = pd.concat([X[column_name], train], axis=1)

    logging.info("extracting sentimental feature")
    train = senti.senti(train)

    logging.info("model training starts...")
    training(train)


if __name__ == "__main__":
    misc.SetupLogger(level=logging.DEBUG)
    script_name = sys.argv[0]

    train_dir = os.path.join(path_config.TRAIN_DATA_DIR, "train_%s_%s.csv" % (sys.argv[1], sys.argv[2]))
    logging.info("%s starts..." % script_name)
    main()
    logging.info("%s completes..." % script_name)

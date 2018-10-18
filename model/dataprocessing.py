#!/usr/bin/env python
# -*- coding:utf8 -*-
# Created by jinfen @ Dec 27, 2017
# Modified by jinfen @ Dec 27, 2017
import os
import pandas as pd
import jieba.analyse
import logging

from config import path_config

GAMEWORD_DIR = os.path.join(path_config.DATA_DIR, 'wordbag', 'game_words.txt')


# load data
def loadtestdata(test_dir):
    logging.debug("func called loadtest data")
    jieba.load_userdict(GAMEWORD_DIR)
    testdataset = pd.read_csv(test_dir)
    #logging.debug("original size: %s" % str(testdataset.shape))
    return testdataset


def loadtraindata(train_dir):
    logging.debug("func called loadtraindata")
    jieba.load_userdict(GAMEWORD_DIR)
    traindata = pd.read_csv(train_dir)
    return traindata


def gen_rawdata(data):
    logging.debug("func called gen_rawdata")
    data = data.loc[:, (
        'Description', 'shortDesc', 'UpdateDescription', 'Downloads', 'pingce_score', 'NeedNet', 'sociaty_attribute',
        'cate_type', 'Languages',
        'game_feature', 'game_key', 'main_play', 'play_key', 'game_play_way', 'subject', 'game_tags', 'Type', 'grade',
        'evaluat_desc', 'game_id', 'mix_server', 'recommend', 'is_enable')]
    return data


# Data pre-processing
# Include text split, feature selection, reindex the grade,combine similar features
# params:data before processing
# return data after processing
def datapreprocessing(raw_data):
    logging.debug("func called datapreprocessing")
    for ind, row in raw_data.iterrows():

        if row['cate_type'] == '网游':
            raw_data.loc[ind, 'cate_type'] = 1
        elif row['cate_type'] == '单机':
            raw_data.loc[ind, 'cate_type'] = 2
        else:
            raw_data.loc[ind, 'cate_type'] = 0

    # fill the null column with mean value or 0
    raw_data.fillna(
            {'pingce_score': raw_data['pingce_score'].mean(), 'Downloads': raw_data['Downloads'].mean(), 'grade': 0,
             'mix_server': 0, 'recommend': 0},
            inplace=True)
    processeddata = raw_data[raw_data.grade != 0]
    processeddata = processeddata.fillna('0')
    # print('value   num')
    # print(processeddata.grade.value_counts())

    # grade reindex
    for ind, row in processeddata.iterrows():
        if row["grade"] == 4:
            processeddata.loc[ind, "grade_"] = 1
        elif row["grade"] == 5:
            processeddata.loc[ind, "grade_"] = 1
        elif row["grade"] == 2:
            processeddata.loc[ind, "grade_"] = 2
        elif row["grade"] == 3:
            processeddata.loc[ind, "grade_"] = 2
        elif row["grade"] == 1:
            processeddata.loc[ind, "grade_"] = 3
        elif row["grade"] == 6:
            processeddata.loc[ind, "grade_"] = 4

    combineplayway(processeddata)
    combinegamefeature(processeddata)
    comebinegametype(processeddata)
    return processeddata


# combine similar feature
# "main_play"+"play_key"+"game_play_way" /  "game_feature"+"game_key" / "game_tags"+"Type"
def combineplayway(data):
    logging.debug("func called combineplayway")
    for ind, row in data.iterrows():
        playway = []
        if row["main_play"] != '0':
            mainplaylist = row["main_play"].split(",")
            playway.extend(mainplaylist)
        if row["play_key"] != '0':
            playkeylist = row["play_key"].split(",")
            playway.extend(playkeylist)
        if row["game_play_way"] != '0':
            gameplaywaylist = row["game_play_way"].split(",")
            playway.extend(gameplaywaylist)
        playway = list(set(playway))
        if len(playway) > 0:
            data.loc[ind, 'playway'] = ",".join(playway)
        else:
            data.loc[ind, 'playway'] = '0'
    return data


def combinegamefeature(data):
    logging.debug("func called combinegamefeature")
    for ind, row in data.iterrows():
        gamefeature = []
        if row["game_feature"] != '0':
            gamefeaturelist = row["game_feature"].split(",")
            gamefeature.extend(gamefeaturelist)
        if row["game_key"] != '0':
            gamekeylist = row["game_key"].split(",")
            gamefeature.extend(gamekeylist)
        gamefeature = list(set(gamefeature))
        if len(gamefeature) > 0:
            data.loc[ind, 'gamefeature'] = ",".join(gamefeature)
        else:
            data.loc[ind, 'gamefeature'] = '0'

    return data


def comebinegametype(data):
    logging.debug("func called comebinegametype")
    for ind, row in data.iterrows():
        gametype = []
        if row["game_tags"] != '0':
            gametagslist = row["game_tags"].split(";")
            gametype.extend(gametagslist)
        if row["Type"] != '0':
            gametypelist = row["Type"].split(",")
            gametype.extend(gametypelist)
        gametype = list(set(gametype))
        if len(gametype) > 0:
            data.loc[ind, 'gametype'] = ",".join(gametype)
        else:
            data.loc[ind, 'gametype'] = '0'

    return data

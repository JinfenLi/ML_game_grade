#!/usr/bin/env python
# -*- coding:utf8 -*-
# Created by jinfen @ Dec 27, 2017
# Modified by jinfen @ Dec 27, 2017
import os
import re
import logging

import jieba
import pandas as pd
import numpy as np

from model import stopword as sw
from config import path_config

WORD_DIR = os.path.join(path_config.DATA_DIR, 'wordbag')


def senti(data):
    logging.debug("func called senti")
    split_result = []
    feature = 'evaluat_desc'
    stopwords = sw.stopword()

    # cut words
    for ind, row in data.iterrows():
        s = row.get(feature)
        train_feature = re.sub(
                "[0-9\[\`\~\!\@\#\$\^\&\*\(\)\=\|\{\}\'\:\;\'\,\[\]\.\<\>\/\?\~\！\@\#\\\&\*\%\【\】\，\。\；\&\#\@\\r\\n\》\《\（\）]",
                "", s)
        split_text = jieba.cut(train_feature, cut_all=False)  # 返回了所有分词结果
        split_result.append(split_text)

    split = []

    # filter words from stopwords
    for num_train in range(len(split_result)):
        word_bag = []
        for word in split_result[num_train]:
            if (word != '' and word != "\n" and word != "\r\n" and word != "\r" and word not in stopwords):
                word_bag.append(word)
        split.append(word_bag)
    split_result = split

    #print("split_result", split_result)

    # Generate positive word as a list
    with open(os.path.join(WORD_DIR, 'positive.txt'), 'rb') as positive_word:
        positive_list = []
        while True:
            line = positive_word.readline().decode('utf8')
            line = re.sub("\r\n", "", line)
            if not line:
                break
            positive_list.append(line)

    # Generate negative word as a list
    with open(os.path.join(WORD_DIR, 'negative.txt'), 'rb') as negative_word:
        negative_list = []
        while True:
            line = negative_word.readline().decode('utf8')
            line = re.sub("\r\n", "", line)
            if not line:
                break
            negative_list.append(line)

    # Generate power word as a Dict like Dict{word:power}
    with open(os.path.join(WORD_DIR, 'degree.txt'), 'rb') as power_word:
        degreeDict = {}
        value = 0
        while True:
            line = power_word.readline().decode('utf8')
            line = re.sub('\r\n', '', line)
            if line.startswith('degree:'):
                value = re.sub('degree:', '', line)
            else:
                degreeDict[line] = value
            if not line:
                break

    # Generate negation word  as a list
    with open(os.path.join(WORD_DIR, 'negation_words.txt'), 'rb') as negation_word:
        negationWord = []
        while True:
            line = negation_word.readline().decode('utf8')
            line = re.sub("\r\n", "", line)
            if not line:
                break
            negationWord.append(line)

    # deduplicate useless words
    posWord, negWord, degreeWord = {}, {}, {}
    for num_train in range(len(split_result)):
        for word in split_result[num_train]:
            if word in positive_list and word not in negative_list and word not in degreeDict.keys():
                posWord[word] = 1
            elif word in negative_list and word not in degreeDict.keys():
                negWord[word] = -1
            elif word in degreeDict.keys():
                degreeWord[word] = degreeDict[word]

    # 根据各种词法组合取词
    sentiment_count_all = np.zeros((1, len(split_result)))
    word_select_list = []
    for num_train in range(len(split_result)):

        word_select = []
        count = 0
        word_bag = split_result[num_train]

        for abs_position in range(len(word_bag)):

            if word_bag[abs_position] in posWord.keys():  # found a positive sentiment word
                if abs_position != 0 and abs_position != 1 and word_bag[abs_position - 2] in degreeDict.keys() and \
                                word_bag[
                                            abs_position - 1] in negationWord:
                    # tensity + negation + sentiment(非常不友好)
                    score = float(degreeDict[word_bag[abs_position - 2]]) / 29 * -float(posWord[word_bag[abs_position]])
                    sentiment_count_all[0][num_train] += score
                    count += 1
                    word_select.append(
                            word_bag[abs_position - 2] + word_bag[abs_position - 1] + word_bag[abs_position] + str(
                                score))

                elif abs_position != 0 and abs_position != 1 and word_bag[abs_position - 2] in negationWord and \
                                word_bag[
                                            abs_position - 1] in degreeDict.keys():
                    # negation + tensity + sentiment(不非常友好)
                    score = float(degreeDict[word_bag[abs_position - 1]]) / 29 * -float(posWord[word_bag[abs_position]])
                    sentiment_count_all[0][num_train] += score
                    count += 1
                    word_select.append(
                            word_bag[abs_position - 2] + word_bag[abs_position - 1] + word_bag[abs_position] + str(
                                score))

                elif abs_position != 0 and abs_position != 1 and abs_position != len(word_bag) - 1 and word_bag[
                            abs_position - 1] in negationWord and word_bag[abs_position + 1] in degreeDict.keys():
                    # negation + sentiment + tensity (不能高兴过头)
                    score = float(degreeDict[word_bag[abs_position + 1]]) / 29 * -float(posWord[word_bag[abs_position]])
                    sentiment_count_all[0][num_train] += score
                    count += 1
                    word_select.append(
                            word_bag[abs_position - 1] + word_bag[abs_position] + word_bag[abs_position + 1] + str(
                                score))

                elif abs_position != 0 and abs_position != len(word_bag) - 1 and abs_position != len(word_bag) - 2 and \
                                word_bag[abs_position + 1] in negationWord and word_bag[
                            abs_position + 2] in degreeDict.keys():
                    # sentiment + negation + tensity (高兴不能过头)
                    score = float(degreeDict[word_bag[abs_position + 2]]) / 29 * -float(posWord[word_bag[abs_position]])
                    sentiment_count_all[0][num_train] += score
                    count += 1
                    word_select.append(
                            word_bag[abs_position] + word_bag[abs_position + 1] + word_bag[abs_position + 2] + str(
                                score))

                elif abs_position != 0 and word_bag[abs_position - 1] in degreeDict.keys():
                    # tensity + sentiment(非常友好)
                    score = float(degreeDict[word_bag[abs_position - 1]]) / 29 * float(posWord[word_bag[abs_position]])
                    sentiment_count_all[0][num_train] += score
                    count += 1
                    word_select.append(word_bag[abs_position - 1] + word_bag[abs_position] + str(score))
                elif abs_position != 0 and word_bag[abs_position - 1] in negationWord:
                    # negation + sentiment(不友好)
                    score = -float(posWord[word_bag[abs_position]])
                    sentiment_count_all[0][num_train] += score
                    count += 1
                    word_select.append(word_bag[abs_position - 1] + word_bag[abs_position] + str(score))
                elif abs_position != 0 and abs_position != len(word_bag) - 1 and word_bag[
                            abs_position + 1] in degreeDict.keys():
                    # sentiment + tensity ?（高兴过头）
                    count += 1
                    score = float(degreeDict[word_bag[abs_position + 1]]) / 29 * float(posWord[word_bag[abs_position]])
                    sentiment_count_all[0][num_train] += score
                    word_select.append(word_bag[abs_position] + word_bag[abs_position + 1] + str(score))
                elif abs_position != 0 and abs_position != len(word_bag) - 1 and word_bag[
                            abs_position + 1] in negationWord:
                    # sentiment+ negation  (价值不大)
                    score = -float(posWord[word_bag[abs_position]])
                    sentiment_count_all[0][num_train] += score
                    count += 1
                    word_select.append(word_bag[abs_position] + word_bag[abs_position + 1] + str(score))
                else:
                    score = float(posWord[word_bag[abs_position]])
                    sentiment_count_all[0][num_train] += score
                    count += 1
                    word_select.append(word_bag[abs_position] + str(score))

            elif word_bag[abs_position] in negWord.keys():  # found a negative sentiment word
                if abs_position != 0 and abs_position != 1 and word_bag[abs_position - 2] in degreeDict.keys() and \
                                word_bag[
                                            abs_position - 1] in negationWord:
                    # tensity + negation + sentiment
                    score = float(degreeDict[word_bag[abs_position - 2]]) / 29 * -float(negWord[word_bag[abs_position]])
                    sentiment_count_all[0][num_train] += score
                    count += 1
                    word_select.append(
                            word_bag[abs_position - 2] + word_bag[abs_position - 1] + word_bag[abs_position] + str(
                                score))

                elif abs_position != 0 and abs_position != 1 and word_bag[abs_position - 2] in negationWord and \
                                word_bag[
                                            abs_position - 1] in degreeDict.keys():
                    # negation + tensity + sentiment
                    score = float(degreeDict[word_bag[abs_position - 1]]) / 29 * -float(negWord[word_bag[abs_position]])
                    sentiment_count_all[0][num_train] += score
                    count += 1
                    word_select.append(
                            word_bag[abs_position - 2] + word_bag[abs_position - 1] + word_bag[abs_position] + str(
                                score))

                elif abs_position != 0 and abs_position != 1 and abs_position != len(word_bag) - 1 and word_bag[
                            abs_position - 1] in negationWord and word_bag[abs_position + 1] in degreeDict.keys():
                    # negation + sentiment + tensity
                    score = float(degreeDict[word_bag[abs_position + 1]]) / 29 * -float(negWord[word_bag[abs_position]])
                    sentiment_count_all[0][num_train] += score
                    count += 1
                    word_select.append(
                            word_bag[abs_position - 1] + word_bag[abs_position] + word_bag[abs_position + 1] + str(
                                score))

                elif abs_position != 0 and abs_position != len(word_bag) - 1 and abs_position != len(word_bag) - 2 and \
                                word_bag[abs_position + 1] in negationWord and word_bag[
                            abs_position + 2] in degreeDict.keys():
                    # sentiment + negation + tensity
                    score = float(degreeDict[word_bag[abs_position + 2]]) / 29 * -float(negWord[word_bag[abs_position]])
                    sentiment_count_all[0][num_train] += score
                    count += 1
                    word_select.append(
                            word_bag[abs_position] + word_bag[abs_position + 1] + word_bag[abs_position + 2] + str(
                                score))

                elif abs_position != 0 and word_bag[abs_position - 1] in degreeDict.keys():
                    # tensity + sentiment
                    score = float(degreeDict[word_bag[abs_position - 1]]) / 29 * float(negWord[word_bag[abs_position]])
                    sentiment_count_all[0][num_train] += score
                    count += 1
                    word_select.append(word_bag[abs_position - 1] + word_bag[abs_position] + str(score))

                elif abs_position != 0 and word_bag[abs_position - 1] in negationWord:
                    # negation + sentiment
                    score = -float(negWord[word_bag[abs_position]])
                    sentiment_count_all[0][num_train] += score
                    count += 1
                    word_select.append(word_bag[abs_position - 1] + word_bag[abs_position] + str(score))

                elif abs_position != 0 and abs_position != len(word_bag) - 1 and word_bag[
                            abs_position + 1] in degreeDict.keys():
                    # sentiment + tensity ?（高兴过头）
                    count += 1
                    score = float(degreeDict[word_bag[abs_position + 1]]) / 29 * float(negWord[word_bag[abs_position]])
                    sentiment_count_all[0][num_train] += score
                    word_select.append(word_bag[abs_position] + word_bag[abs_position + 1] + str(score))

                elif abs_position != 0 and abs_position != len(word_bag) - 1 and word_bag[
                            abs_position + 1] in negationWord:
                    # sentiment+ negation  (价值不大)
                    score = -float(negWord[word_bag[abs_position]])
                    sentiment_count_all[0][num_train] += score
                    count += 1
                    word_select.append(word_bag[abs_position] + word_bag[abs_position + 1] + str(score))

                else:
                    score = float(negWord[word_bag[abs_position]])
                    sentiment_count_all[0][num_train] += score
                    count += 1
                    word_select.append(word_bag[abs_position] + str(score))

        word_select_list.append(word_select)
        if count != 0:
            sentiment_count_all[0][num_train] = sentiment_count_all[0][num_train] / count

    # reset column index of traindata
    data = data.reset_index(drop=True)
    data = pd.concat([data, pd.DataFrame(sentiment_count_all.T, columns=["senti"])], axis=1)
    return data

#!/usr/bin/env python
# -*- coding:utf8 -*-
# Created by jinfen @ Dec 27, 2017
# Modified by jinfen @ Dec 27, 2017
import sys
import os
import re
import codecs
from collections import defaultdict

import pandas as pd
from snownlp import SnowNLP
import jieba

import stopword as sw
from config import path_config

WORD_DIR = os.path.join(path_config.DATA_DIR, 'wordbag')


def biCorpus(data, feature):
    corpus = []
    stopwords = sw.stopword()
    zhPattern = re.compile(u'[\u4e00-\u9fa5]+')
    for ind, row in data.iterrows():
        text = row[feature]
        seg_list = jieba.cut(text, cut_all=False)
        result = []

        for seg in seg_list:
            seg = ''.join(seg.split())
            seg = re.sub(
                "[0-9\[\`\~\!\@\#\$\^\&\*\(\)\=\|\{\}\'\:\;\'\,\[\]\.\<\>\/\?\~\！\@\#\\\&\*\%\【\】\《\》\（\）\：\、\“\”\，\。\；]",
                "", seg)
            match = zhPattern.search(seg)
            if match is not None and seg not in stopwords:
                result.append(seg)
        corpus.append(result)
    # bi_Corpus指每句话形成的bigram,[[],[],[]]
    bi_Corpus = [[] for i in range(len(corpus))]
    for i in range(len(corpus)):
        for j in range(len(corpus[i]) - 1):
            word = "".join([corpus[i][j], corpus[i][j + 1]])
            bi_Corpus[i].append(word)  # 每条的bigram
    return bi_Corpus


def saveidf(bi_Corpus,feature):
    total_bi_Corpus = []
    for i in range(len(bi_Corpus)):
        total_bi_Corpus.extend(bi_Corpus[i])
    s = SnowNLP(bi_Corpus)
    # idf={bigram:idf,bigram:idf}
    idf = s.idf
    # tf=[{bigram:tf},{bigram:tf},{bigram:tf}]
    tf = s.tf
    idffile = open(os.path.join(WORD_DIR, feature+"_bigramidf.txt"), 'w')
    for key in idf:
        idffile.write(key.encode('utf-8') + "," + str(idf[key]).encode('utf-8') + "," + "\n")
    idffile.close()
    return total_bi_Corpus, idf, tf


def loadidf(feature):
    idf = {}
    idffile = codecs.open(os.path.join(WORD_DIR, feature+"_bigramidf.txt"), 'r',encoding='utf-8')
    for line in idffile:
        l = line.split(",")
        idf[l[0]] = float(l[1])
    idffile.close()
    return idf


def savekeyword(total_bi_Corpus, idf, n,feature):
    s_total = SnowNLP([total_bi_Corpus])
    # total_tf指词频字典，因为total_tf=[{bigram:tf，bigram:tf，bigram:tf}]
    total_tf = s_total.tf[0]
    # textotal所有关键词的tfidf值，{bigram:tfidf,bigram:tfidf}
    texttotal = defaultdict(dict)
    for key, value in total_tf.items():
        if key in idf.keys():
            texttotal[key] = value * idf[key]
    # sorted_dict 指 将tfidf降序排序，[(bigram,tfidf),(bigram,tfidf)]
    sorted_dict = sorted(sort_dict(texttotal), key=lambda x: x[1], reverse=True)
    keyword = []
    for i in range(min(n, len(sorted_dict))):
        keyword.append(sorted_dict[i][0])  # 整体前n个关键词

    keywordfile = open(os.path.join(WORD_DIR, feature+"_keyword.txt"), 'w')
    for k in keyword:
        keywordfile.write(k.encode('utf-8') + "\n")
    keywordfile.close()
    return keyword


def loadkeyword(feature):
    keywordfiile = codecs.open(os.path.join(WORD_DIR, feature+"_keyword.txt"), 'r',encoding='utf-8')
    keyword = []
    for line in keywordfiile:
        keyword.append(line.rstrip("\n"))
    keywordfiile.close()
    return keyword


def tfidf(tf, idf, n):
    texttotal = defaultdict(dict)
    # texttotal_指每句话的tfidf{}
    for i in range(len(tf)):
        texttotal_ = defaultdict(dict)
        for key, value in tf[i].items():
            if key in idf.keys():
                texttotal_[key] = value * idf[key]
        sorted_dict_ = sorted(sort_dict(texttotal_), key=lambda x: x[1], reverse=True)
        texttotal[i].update(sorted_dict_[0:n])  # 每条语句的前n个关键词
    return texttotal


def totaldataframe(texttotal, keyword, n, feature):
    text = defaultdict(dict)
    key_importance = []
    for key, value in texttotal.items():
        text_feature_dict = defaultdict(dict)
        count = 0
        for Feature, weight in value.items():
            if Feature in keyword:
                count += 1
                text_feature_dict[Feature] = weight
        key_importance.append(count * count / n)
        # 每句话能匹配上keyword的tfidf值
        text[key].update(text_feature_dict)
    key_importance = pd.DataFrame(key_importance, columns=[feature + '_key_importance'])
    text = pd.DataFrame(text).T
    text_sum = text.apply(lambda x: x.sum(), axis=1)
    text_sum = pd.DataFrame(text_sum, columns=[feature + "_sum"])
    text_sum.fillna(0, inplace=True)
    text_sum = text_sum.reset_index(drop=True)
    text_sum = pd.concat([text_sum, key_importance], axis=1)
    return text_sum


def traindataset(feature, n, data):
    bi_Corpus = biCorpus(data, feature)
    total_bi_Corpus, idf, tf = saveidf(bi_Corpus,feature)
    keyword = savekeyword(total_bi_Corpus, idf, n,feature)
    texttotal = tfidf(tf, idf, n)
    df = totaldataframe(texttotal, keyword, n, feature)
    return df


def testdataset(feature, n, data):
    bi_Corpus = biCorpus(data, feature)
    s = SnowNLP(bi_Corpus)
    tf = s.tf
    idf = loadidf(feature)
    keyword = loadkeyword(feature)
    texttotal = tfidf(tf, idf, n)
    df = totaldataframe(texttotal, keyword, n, feature)
    return df


# key word feature
def sort_dict(dic):
    keys = dic.keys()
    vals = dic.values()
    lst = [(key, val) for key, val in zip(keys, vals)]
    return lst

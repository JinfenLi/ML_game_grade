#!/usr/bin/env python
# -*- coding:utf8 -*-
# Created by jinfen @ Dec 27, 2017
# Modified by jinfen @ Dec 27, 2017

import os
import codecs
import logging

import pandas as pd
import numpy as np
import jieba.analyse

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from model import stopword
from config import path_config

WORD_DIR = os.path.join(path_config.DATA_DIR, 'wordbag')

# input:text
# return: word list
def word_cut(text):
    seg_list = " ".join(jieba.cut(text, cut_all=False))
    return seg_list

# params: feature:simple text, n_topic:topics ,train:data
# return: traindata dataframe with lda score


def savedldaidf(feature,data):
    logging.debug("func called savedldaidf")
    stopwords = stopword.stopword()
    XX = data[feature].apply(word_cut)
    tf_vectorizer = CountVectorizer(strip_accents='unicode',
                                    max_features=1000,
                                    stop_words=stopwords,
                                    max_df=1.0,
                                    min_df=1)
    tf = tf_vectorizer.fit_transform(XX)
    transformer = TfidfTransformer(norm = None)
    train_feature_dtm = transformer.fit_transform(tf)
    lda_dictionary = dict(zip(tf_vectorizer.get_feature_names(), transformer.idf_))
    lda_idffile = open(os.path.join(WORD_DIR,feature + '_ldaidf' + '.txt'),'w')
    for key in lda_dictionary:
        lda_idffile.write(key.encode('utf-8') + "," + str(lda_dictionary[key]).encode('utf-8') + "," + "\n")
    lda_idffile.close()
    return train_feature_dtm.toarray()


def loadldaidf(feature):
    logging.debug("func called loadldaidf")
    ldaidf = {}
    lda_idffile = codecs.open(os.path.join(WORD_DIR, feature+'_ldaidf' + '.txt'), 'r',encoding='utf-8')
    for line in lda_idffile:
        l = line.split(",")
        ldaidf[l[0]] = float(l[1])
    lda_idffile.close()
    return ldaidf


def lda_similarity(feature, n_topic, feature_idf_array, data):
    logging.debug("func called lda_similarity")
    column_name = [feature + "_topic"]
    lda = LatentDirichletAllocation(n_components=n_topic, max_iter=50,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0)

    feature_dtm = lda.fit_transform(feature_idf_array)

    new_column_name = [str(i) + feature for i in range(n_topic)]
    ldadata = pd.DataFrame(np.matrix(pd.DataFrame(feature_dtm, columns=new_column_name)).argmax(axis=1) + 1,
                           columns=column_name)

    ldadata.loc[data[feature] == "0", column_name] = 0
    return ldadata


def traindatalda(feature,n_topic,data):
    logging.debug("func called traindatalda")
    feature_idf_array = savedldaidf(feature, data)
    ldatraindata = lda_similarity(feature,n_topic,feature_idf_array,data)
    return ldatraindata

def testdatalda(feature,n_topic,data):
    logging.debug("func called testdatalda")
    stopwords = stopword.stopword()
    column_name = [feature + "_topic"]
    XX = data[feature].apply(word_cut)

    # prevent void vocabulary or stopwords only
    if sum(XX == "0") == len(XX):
        ldatestdata = pd.DataFrame(np.zeros((len(XX), 1)), columns=column_name)
    else:
        ldaidf = loadldaidf(feature)
        tf_vectorizer = CountVectorizer(strip_accents='unicode',
                                        max_features=1000,
                                        stop_words=stopwords,
                                        max_df=1.0,
                                        min_df=1)
        tf = tf_vectorizer.fit_transform(XX)
        name = tf_vectorizer.get_feature_names()

        lda_feature = pd.DataFrame(tf.toarray(),columns = name)
        for ind,row in lda_feature.iterrows():
            for names in name:
                if names in ldaidf.keys():
                    lda_feature.loc[ind,names] = lda_feature.loc[ind,names] * ldaidf[names]

        #lda_feature.to_csv("test_array.csv", index=True, header=True)
        ldatestdata = lda_similarity(feature, n_topic, lda_feature, data)
    return ldatestdata




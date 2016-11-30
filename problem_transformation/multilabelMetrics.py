# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 21:39:50 2016

@author: Ouadie GHARROUDI g.ouadie@gmail.com
"""

import numpy as np
from sklearn.metrics import zero_one_loss as sal
from sklearn.metrics import hamming_loss as hl
from sklearn.metrics import f1_score as f1
from sklearn.metrics import jaccard_similarity_score as ji
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import coverage_error
from sklearn.metrics import label_ranking_loss as rl

def instanceF1(y_true, y_pred):

    """
    y_true : 2d array-like, of size n x q
    y_pred : 2d array-like, of size n x q
    """
    n, q = y_true.shape
    if1 = 0
    for i in np.arange(n):
        if1 = if1 + f1(y_true[i, :], y_pred[i, :])
    return if1/n

def evaluatesMLPredictions(y_true, y_pred):

    esal = sal(y_true, y_pred)
    ehl = hl(y_true, y_pred)
    ma = 1 - f1(y_true, y_pred, average='macro')
    mi = 1 - f1(y_true, y_pred, average='micro')
    if1 = 1 - instanceF1(y_true, y_pred)
    eji = 1 - ji(y_true, y_pred)
    mapre = 1 - precision_score(y_true, y_pred, average='macro')
    marec = 1 - recall_score(y_true, y_pred, average='macro')
    mipre = 1 - precision_score(y_true, y_pred, average='micro')
    mirec = 1 - recall_score(y_true, y_pred, average='micro')

    # probability metrics
    cov = coverage_error(y_true, y_pred)
    erl = rl(y_true, y_pred)

    return esal, ehl, ma, mi, if1, eji, mapre, marec, mipre, mirec, cov, erl


def computesMetric(y_true, y_pred, metric='HL'):

    if metric == 'HL':
        r = hl(y_true, y_pred)
    elif metric == 'SA':
        r = sal(y_true, y_pred)
    elif metric == 'Ma':
        r = 1 - f1(y_true, y_pred, average='macro')
    elif metric == 'Mi':
        r = 1 - f1(y_true, y_pred, average='micro')
    elif metric == 'IF1':
        r = instanceF1(y_true, y_pred)
    elif metric == 'IJ':
        r = 1 - ji(y_true, y_pred)
    elif metric == 'MaP':
        r = 1 - precision_score(y_true, y_pred, average='macro')
    elif metric == 'MiP':
        r = 1 - precision_score(y_true, y_pred, average='micro')
    elif metric == 'MaR':
        r = 1 - recall_score(y_true, y_pred, average='macro')
    elif metric == 'MiR':
        r = 1 - recall_score(y_true, y_pred, average='micro')
    return r

def printMLPerformances(y_true, y_pred):

    listename = ('zero_one_loss', 'hamming_loss',
                 'Macro-f1_score', 'Micro-f1_score',
                 'jaccard_similarity_score',
                 'Macro-precision_score', 'Micro-precision_score',
                 'Macro-recall_score', 'Micro-recall_score',
                 'coverage_error', 'label_ranking_loss')
    perf = evaluatesMLPredictions(y_true, y_pred)
    for metric, value in zip(listename, perf):
        print (metric, ': ', value)


if __name__ == "__main__":



    y_true = np.array([[0, 1, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0]])
    y_pred = np.array([[0, 1, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0]])

    eval0 = evaluatesMLPredictions(y_true, y_pred)
    #print(eval0)
    printMLPerformances(y_true, y_pred)


    y_true = np.array([[1], [1], [1], [0]])
    y_pred = np.array([[0], [1], [1], [0]])

    ee = computesMetric(y_true, y_pred, metric='IJ')
    #print(ee)
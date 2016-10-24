# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 04:35:58 2016

@author: Ouadie GHARROUDI g.ouadie@gmail.com
"""

import numpy as np
from multilabelMetrics import computesMetric


def forwardThresholdCalibration(y_true, y_scores, metric='HL'):
    _, q = y_true.shape
    sq = list([])
    st = list([])
    while len(sq) < q:
        tempt_vect = list([])
        tmp_perf_vect = list([])
        for qi in list(set(np.arange(q)) - set(sq)):
            if len(sq) == 0:
                tempt, tmp_perf = select_best_Threshold([], [],
                                                        y_true[:, qi],
                                                        y_scores[:, qi],
                                                        metric=metric)
            elif len(sq) == 1:
                tmp_oldy_true = y_true[:, sq]
                st0 = np.array(st)
                tmp_oldy_pred = threshold(y_scores[:, sq], T=st0[sq])
                tempt, tmp_perf = select_best_Threshold(tmp_oldy_true,
                                                        tmp_oldy_pred,
                                                        y_true[:, qi],
                                                        y_scores[:, qi],
                                                        metric=metric)
            else:
                tmp_oldy_true = y_true[:, sq]
                st0 = np.array(st)
                tmp_oldy_pred = threshold(y_scores[:, sq], T=st0[sq])
                tempt, tmp_perf = select_best_Threshold(tmp_oldy_true,
                                                        tmp_oldy_pred,
                                                        y_true[:, qi],
                                                        y_scores[:, qi],
                                                        metric=metric)
            tempt_vect.append(tempt)
            tmp_perf_vect.append(tmp_perf)

        temp_q = np.argmin(tmp_perf_vect)
        sq.append(list(set(np.arange(q)) - set(sq))[temp_q])
        st.append(tempt_vect[temp_q])
    T = np.zeros(q)
    for i, q in enumerate(sq):
        T[q] = st[i]
    return T

def select_best_Threshold(oldy_true, oldy_pred, y_t, y_s, metric='HL'):
    """ Select the best threshold for the y_s given the old predi ctions """
    y_t = np.expand_dims(y_t, axis=1)
    y_s = np.expand_dims(y_s, axis=1)
    t_range = np.arange(0, 1, 0.01)
    t_perf = np.zeros(t_range.shape)
    if oldy_true == [] and oldy_pred == []:
        for i, ti in enumerate(t_range):
            # transforms the P_scores
            tmp_y = applayonethreshold(y_s, t=ti)
            # computes the metric results
            t_perf[i] = computesMetric(y_t, tmp_y, metric=metric)
            # select the position fo the best performing threshold
    else:
       for i, ti in enumerate(t_range):
            # transforms the P_scores
            tmp_y = applayonethreshold(y_s, t=ti)
            # computes the metric results
            t_perf[i] = computesMetric(np.concatenate((oldy_true, y_t), axis=1),
                                       np.concatenate((oldy_pred, tmp_y), axis=1),
                                       metric=metric)
            # select the position fo the best performing threshold
    return t_range[np.argmin(t_perf)], np.min(t_perf)

def threshold(y_s, T):
    if y_s.shape[1] == len(T):
        y_pred = np.zeros(y_s.shape)
        for q in np.arange(y_s.shape[1]):
            y_pred[:, q] = applayonethreshold(y_s[:, q], t=T[q])
    return y_pred

def applayonethreshold(y_s0, t=0.5):
    y_pred = np.zeros(y_s0.shape)
    y_pred[y_s0>t] = 1
    return y_pred

if __name__ == "__main__":

    y_s = np.random.rand(3,1)
    y_t = np.array([[0],
                    [0],
                    [1]])

    oldy_true = np.array([[0, 1],
                          [0, 1],
                          [1, 1]])

    oldy_pred = np.array([[0, 1],
                          [0, 1],
                          [1, 1]])

    y_scores=np.array([[0.9649, 0.9595, 0.1712, 0.0344],
                       [0.1576, 0.6557, 0.7060, 0.4387],
                       [0.9706, 0.0357, 0.0318, 0.3816],
                       [0.9572, 0.8491, 0.2769, 0.7655],
                       [0.4854, 0.9340, 0.0462, 0.7952],
                       [0.8003, 0.6787, 0.0971, 0.1869],
                       [0.1419, 0.7577, 0.8235, 0.4898],
                       [0.4218, 0.7431, 0.6948, 0.4456],
                       [0.9157, 0.3922, 0.3171, 0.6463],
                       [0.7922, 0.6555, 0.9502, 0.7094]])

    y_true =np.array([[1, 1, 0, 1],
                      [1, 1, 1, 1],
                      [1, 0, 0, 1],
                      [1, 1, 1, 1],
                      [1, 1, 0, 1],
                      [1, 1, 0, 0],
                      [0, 1, 1, 0],
                      [1, 1, 1, 0],
                      [1, 0, 1, 0],
                      [1, 1, 1, 1]])

    filter_indices = [1,3,5]
    AA = np.array([11,13,155,22,0xff,32,56,88])
    #print( AA[filter_indices] )

    T = forwardThresholdCalibration(y_true, y_scores, metric='IF1')
    print ('The Threshold: ', T)

    A = [1, 2, 5, 4, 19]

    y_true = np.expand_dims(y_true[:, 0], axis=1)
    y_scores = np.expand_dims(y_scores[:, 0], axis=1)

    T = forwardThresholdCalibration(y_true, y_scores, metric='IF1')
    print ('The Threshold: ', T)
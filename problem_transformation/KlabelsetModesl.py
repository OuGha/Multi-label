# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 16:43:53 2016

@author: Ouadie GHARROUDI : ouadie.gharroudi@liris.cnrs.fr
"""
import copy
import numpy as np

from scipy.misc import comb
#from ..multi_target_models.exceptions import TargetShapeError
from  label_powerset import LabelPowersetClassifier
from sklearn import tree
from  ForwardThresholdClibration import applayonethreshold




class Rakel(object):
    """
    The multi-label Rakel model.

    Attributes
    ----------
    n_targets_ : Number of feature labels.

    models_ : List of models; its size is equal to n_targets_.
    """

    def __init__(self, model=None, m=100, k=3):
        self.model = model
        self.k = k
        self.n_estimators = m
        self.n_targets_ = 0
        self.regressors_ = []
        self.klps_ = None


    def klabesetGeneration__(m, Q, K=3):
        if m> comb(Q, K, exact=False):
            print(comb(Q, K, exact=False))
        klabelsets = []

        while(len(klabelsets) < m):
            x = np.random.choice(range(Q), size=K, replace=False, p=None).astype(int)
            x = np.sort(x)
            xs = ",".join([str(i) for i in x])
            if not any(xs in s for s in klabelsets):
                klabelsets.append(xs)

        return np.array([string.split(",") for string in klabelsets]).astype(int)

    def fit(self, X, Y):
        #print("local0")
        if len(Y.shape)==2:
            #print("local1")
            self.n_targets_ = int(Y.shape[1])
            self.estimators_ = []
            nX = X.shape[0]
            self.klps_ = Rakel.klabesetGeneration__(self.n_estimators, self.n_targets_, self.k)
            #print("local3")
            for i in range(0, self.n_estimators):
                #print("local4")
                lp = LabelPowersetClassifier(self.model)
                #print("arrrr")
                #print(Y[:, self.klps_[i]])
                lp.fit(X, Y[:, self.klps_[i]])

                self.estimators_.append(lp)
            return self
        else:
            raise Exception("The target musut be a two dimentions matrix you can use MultiTaret_Regressors for a vector target.")

    def predict_proba(self, Xtest):
        """
        Predict the regression value for Xtest.

        Parameters
        ----------
        Xtest : array-like of shape = [n_samples, n_features].

        Returns
        -------
        Predictions : an array whith the predict values of
            shape = [n_samplesn_targets] (n_targets = self.Q_).
        """
        nt = Xtest.shape[0]
        V = np.zeros((self.n_targets_))
        Predictions = np.zeros((nt, self.n_targets_))
        for i in range(0, self.n_estimators):
            P = self.estimators_[i].predict(Xtest)
            Predictions[:, self.klps_[i]] = Predictions[:, self.klps_[i]] + P
            V[self.klps_[i]]  = V[self.klps_[i]]  + np.ones((1, self.k))

        for i in range(0, self.n_estimators):
            Predictions[:, i] = Predictions[:, i]/V[i]
        return Predictions

    def predict(self, Xtest):
        """
        Predict Xtest the probabilities

        Parameters
        ----------
        Xtest : array-like with shape (n_samples, n_features).

        Returns
        -------
        predictions : an array with shape (n_samples, n_targets)
                      the predict values.
        """
        Predictions = self.predict_proba( Xtest)
        return applayonethreshold(Predictions, t=0.5)


if __name__ == '__main__':
#    unittest.main()
    """
    #import scipy.misc.comb
    a = comb(100, 3, exact=False)
    print(a)
    #range(self.n_targets_)
    a = range(10)
    x = np.random.choice(a, size=3, replace=False, p=None)

    X=np.zeros((3,3))
    print(x)
    print(type(x))
    for i in range(X.shape[0]):
        x = np.random.choice(a, size=3, replace=False, p=None).astype(int)
        X[i,:] = x
    X = X.astype(int)
    aa = list(map(lambda array: ",".join([str(i) for i in array]), X))
    print(aa)

    #any("abc" in s for s in some_list)
    any('3,9,8' in s for s in aa)
    #np.random.choice(a, size=None, replace=True, p=None)

    """

    r = Rakel.klabesetGeneration__(20, 6, 3)
#    print(r)
#    print(r[10])





    X = np.array([[411, 500, 426],
                  [100, -11, -96],
                  [125, 900, .00],
                  [.11, 60., 126],
                  [211, 100, 16],
                  [300, .60, 926],
                  [11., .00, 26],
                  [341, 700, 126]])
    Y = np.array([[0, 0, 0],
                  [0, 0, 1],
                  [0, 1, 0],
                  [0, 1, 1],
                  [1, 0, 0],
                  [1, 1, 0],
                  [1, 1, 1],
                  [1, 1, 1]])
    bmodel = tree.DecisionTreeClassifier(min_samples_leaf=1)
    sml = Rakel(bmodel, 3, 2)
    sml.fit(X, Y)
    sml.predict(X)
    print(sml.predict(X))
    print(sml.predict_proba(X))


# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 18:11:39 2016

@author: Ouadie GHARROUDI g.ouadie@gmail.com
"""
import copy
import numpy as np
from  ForwardThresholdClibration import applayonethreshold
#from MultiTarget_Regressors import SimpleMultiRegressor as SMR


class ClassifierChain:
    """
    The multi-regressor chain learn for each continous target i a regressor model
    using all j targets in the feature space with 0<=j<i.
    The base learner in this implemetation is the classical Regression tree.

    Attributes
    ----------
    `order` : The order in wich the target are learned.

    `Q_` : Number of feature targets.

    `Regressors_` : List of regressor the size of Regressors is equal to Q_.
    """

    def __init__(self, model=None):
        self.n_targets_ = 0
        self.regressors_ = []
        self.model = model
        self._flagmonolabel = None

        pass

    def fit(self, X, Y, randorder=True):
        """
        Build a multi-trget regression chain from the training set (X, Y)
        using the decision tree regressor.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Y : array-like, shape = [n_samples, n_targets]; (n_targets = self.Q_)
            The target values are real numbers in as in classical regression).

        randorder=bool (default=True)
        The order in which the target are learned.
        If True, then the order is random; otherwise uses the order givien in Y.

        Returns
        -------
        self : object
            Returns self.
        """
        if len(Y.shape)==2:
            self.Q_ = int(Y.shape[1])
            self.Regressors_ = []
            self._flagmonolabel = np.ones((self.Q_,))
            if randorder:
                self.order = np.random.permutation(self.Q_)
            else:
                self.order = range(0, self.Q_)
            for i in range(0, self.Q_):
                j = self.order[i]
                reg = copy.deepcopy(self.model)
                reg = reg.fit(X, Y[:,j])
                y = reg.predict(X)
                y = np.expand_dims(y, axis=0).T
                self.Regressors_.append(reg)
                X = np.concatenate((X, y), axis=1)
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
        Predictions = -1 * np.ones((nt, self.Q_))
        for i in range(0, self.Q_):
            j = self.order[i]
            y = self.Regressors_[i].predict(Xtest)
            if len(self.Regressors_[i].classes_) == 1:
                indx = 0
                if self.Regressors_[i].classes_[0] == 0:
                    Predictions[:, j] = 1-self.Regressors_[i].predict_proba(Xtest)[:, indx]
                elif self.Regressors_[i].classes_[0] == 1:
                    Predictions[:, j] = self.Regressors_[i].predict_proba(Xtest)[:, indx]
            else:
                indx = self.Regressors_[i].classes_[self.Regressors_[i].classes_==1][0]
                Predictions[:,j] = self.Regressors_[i].predict_proba(Xtest)[:, indx]
            y = np.expand_dims(y, axis=0).T
            Xtest = np.concatenate((Xtest, y), axis=1)

        return Predictions

    def predict(self, Xtest):
        return applayonethreshold(self.predict_proba(Xtest), t=0.5)


if __name__ == '__main__':
    """
    X, Y = make_multilabel_classification(n_samples=10000, n_features=20, n_classes=1000)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    """
    from sklearn import tree

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
#    bmodel = tree.DecisionTreeClassifier(min_samples_leaf=1)
#    sml = ClassifierChain(bmodel)
#    sml.fit(X, Y)
#    sml.predict(X)
#    print(sml.predict(X))
    #print(sml.predict_proba(X))

    X = np.array([[411, 500, 426],
                  [100, -11, -96],
                  [125, 900, .00],
                  [100, -11, -96],
                  [125, 900, .00],
                  [.11, 60., 126],
                  [211, 100, 16],
                  [300, .60, 926],
                  [11., .00, 26],
                  [.11, 60., 126],
                  [211, 100, 16],
                  [300, .60, 926],
                  [100, -11, -96],
                  [100, -11, -96],
                  [125, 900, .00],
                  [.11, 60., 126],
                  [211, 100, 16],
                  [300, .60, 926],
                  [11., .00, 26],
                  [125, 900, .00],
                  [.11, 60., 126],
                  [211, 100, 16],
                  [300, .60, 926],
                  [11., .00, 26],
                  [11., .00, 26],
                  [341, 700, 126]])
    Y = np.array([[0, 0, 0],
                  [0, 0, 1],
                  [0, 1, 0],
                  [0, 0, 1],
                  [0, 1, 0],
                  [0, 1, 1],
                  [0, 0, 0],
                  [0, 0, 1],
                  [0, 1, 0],
                  [0, 1, 1],
                  [0, 0, 1],
                  [0, 1, 1],
                  [0, 1, 1],
                  [0, 1, 0],
                  [0, 0, 1],
                  [0, 1, 0],
                  [0, 1, 1],
                  [0, 0, 0],
                  [0, 1, 0],
                  [0, 1, 1],
                  [0, 1, 1],
                  [0, 1, 1],
                  [0, 0, 0],
                  [0, 1, 0],
                  [0, 1, 1],
                  [0, 1, 1]])

    Y = np.array([[0, 0, 1],
                  [0, 0, 1],
                  [0, 1, 1],
                  [0, 0, 1],
                  [0, 1, 1],
                  [0, 1, 1],
                  [0, 0, 1],
                  [0, 0, 1],
                  [0, 1, 1],
                  [0, 1, 1],
                  [0, 0, 1],
                  [0, 1, 1],
                  [0, 1, 1],
                  [0, 1, 1],
                  [0, 0, 1],
                  [0, 1, 1],
                  [0, 1, 1],
                  [0, 0, 1],
                  [0, 1, 1],
                  [0, 1, 1],
                  [0, 1, 1],
                  [0, 1, 1],
                  [0, 0, 1],
                  [0, 1, 1],
                  [0, 1, 1],
                  [0, 1, 1]])
    Y = np.ones(Y.shape)
    print(Y)
    from sklearn.datasets import make_multilabel_classification
    #X, Y = make_multilabel_classification(n_samples=100, n_features=20, n_classes=5)

    bmodel = tree.DecisionTreeClassifier()
    sml = ClassifierChain(bmodel)
    sml.fit(X, Y)
    sml.predict(X)

#    print(sml.predict_proba(X))
#    print(sml.predict(X)-sml.predict_proba(X))
#
#    print(sml.predict(X))
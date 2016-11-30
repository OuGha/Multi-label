# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 15:42:28 2016

@author: Ouadie GHARROUDI : ouadie.gharroudi@liris.cnrs.fr
"""
import copy
import numpy as np
#from ..multi_target_models.exceptions import TargetShapeError


class BinaryRelevance(object):
    """
    The multi-label BR model.

    Attributes
    ----------
    n_targets_ : Number of feature labels.

    models_ : List of models; its size is equal to n_targets_.
    """

    def __init__(self, model=None):
        self.n_targets_ = 0
        self.regressors_ = []
        self.model = model

    def fit(self, X, Y):
        """
        Build a multi-label regression model from the training set (X, Y).

        Parameters
        ----------
        model : the base-model
        X : array-like, with shape (n_samples, n_features).

        Y : array-like, with shape (n_samples, n_targets);
            The target values are real numbers as in classical regression.

        Returns
        -------
        self : object
            Returns self.
        """

        if len(Y.shape) == 1:
            self.n_targets_ = 1
            self.regressors_ = self.model.fit(X, Y)
        elif len(Y.shape) == 2:
            self.n_targets_ = int(Y.shape[1])
            self.regressors_ = []
            for i in range(self.n_targets_):
                regressor = copy.deepcopy(self.model)
                regressor = regressor.fit(X, Y[:, i])
                self.regressors_.append(regressor)
        else:
            pass
            #raise TargetShapeError("The target must be a two dimensions matrix or a vector.")

        return self

    def predict(self, Xtest):
        """
        Predict Xtest.

        Parameters
        ----------
        Xtest : array-like with shape (n_samples, n_features).

        Returns
        -------
        predictions : an array with shape (n_samples, n_targets)
                      the predict values.
        """
        nt = Xtest.shape[0]
        predictions = -1 * np.ones((nt, self.n_targets_))
        if self.n_targets_ == 1:
            predictions = self.regressors_.predict(Xtest)
            return predictions
        else:
            for i in range(self.n_targets_):
                predictions[:, i] = self.regressors_[i].predict(Xtest)
            return predictions


    def predict_proba(self, Xtest):
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
        nt = Xtest.shape[0]
        predictions = -1 * np.ones((nt, self.n_targets_))
        if self.n_targets_ == 1:
            predictions = self.regressors_.predict_proba(Xtest)
            return predictions
        else:
            for i in range(self.n_targets_):
                predictions[:, i] = self.regressors_[i].predict_proba(Xtest)
            return predictions

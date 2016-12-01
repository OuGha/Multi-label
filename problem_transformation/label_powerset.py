import numpy as np
import numpy.testing as npt

from sklearn.ensemble import RandomForestClassifier
#from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, hamming_loss, zero_one_loss

from sklearn.datasets import make_multilabel_classification



class LabelPowersetClassifier(object):

    def __init__(self,
                 base_estimator=None):
        self.base_estimator=base_estimator

    def target_transformation_(self, y, backward=False):
        """
        :param y: target to predict
        :return: string representation of this target
        """
        if not backward:
            return list(map(lambda array: "".join([str(i) for i in array]), y))
        else:
            return np.array([list(string) for string in y]).astype(int)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        _, n_labels = y.shape

        y_transform = self.target_transformation_(y)
        self.base_estimator.fit(X, y_transform)

        return self

    def predict(self, X):
        y_predt = self.base_estimator.predict(X)
        return self.target_transformation_(y_predt, backward=True)


if __name__ == '__main__':
    """
    X, Y = make_multilabel_classification(n_samples=10000, n_features=20, n_classes=1000)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    """
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


    Y = np.array([[0, 0],
                  [0, 1],
                  [0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 0],
                  [1, 1],
                  [1, 1]])


    lp = LabelPowersetClassifier(base_estimator=RandomForestClassifier(n_estimators=50))
    lp.fit(X, Y)
    print(lp.predict(X))
    """
    Y_pred = lp.predict(X_test)

    print("Zero One Loss {}".format(zero_one_loss(Y_test, Y_pred)))
    print("Hamming Loss {}".format(hamming_loss(Y_test, Y_pred)))
    print("F1 Score {}".format(f1_score(Y_test, Y_pred, average='macro')))
    """


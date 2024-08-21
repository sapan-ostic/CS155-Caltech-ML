"""
Polynomial regression model
"""
import numpy as np

class PolynomialRegression:
    """
    Polynomial regression model
    """
    def __init__(self, degree):
        """
        Initialize a Regression instance.

        :param degree: Degree of the polynomial regression.
        """
        self.degree_ = degree
        self.coefficients_ = None

    def train(self, x_train, y_train):
        """
        Fit the polynomial regression model to the training data.

        :param x_train: Input features for training [n_train_sets X n_train_samples].
        :param y_train: Target values for training [n_train_sets X n_train_samples].
        """
        self.coefficients_ = np.polyfit(x_train, y_train, self.degree_)

    def predict(self, x):
        """
        Predict target values using the fitted regression model.

        :param x: Input features for prediction [batch_size x n_train_samples].
        :return: Predicted target values.
        """
        pred_batch = np.polyval(self.coefficients_, x)
        return pred_batch

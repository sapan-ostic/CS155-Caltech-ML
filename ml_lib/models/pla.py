"""
Perceptron Learning Algorithm
"""
import numpy as np

class PerceptronLearningAlgorithm:
    """
    Perceptron Learning Algorithm.
    """
    def __init__(self, rng, n_features=2, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.weights = rng.random(size=n_features)
        self.bias = rng.random(size=n_features-1)

    def train(self, x_train, y_train):
        """
        Fit the perceptron model to the training data.

        :param x_train: Input features for training [n_train_samples x n_features].
        :param y_train: Target values for training [n_train_samples x 1].
        """
        n_points = x_train.shape[0]
        for i in range(n_points):
            y_pred = self.predict(x_train[i])

            if y_pred != y_train[i]:
                self.weights += self.learning_rate * (y_train[i]) * x_train[i]
                self.bias += self.learning_rate * (-y_train[i])

    def predict(self, x):
        """
        Predict target values using the fitted perceptron model.

        :param x: Input features for prediction [batch_size x n_train_samples].
        :return: Predicted target values.
        """
        return np.sign(self.weights.dot(x.T)- self.bias)

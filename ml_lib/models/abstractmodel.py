"""
Abstract class for models
"""

class AbstractModel:
    """
    Abstract class for models
    """
    def __init__(self):
        """
        Initialize the model
        """

    def train(self, x, y):
        """
        Fit the model with the given data

        Parameters
        ----------
        x : array-like of shape (n_samples, n_features)
            The input data
        y : array-like of shape (n_samples,)
            The target values

        Returns
        -------
        self : object
        """
        raise NotImplementedError("fit method is not implemented")

    def predict(self, x):
        """
        Predict the target values

        Parameters
        ----------
        x : array-like of shape (n_samples, n_features)
            The input data

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            The predicted target values
        """
        raise NotImplementedError("predict method is not implemented")

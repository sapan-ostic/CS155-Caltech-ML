"""
This file contains the class MetricsCalculator which is used
to calculate metrics for regression models.
"""

import numpy as np

class MetricsCalculator:
    """
        Class to calculate metrics for regression models.
    """

    @staticmethod
    def compute_rmse(y_pred, y):
        """
        Computes the mean square error between the predicted and the ground truth value
        """
        error = np.sqrt(np.mean((y_pred - y) ** 2))
        return error

    @staticmethod
    def compute_squared_bias(x, y_pred, y):
        """
        Computes the squared bias between the average prediction and the true values.
        :param y_pred: Prediction values for all samples with dimension [kFold x samples x features]
        :param y: True values for all samples with dimension [kFold x samples x features]
        :return bias_sq: Squared bias at each sample point [1 x samples]
        :return mean_bias_sq: Average of squared bias across all samples [1]
        """
        ## Find the prediction values corresponding to each sample
        y_pred_dict = {}
        for ifold in range(x.shape[0]):
            for sample in range(x.shape[1]):
                if x[ifold, sample] in y_pred_dict:
                    y_pred_dict[x[ifold, sample]].append(y_pred[ifold, sample])
                else:
                    y_pred_dict[x[ifold, sample]] = [y_pred[ifold, sample]]

        # Compute the average of prediction values corresponding to each sample
        mean_predictions = {x: np.mean(y_preds) for x, y_preds in y_pred_dict.items()}

        ## Find the ground truth value corresponding to each sample
        y_dict = {}
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if x[i, j] not in y_dict:
                    y_dict[x[i, j]] = y[i, j]

        # Check if the keys corresponding to the prediction and ground truth values are same
        if y_dict.keys() != mean_predictions.keys():
            raise ValueError("Dictionaries have different sets of keys.")

        x_values = np.array(list(y_dict.keys()))

        # Compute the bias corresponding to each sample
        bias_sq = []
        for key, y_value in y_dict.items():
            error = (y_value - mean_predictions[key]) ** 2
            bias_sq.append(error)
        bias_sq = np.array(bias_sq)

        # Compute the mean bias corresponsing to the batch
        mean_bias_sq = np.mean(bias_sq, axis=0)

        # Sort x_values & mean_bias based on x_values
        x_sorted_indices = np.argsort(x_values)
        x_values = x_values[x_sorted_indices]
        bias_sq = bias_sq[x_sorted_indices]

        return x_values, bias_sq, mean_bias_sq

    @staticmethod
    def compute_variance(x, y_pred):
        """
        Computes how much the predictions vary for a given data point across different train sets.
        :param y_pred: Prediction values for all samples with dim [train_sets x samples x features]
        :return variance: Variance at each sample point [samples x 1]
        :return mean_variance: Average of variance across all samples [1]
        """
        ## Find the prediction values corresponding to each sample
        y_pred_dict = {}
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if x[i, j] in y_pred_dict:
                    y_pred_dict[x[i, j]].append(y_pred[i, j])
                else:
                    y_pred_dict[x[i, j]] = [y_pred[i, j]]

        # Compute the average of prediction values corresponding to each sample
        mean_predictions = {x: np.mean(y_preds) for x, y_preds in y_pred_dict.items()}

        x_values = np.array(list(mean_predictions.keys()))

        # Compute the bias corresponding to each sample
        variance = []
        for key, y_pred_value in y_pred_dict.items():
            error = np.mean((y_pred_value - mean_predictions[key]) ** 2)
            variance.append(error)
        variance = np.array(variance)

        # Compute the mean bias corresponsing to the batch
        mean_variance = np.mean(variance, axis=0)

        # Sort x_values & mean_bias based on x_values
        x_sorted_indices = np.argsort(x_values)
        x_values = x_values[x_sorted_indices]
        variance = variance[x_sorted_indices]

        return x_values, variance, mean_variance

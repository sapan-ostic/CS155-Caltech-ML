"""
This class is used to evaluate the models for training and testing data.
"""
import numpy as np
import matplotlib.pyplot as plt
from ml_lib.metrics.metrics_calculator import MetricsCalculator
from ml_lib.utils.kfold import KFold

class ModelEvaluator:
    """
    Class to evaluate the models for training and testing data.
    """
    def __init__(self, models, x_train, y_train, x_test, y_test, n_training, n_testing, n_folds=5):
        """
        Initializes the ModelEvaluator class with the given parameters.

        Parameters:
        - models: List of models to be evaluated.
        - x_train: Input features for training [n_train_sets X n_train_samples].
        - y_train: Target values for training [n_train_sets X n_train_samples].
        - x_test: Input features for testing [n_test_sets X n_test_samples].
        - y_test: Target values for testing [n_test_sets X n_test_samples].
        - n_training: Number of training samples.
        - n_testing: Number of testing samples.
        - n_folds: Number of folds for cross-validation.
        """
        self.models = models
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.n_folds = n_folds
        self.n_training = n_training
        self.n_testing = n_testing
        self.n_samples = n_training + n_testing
        self.kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        self.initialize_accumulators()

    def evaluate_models(self, plotter, save_fig=False):
        """
        Evaluates the models for training and testing data.
        """
        rmse_all_model = []
        bias_all_model = []
        variance_all_model = []
        general_error_all_model = []
        for model in self.models:
            self.evaluate_model(model)

            if save_fig:
                self.plot_data(plotter, str(model.degree_))

            # Compute the bias and variance for the test data for each model
            mean_rmse = self.compute_rmse()
            mean_bias = self.compute_bias()
            mean_variance = self.compute_variance()
            mean_general_error = np.array(mean_bias) + np.array(mean_variance)

            # Append the RMSE, bias, variance, and general error for each model
            rmse_all_model.append(mean_rmse)
            bias_all_model.append(mean_bias)
            variance_all_model.append(mean_variance)
            general_error_all_model.append(mean_general_error)

        return (np.array(rmse_all_model), np.array(bias_all_model),
               np.array(variance_all_model), np.array(general_error_all_model))

    def initialize_accumulators(self):
        """
        Initializes the accumulators for the predictions and ground truth values for each fold.
        """
        # get number of samples in each training fold
        n_training_fold_samples = (self.n_training // self.n_folds)*(self.n_folds - 1)
        n_validation_fold_samples = self.n_training // self.n_folds
        n_test_fold_samples = self.n_testing

        # Accumulate all samples for each fold [N_FOLD X n_train_samples X n_features]
        self.x_train_folds = np.empty((0, n_training_fold_samples), np.float64)
        self.train_preds_folds = np.empty((0, n_training_fold_samples), np.float64)
        self.y_train_folds = np.empty((0, n_training_fold_samples), np.float64)

        # Accumulate the predictions for each fold [N_FOLD X n_val_samples X n_features]
        self.x_val_folds = np.empty((0, n_validation_fold_samples), np.float64)
        self.val_preds_folds = np.empty((0, n_validation_fold_samples), np.float64)
        self.y_val_folds = np.empty((0, n_validation_fold_samples), np.float64)

        # Accumulate the ground truch values for each fold [N_FOLD X n_test_samples X n_features]
        self.x_test_folds = np.empty((0, n_test_fold_samples), np.float64)
        self.test_preds_folds = np.empty((0, n_test_fold_samples), np.float64)
        self.y_test_folds = np.empty((0, n_test_fold_samples), np.float64)

    def evaluate_model(self, model):
        """
        Evaluates the model for training and testing data.
        """

        # Initialize the accumulators for the predictions and ground truth values for each fold
        self.initialize_accumulators()

        for train_fold_indices, val_fold_indices in self.kf.split(self.x_train):
            # Get the k-fold training and validation dataset
            x_train_fold = self.x_train[train_fold_indices]
            y_train_fold = self.y_train[train_fold_indices]
            x_val_fold = self.x_train[val_fold_indices]
            y_val_fold = self.y_train[val_fold_indices]

            # Train the model using the k-fold training dataset
            model.train(x_train_fold, y_train_fold)

            # Get the predictions for the training, validation sets and test set
            # using the trained model
            y_train_fold_pred = model.predict(x_train_fold)
            y_val_fold_pred = model.predict(x_val_fold)
            y_test_pred = model.predict(self.x_test)

            # Gather all samples
            self.x_train_folds = np.vstack((self.x_train_folds, x_train_fold))
            self.x_val_folds = np.vstack((self.x_val_folds, x_val_fold))
            self.x_test_folds = np.vstack((self.x_test_folds, self.x_test))

            # Gather all predictions
            self.train_preds_folds = np.vstack((self.train_preds_folds, y_train_fold_pred))
            self.val_preds_folds = np.vstack((self.val_preds_folds, y_val_fold_pred))
            self.test_preds_folds = np.vstack((self.test_preds_folds, y_test_pred))

            # Gather all ground truth values
            self.y_train_folds = np.vstack((self.y_train_folds, y_train_fold))
            self.y_val_folds = np.vstack((self.y_val_folds, y_val_fold))
            self.y_test_folds = np.vstack((self.y_test_folds, self.y_test))

    def compute_rmse(self):
        """
        Computes the RMSE for the test data for a model.
        """
        train_rmse = MetricsCalculator.compute_rmse(self.y_train_folds, self.train_preds_folds)
        val_rmse = MetricsCalculator.compute_rmse(self.y_val_folds, self.val_preds_folds)
        test_rmse = MetricsCalculator.compute_rmse(self.y_test_folds, self.test_preds_folds)
        return [train_rmse, val_rmse, test_rmse]

    def compute_bias(self):
        """
        Computes the bias and variance for the test data for a model.
        """
        _, _, train_mean_bias_sq = MetricsCalculator.compute_squared_bias(self.x_train_folds,
                                                                          self.train_preds_folds,
                                                                          self.y_train_folds)
        _, _, val_mean_bias_sq = MetricsCalculator.compute_squared_bias(self.x_val_folds,
                                                                        self.val_preds_folds,
                                                                        self.y_val_folds)
        _, _, test_mean_bias_sq = MetricsCalculator.compute_squared_bias(self.x_test_folds,
                                                                         self.test_preds_folds,
                                                                         self.y_test_folds)

        return [train_mean_bias_sq, val_mean_bias_sq, test_mean_bias_sq]

    def compute_variance(self):
        """
        Computes the variance for the test data for a model.
        """
        _, _, train_mean_variance = MetricsCalculator.compute_variance(self.x_train_folds, self.train_preds_folds)
        _, _, val_mean_variance = MetricsCalculator.compute_variance(self.x_val_folds, self.val_preds_folds)
        _, _, test_mean_variance = MetricsCalculator.compute_variance(self.x_test_folds, self.test_preds_folds)
        return [train_mean_variance, val_mean_variance, test_mean_variance]

    # plot the data
    def plot_data(self, plotter, model_name):
        """
        Plots the training and testing data points
        """

        fig, _ = plt.subplots(figsize=plotter.figsize)

        # for each fold plot the regression line
        for i_fold in range(self.n_folds):
            # Concatenate the train, validation and test data points
            x = np.concatenate((self.x_train_folds[i_fold], self.x_val_folds[i_fold], self.x_test_folds[i_fold]))
            y = np.concatenate((self.y_train_folds[i_fold], self.y_val_folds[i_fold], self.y_test_folds[i_fold]))
            y_pred = np.concatenate((self.train_preds_folds[i_fold], self.val_preds_folds[i_fold], self.test_preds_folds[i_fold]))

            # Sort the data points based on x values
            sorted_indices = np.argsort(x)
            x = x[sorted_indices]
            y = y[sorted_indices]
            y_pred = y_pred[sorted_indices]
            plotter.plot_regression(x, y_pred, f"Regression Line for Model {model_name}", fig)

        # for each fold plot the data points
        plotter.plot_data(self.x_train_folds,
                          self.y_train_folds,
                          self.x_test_folds,
                          self.y_test_folds,
                          fig)

        # Set axes limits
        plt.xlim(-11, 11)
        plt.ylim(-300, 60)

        # save the figure with name same as the title
        plt.savefig(plotter.save_path + 'Regression Model ' + model_name + '.png')

        plt.show()

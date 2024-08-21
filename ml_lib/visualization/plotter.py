"""
This file contains the Plotter class which is used to plot the data points,
regression line, and errors for different models
"""
import matplotlib.pyplot as plt
import numpy as np

TEST_COLOR = "#E57873" #Red
TRAIN_COLOR = "#94CAE0" #Blue
VALIDATION_COLOR = "#FF9D26" #Orange
POLYNOMIAL_MODELS_LABEL = 'Polynomial Models with Degree'

class Plotter:
    """
    Class to plot the data points, regression line, and errors for different models
    """

    def __init__(self, figsize = (10, 6), model_name = None, save_path = None) -> None:
        self.figsize = figsize
        self.model_name = model_name
        self.save_path = save_path

    def plot_data(self, x_train, y_train, x_test, y_test, fig = None, save_png=False):
        """
        Plots the training and testing data points
        param x_train: Training data points [N_TRAIN_SAMPLES x features]
        param y_train: Training target values [N_TRAIN_SAMPLES x features]
        param x_test: Testing data points [N_TEST_SAMPLES x features]
        param y_test: Testing target values [N_TEST_SAMPLES x features]
        return fig, ax: Figure and axis of the plot
        """
        # Check if the figure exists if not create a new figure
        if fig is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            ax = fig.axes[0]

        ax.scatter(x_train, y_train, color=TRAIN_COLOR, label="Train", s = 150, zorder=2, edgecolor='white')
        ax.scatter(x_test, y_test, color=TEST_COLOR, label="Test", s = 150, zorder=2, edgecolor='white')
        ax.legend()
        plt.xlabel("x")
        plt.ylabel("y")
        if save_png and self.save_path is not None:
            png_file = self.save_path + "data.png"
            # Save the plot as a png file
            plt.savefig(png_file)
        return fig

    def plot_regression(self, x, y, title, fig = None, save_png=False):
        """
        Plots the regression line along with the data points for each fold
        param x: Data points [N_SAMPLES x features]
        param y: Target values [N_SAMPLES x features]
        param title: Title of the plot
        """
        # Check if the figure exists if not create a new figure
        if fig is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            plt.figure(fig)
            ax = fig.axes[0]

        ax.plot(x, y, color=VALIDATION_COLOR, label=None, alpha=0.7, linewidth=3, zorder=1)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(title)
        if save_png and self.save_path is not None:
            png_file = self.save_path + "regression.png"
            # Save the plot as a png file
            plt.savefig(png_file)
        return fig

    def plot_rmse(self, rmse_all_model, save_png=False):
        """
        Plots the RMSE values for the training, validation, and testing data for different models
        param rmse_all_model: (Train, Validation, Test) RMSE values for all data [n_models x 3]
        param model_name: Name of the models [n_models x 1]
        """
        # For every model plot the train, validation and test error
        plt.figure(figsize=self.figsize)
        bar_width = 0.2
        index = np.arange(len(self.model_name))

        bar1 = plt.bar(index, rmse_all_model[:, 0], bar_width, label='Train Error', color=TRAIN_COLOR)
        bar2 = plt.bar(index + bar_width, rmse_all_model[:, 1], bar_width, label='Validation Error', color=VALIDATION_COLOR)
        bar3 = plt.bar(index + 2 * bar_width, rmse_all_model[:, 2], bar_width, label='Test Error', color=TEST_COLOR)

        # Adding the values on top of the bars
        for plot_bar in bar1 + bar2 + bar3:
            height = plot_bar.get_height()
            plt.text(plot_bar.get_x() + plot_bar.get_width() / 2.0, height, f'{height:.0f}',
                     ha='center', va='bottom')

        plt.xlabel(POLYNOMIAL_MODELS_LABEL)
        plt.ylabel('Error Value')
        plt.title('Train, Validation, and Test Errors for Different Models')
        plt.xticks(index + bar_width, self.model_name)
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        if save_png and self.save_path is not None:
            png_file = self.save_path + "rsme.png"
            # Save the plot as a png file
            plt.savefig(png_file)
        plt.show()

    def plot_bias_variance(self, bias_all_model, variance_all_model, general_error_all_model, save_png=False):
        """
        Plots the bias, variance, and general error for the test data for different models
        param model_name: Name of the models [n_models x 1]
        param bias_all_model: Bias values for test data [n_models x 1]
        param variance_all_model: Variance values for test data [n_models x 1]
        param general_error_all_model: General error values for test data [n_models x 1]
        """
        plt.figure(figsize=self.figsize)
        plt.plot(self.model_name, bias_all_model[:, 2], label='Bias', linewidth=3)
        plt.plot(self.model_name, variance_all_model[:, 2], label='Variance', linewidth=3)
        plt.plot(self.model_name, general_error_all_model[:, 2], label='General Error', linewidth=3)
        plt.legend()
        plt.xlabel(POLYNOMIAL_MODELS_LABEL)
        plt.ylabel('Error Value')
        plt.title('Bias, Variance for Different Models for test data')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        if save_png and self.save_path is not None:
            png_file = self.save_path + "bias_variance.png"
            plt.savefig(png_file)
        plt.show()

    # Plot the bias for training, validation and test data for different models
    def plot_bias(self, bias_all_model, save_png=False):
        """
        Plots the bias for training, validation, and test data for different models
        param model_name: Name of the models [n_models x 1]
        param bias_all_model: Bias values for test data [n_models x 3]
        """
        plt.figure(figsize=self.figsize)
        plt.plot(self.model_name, bias_all_model[:, 0], label='Train Bias', linewidth=3, color=TRAIN_COLOR)
        plt.plot(self.model_name, bias_all_model[:, 1], label='Validation Bias', linewidth=3, color=VALIDATION_COLOR)
        plt.plot(self.model_name, bias_all_model[:, 2], label='Test Bias', linewidth=3, color=TEST_COLOR)
        plt.legend()
        plt.xlabel(POLYNOMIAL_MODELS_LABEL)
        plt.ylabel('Bias Value')
        plt.title('Bias for Different Models and data')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        if save_png and self.save_path is not None:
            png_file = self.save_path + "bias.png"
            plt.savefig(png_file)
        plt.show()

    # Plot the variance for training, validation and test data for different models
    def plot_variance(self, variance_all_model, save_png=False):
        """
        Plots the variance for training, validation, and test data for different models
        param model_name: Name of the models [n_models x 1]
        param variance_all_model: Variance values for test data [n_models x 3]
        """
        plt.figure(figsize=self.figsize)
        plt.plot(self.model_name, variance_all_model[:, 0], label='Train Variance', linewidth=3, color=TRAIN_COLOR)
        plt.plot(self.model_name, variance_all_model[:, 1], label='Validation Variance', linewidth=3, color=VALIDATION_COLOR)
        plt.plot(self.model_name, variance_all_model[:, 2], label='Test Variance', linewidth=3, color=TEST_COLOR)
        plt.legend()
        plt.xlabel(POLYNOMIAL_MODELS_LABEL)
        plt.ylabel('Variance Value')
        plt.title('Variance for Different Models and data')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        if save_png and self.save_path is not None:
            png_file = self.save_path + "variance.png"
            plt.savefig(png_file)
        plt.show()

    # Plot the general error for training, validation and test data for different models
    def plot_general_error(self, general_error_all_model, save_png=False):
        """
        Plots the general error for training, validation, and test data for different models
        param model_name: Name of the models [n_models x 1]
        param general_error_all_model: General error values for test data [n_models x 3]
        """
        plt.figure(figsize=self.figsize)
        plt.plot(self.model_name, general_error_all_model[:, 0], label='Train General Error', linewidth=3, color=TRAIN_COLOR)
        plt.plot(self.model_name, general_error_all_model[:, 1], label='Validation General Error', linewidth=3, color=VALIDATION_COLOR)
        plt.plot(self.model_name, general_error_all_model[:, 2], label='Test General Error', linewidth=3, color=TEST_COLOR)
        plt.legend()
        plt.xlabel(POLYNOMIAL_MODELS_LABEL)
        plt.ylabel('General Error Value')
        plt.title('General Error for Different Models and data')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        if save_png and self.save_path is not None:
            png_file = self.save_path + "general_error.png"
            plt.savefig(png_file)
        plt.show()

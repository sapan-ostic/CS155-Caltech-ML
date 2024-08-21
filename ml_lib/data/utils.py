"""
Utility functions for generating and splitting data
"""

import numpy as np
rng = np.random.default_rng(seed=21)  # Set the seed for the random generator

def check_if_numpy_array(data):
    """
    Checks if the provided data is a NumPy array
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("The provided data is not a NumPy array")

def get_2d_data(n_points, coeffs, data_range):
    """
    Generates the 2D data required for training and testing
    :param n_points: Number of samples to generate
    :coeffs: [a, b, c] Coefficients of the quadratic function, y(x) = ax^2 + bx + c:
    :data_range: Range of data to be generated
    """
    noise = np.array([8*rng.uniform(-5, 5) for _ in range(n_points)])
    x = np.linspace(data_range[0], data_range[1], num = n_points)

    # Get coeefs by generating n random floating values between -10 and 10
    y = 0.5*np.polyval(coeffs, x)*np.sin(coeffs[0]*x) + noise

    check_if_numpy_array(x)
    check_if_numpy_array(y)

    return x, y

def split_train_test(x, y, n_train_points):
    """
    Splits the data into training set and a test set
    return
    """
    n_points = x.shape[0]

    # Get Train data
    train_indices = rng.choice(n_points, size=n_train_points, replace=False)
    train_indices = np.sort(train_indices)
    x_train = x[train_indices]
    y_train = y[train_indices]

    # Get test data
    x_test = np.delete(x, train_indices)
    y_test = np.delete(y, train_indices)

    check_if_numpy_array(x_train)
    check_if_numpy_array(y_train)
    check_if_numpy_array(x_test)
    check_if_numpy_array(y_test)

    return x_train, y_train, x_test, y_test

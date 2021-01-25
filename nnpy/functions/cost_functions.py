import numpy as np


def mse(actual, predicted):
    """
    https://ml-cheatsheet.readthedocs.io/en/latest/linear_regression.html#cost-function
    MSE = the mean of (actual_outcome - predicted_outcome) squared
    """
    return np.mean(np.power(actual - predicted, 2))


def prime_mse(actual, predicted):
    """
    Derivate of mse for use with the Gradient Descent optimizer.
    Output error is calculated either as the result of the derivative
    of the cost function or as the result of the backpropagation
    of the previous layer.
    """
    return 2 * (predicted - actual) / actual.size

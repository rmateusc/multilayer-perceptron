"""
Cost function for Neural Network.
author: @rmateusc

Contains:
    - Cross Entropy Cost
    - Mean Squared Error Cost
    - Quadratic Cost
    - Absolute Error Cost
"""
import numpy as np

def cross_entropy(y_pred, y_true):
    """Computes the cross-entropy cost given in equation

    Arguments:
    y_pred -- The sigmoid output of the second activation, of shape (1, number of examples)
    y_true -- "true" labels vector of shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost given equation
    """
    m = y_true.shape[1] # number of examples

    # Compute the cross-entropy cost
    cost = -1/m * np.sum(y_true * np.log(y_pred) + (1-y_true) * np.log(1 - y_pred))

    return cost

def cross_entropy_derivative(y_pred, y_true):
    """Computes the derivative of the cross-entropy cost function

    Arguments:
    y_pred -- The sigmoid output of the second activation, of shape (1, number of examples)
    y_true -- "true" labels vector of shape (1, number of examples)

    Returns:
    dZ -- Gradient of the cost with respect to Z1, of shape (1, number of examples)
    """
    # Compute the derivative of the cross-entropy cost
    dZ = y_pred - y_true

    return dZ

def mean_squared_error(y_pred, y_true):
    """Computes the mean squared error cost given in equation

    Arguments:
    y_pred -- The sigmoid output of the second activation, of shape (1, number of examples)
    y_true -- "true" labels vector of shape (1, number of examples)

    Returns:
    cost -- mean squared error cost given equation
    """
    m = y_true.shape[1] # number of examples

    # Compute the mean squared error cost
    cost = 1/m * np.sum(np.square(y_pred - y_true))

    return cost

def mean_squared_error_derivative(y_pred, y_true):
    """Computes the derivative of the mean squared error cost function

    Arguments:
    y_pred -- The sigmoid output of the second activation, of shape (1, number of examples)
    y_true -- "true" labels vector of shape (1, number of examples)

    Returns:
    dZ -- Gradient of the cost with respect to Z1, of shape (1, number of examples)
    """
    m = y_true.shape[1] # number of examples

    # Compute the derivative of the mean squared error cost
    dZ = 2/m * (y_pred - y_true)

    return dZ

def quadratic_cost(y_pred, y_true):
    """Computes the quadratic cost given in equation

    Arguments:
    y_pred -- The sigmoid output of the second activation, of shape (1, number of examples)
    y_true -- "true" labels vector of shape (1, number of examples)

    Returns:
    cost -- quadratic cost given equation
    """
    # Compute the quadratic cost
    cost = 1/2 * np.sum(np.square(y_true - y_pred))

    return cost

def quadratic_cost_derivative(y_pred, y_true):
    """Computes the derivative of the quadratic cost function

    Arguments:
    y_pred -- The sigmoid output of the second activation, of shape (1, number of examples)
    y_true -- "true" labels vector of shape (1, number of examples)

    Returns:
    dZ -- Gradient of the cost with respect to Z1, of shape (1, number of examples)
    """
    # Compute the derivative of the quadratic cost
    dZ = (y_true - y_pred)

    return dZ

def absolute_error(y_pred, y_true):
    """Computes the absolute error cost given in equation

    Arguments:
    y_pred -- The sigmoid output of the second activation, of shape (1, number of examples)
    y_true -- "true" labels vector of shape (1, number of examples)

    Returns:
    cost -- absolute error cost given equation
    """
    m = y_true.shape[1] # number of examples

    # Compute the absolute error cost
    cost = np.sum(np.abs(y_pred - y_true))
    cost = float(np.squeeze(cost))

    return cost

def absolute_error_derivative(y_pred, y_true):
    """Computes the derivative of the absolute error cost function

    Arguments:
    y_pred -- The sigmoid output of the second activation, of shape (1, number of examples)
    y_true -- "true" labels vector of shape (1, number of examples)

    Returns:
    dZ -- Gradient of the cost with respect to Z1, of shape (1, number of examples)
    """
    m = y_true.shape[1] # number of examples

    # Compute the derivative of the absolute error cost
    dZ = np.sign(y_pred - y_true)

    return dZ

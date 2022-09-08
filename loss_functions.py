"""
Loss functions for Neural Network.
author: @rmateusc

Contains:
    - Quadratic Loss
"""
import numpy as np

def quadratic_loss(y_true, y_pred):
    """Computes the quadratic cost given in equation

    Arguments:
    y_true -- "true" labels vector of shape (1, number of examples)
    y_pred -- The sigmoid output of the second activation, of shape (1, number of examples)

    Returns:
    cost -- quadratic cost given equation
    """
    # Compute the quadratic cost
    cost = 1/2 * np.sum(np.square(y_true - y_pred))

    return cost

def quadratic_loss_derivative(y_true, y_pred):
    """Computes the derivative of the quadratic cost function

    Arguments:
    y_true -- "true" labels vector of shape (1, number of examples)
    y_pred -- The sigmoid output of the second activation, of shape (1, number of examples)

    Returns:
    dL -- derivative of the quadratic loss function
    """
    # Compute the derivative of the quadratic cost
    dL = -(y_true - y_pred)

    return dL

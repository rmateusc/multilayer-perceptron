"""
Activation functions for Neural Network.
@Author: rmateusc

Contains:
    - Linear
    - Sigmoid
    - Tanh
    - ReLU
"""
import numpy as np

def linear(a:float, b:float, x:float) -> float:
    """Compute linear function of v.
    """
    z = a*x + b
    return z

def linear_derivative(a:float, b:float, x:float) -> float:
    """Compute linear derivative of v.
    """
    dz = a
    return dz

def sigmoid(x: float) -> float:
    """Compute sigmoid of x.
    """
    z = 1/(1+np.exp(-x))
    return z

def sigmoid_derivative(x: float) -> float:
    """Compute sigmoid derivative of x.
    """
    s = sigmoid(x) * (1 - sigmoid(x))
    return s

def tanh(x:float) -> float:
    """Compute tanh of x.
    """
    z = np.tanh(x)
    return z

def tanh_derivative(x:float) -> float:
    """Compute tanh derivative of x.
    """
    s = 1 - tanh(x)**2
    return s

def relu(x:float) -> float:
    """Compute relu of x.
    """
    z = np.maximum(0, x)
    return z

def relu_derivative(x:float) -> float:
    """Compute relu derivative of x.
    """
    dz = np.where(x > 0, 1, 0)
    return dz

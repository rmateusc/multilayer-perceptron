"""
Implement a MultiNode Perceptron Layer.
"""
from layer import Layer
from activation_functions import *
import numpy as np

# Fully connected layer
class PerceptronLayer(Layer):
    def __init__(
        self, input_size: int, output_size: int, activation_function: str
        ):
        # Initialize parameters: weights and bias
        self.weights = np.random.randn(output_size, input_size)
        # Define activation function and derivate
        if activation_function == 'linear':
            self.activation_function = linear
            self.activation_function_derivative = linear_derivative
        elif activation_function == 'sigmoid':
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        elif activation_function == 'tanh':
            self.activation = tanh
            self.activation_derivative = tanh_derivative
        elif activation_function == 'relu':
            self.activation = relu
            self.activation_derivative = relu_derivative
        else:
            raise ValueError('Unknown activation function')

    def forward_propagation(self, inputs):
        # Input of the layer
        self.input = inputs
        # Local field of the layer
        self.local_field = np.dot(self.weights, inputs)
        # Output of the layer
        self.output = self.activation(self.local_field)
        # Return output
        return self.output

    def backward_propagation(self, previous_gradient):
        # Calculate local field gradient
        local_field_gradient = (
            previous_gradient * self.activation_derivative(self.local_field)
        )
        self.weights_delta = np.dot(local_field_gradient, self.input.T)
        # Calculate local gradient
        local_gradient = np.dot(self.weights.T, local_field_gradient)
        # Return local gradient
        return local_gradient

    def update_weights(self, learning_rate):
        # Update weights
        self.weights -= learning_rate * self.weights_delta

    def gradients_cache(self):
        # Return gradients cache
        return self.weights_delta

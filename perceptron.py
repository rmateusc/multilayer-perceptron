"""
Implement a Multinode Perceptron Layer.
"""
from layer import Layer
from activation_functions import sigmoid, sigmoid_derivative
import numpy as np

# Fully connected layer
class Perceptron:
    def __init__(self, input_size, output_size):
        # Initialize parameters: weights and bias
        self.weights = np.random.randn(input_size, output_size)
        self.bias = np.zeros(output_size)

    def forward_propagation(self, inputs):
        """Forward propagation for the perceptron.

        Arguments:
        inputs -- input data

        Returns:
        output -- output of the perceptron
        """
        self.input = inputs
        self.output = np.dot(self.weights, inputs) + self.bias

        return self.output

    # def backward_propagation(self, output_error, learning_rate):
    #     """Implement the backward propagation for the perceptron.

    #     Arguments:
    #     output_error -- error of the perceptron's output
    #     learning_rate -- learning rate for the perceptron

    #     Returns:
    #     grads -- python dictionary with gradients with respect to different parameters
    #     """

    #     # Retrieve from cache
    #     Yi = cache['Yi']
    #     Yh = cache['Yh']
    #     Yo = cache['Yo']
    #     Vi = cache['Vi']
    #     Vh = cache['Vh']
    #     Vo = cache['Vo']

    #     # Local gradient for output layer
    #     localgrad_o = np.multiply((Yo - Y), sigmoid_derivative(Vo))
    #     # Local gradient for hidden layer
    #     localgrad_h = np.multiply(localgrad_o, sigmoid_derivative(Vh)) * Wo
    #     # Local gradient for input layer
    #     localgrad_i = np.multiply(localgrad_h, sigmoid_derivative(Vi)) * Wh

    #     # Delta for weights of output layer
    #     delta_o = - learning_rate * np.dot(localgrad_o, Yh.T)
    #     # Delta for weights of hidden layer
    #     delta_h = - learning_rate * np.dot(localgrad_h, Yi)
    #     # Delta for weights of input layer
    #     delta_i = - learning_rate * np.dot(localgrad_i, X.T)

    #     # Update parameters
    #     Wi = Wi + delta_i
    #     Wh = Wh + delta_h
    #     Wo = Wo + delta_o

    #     gradients = {
    #         'localgrad_i': localgrad_i,
    #         'localgrad_h': localgrad_h,
    #         'localgrad_o': localgrad_o
    #         }

    #     parameters = {
    #         'Wi': Wi,
    #         'Wh': Wh,
    #         'Wo': Wo
    #         }

    #     return gradients, parameters

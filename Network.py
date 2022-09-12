"""
Neural Netork class.
"""
import numpy as np
from turtle import forward
from loss_functions import (
    quadratic_mean_loss,
    quadratic_mean_loss_derivative
)

class Network:
    def __init__(self, layers, loss_function):
        self.layers = layers
        if loss_function == 'quadratic':
            self.loss = quadratic_mean_loss
            self.loss_derivative = quadratic_mean_loss_derivative
        else:
            raise ValueError('Unknown loss function')

    def predict(self, input):
        # Predict output for given input
        samples, features = input.shape
        results = []
        # Run for every row
        for i in range(samples):
            # forward propagation
            input_row = input[i].reshape(features, 1)
            for layer in self.layers:
                output = layer.forward_propagation(input_row)
            results.append(output)
        return results

    def fit(self, x_train, y_train, learning_rate, epochs, verbose=True):
        # training sample size
        samples, features = x_train.shape
        classes = y_train.shape[1]
        # train for every epoch
        for epoch in range(epochs):
            # losses for every epoch
            losses = np.empty([3, 1])
            # for every data row
            for i in range(samples):
                x_i = x_train[i].reshape(features, 1)
                y_i = y_train[i].reshape(classes, 1)
                # forward prop for each layer
                layers_inputs = [x_i]
                for layer in self.layers:
                    layer_output = layer.forward_propagation(layers_inputs[-1])
                    layers_inputs.append(layer_output)

                output = layer_output
                # calculate loss
                loss = self.loss(y_i, output)
                losses = np.append(losses, loss, axis=0)

                # back prop for each layer
                previous_grads = [self.loss_derivative(y_i, output)]

                for layer in self.layers[::-1]:
                    local_grads = layer.backward_propagation(previous_grads[-1])
                    previous_grads.append(local_grads)
                    layer.update_weights(learning_rate)

                    grads = layer.gradients_cache()

            # Calculate average loss for each epoch
            print(f'Epoch [{epoch+1}|{epochs}]:')
            print(f'  Loss: {losses.mean()}')

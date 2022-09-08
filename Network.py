"""
Neural Netork class.
"""
from loss_functions import *
class Network:
    def __init__(self, layers, loss_function, learning_rate):
        self.layers = layers
        self.learning_rate = learning_rate
        if loss_function == 'quadratic':
            self.loss = quadratic_loss
            self.loss_derivative = quadratic_loss_derivative
        else:
            raise ValueError('Unknown loss function')

    def predict(self, input):
        # Predict output for given input
        results = []
        # Forward propagation
        for i in range(input.shape[1]):
            input_i = input[i].reshape((1, input[i].shape[0]))
            for layer in self.layers:
                output = layer.forward_propagation(input_i)
            results.append(output)
        return results

    # Train the network
    def fit(self, x_train, y_train, learning_rate, epochs, verbose=True):
        # training sample size
        x_size = x_train.shape[0]

        for epoch in range(epochs):
            losses = []
            for i in range(x_size):
                x_i = x_train[i].reshape((1, x_train[i].shape[0]))
                y_i = y_train[:, i]
                for layer in self.layers:
                    output = layer.forward_propagation(x_i)

                loss = self.loss(y_i, output)
                losses.append(loss)

                # Backward propagation
                error = self.loss_derivative(y_i, output)
                for layer in self.layers:
                    error = layer.backward_propagation(error, learning_rate)

            # Calculate average loss for each epoch

            print(f'Epoch [{epoch+1}|{epochs}]:')
            print(f'  Loss: {1 / x_size * sum(losses)}')

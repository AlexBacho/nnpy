import math
import numpy as np

from nnpy.utils import get_random_array


class BaseLayer:
    def __init__(self):
        pass

    def forward_propagation(self, input):
        raise NotImplementedError

    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError


class InputLayer(BaseLayer):
    def forward_propagation(self, input_data):
        pass

    def backward_propagation(self, output_error, learning_rate):
        pass


class OutputLayer(BaseLayer):
    def forward_propagation(self, input_data):
        pass

    def backward_propagation(self, output_error, learning_rate):
        pass


# Hidden Layers
class SineLayer(BaseLayer):
    def forward_propagation(self, input_data):
        return math.sin(input_data)

    def backward_propagation(self, output_error, learning_rate):
        return math.cos(output_error)


class FullyConnectedLayer(BaseLayer):
    """
    The most widely used class type. Based on the Rosenblatt model:
    Every single perceptron from the previous layer is linked to every single perceptron of this layer.
    Weights: 2D array, m*n (num of previous and current layer perceptrons).
    Bias: 1D array with size n.
    Both initialized as random values
    """

    def __init__(self, number_of_perceptrons):
        self.number_of_perceptrons = number_of_perceptrons
        self.bias = get_random_array(1, self.number_of_perceptrons, offset=-0.5)
        self.weights = None
        self.input = None

    def forward_propagation(self, input_data):
        if self.weights is None:
            self.weights = self._get_random_weights(input_data)
        self.input = input_data.reshape((1, -1))
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error

    def _get_random_weights(self, input_data):
        return get_random_array(
            input_data.size, self.number_of_perceptrons, offset=-0.5
        )


class ActivationLayer(BaseLayer):
    """
    Activation layers are just like any other type of layer except
    they don’t have weights but use a non-linear function over the input instead.
    """

    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(input_data)
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        return self.activation_prime(self.input, self.output) * output_error
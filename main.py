import numpy as np
from functools import reduce

import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class Layer:
    def __init__(self, num_inputs, num_neurons):
        # Shape of weights is (inputs x neurons) to save doing a transpose
        self.weights = np.random.randn(num_inputs, num_neurons)
        self.biases = np.zeros((1, num_neurons))

    def __call__(self, inputs):
        return sigmoid(np.dot(inputs, self.weights) + self.biases)

class Network:
    def __init__(self, layer_sizes):
        self.layers = [Layer(*size) for size in layer_sizes]

    def __call__(self, x):
        return reduce(lambda out, layer: layer(out), self.layers, x)

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

X, y = spiral_data(100, 3)

net = Network([(2, 5)])

outputs = net(X)

print(outputs)

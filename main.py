import numpy as np
from functools import reduce

import mnist

class Layer:
    def __init__(self, num_inputs, num_neurons):
        # Shape of weights is (neurons x inputs) to save doing a transpose
        self.weights = np.random.randn(num_neurons, num_inputs)
        self.biases = np.zeros((num_neurons, 1))

    def __call__(self, inputs):
        return sigmoid(np.dot(self.weights, inputs) + self.biases)

EPOCHS = 10
BATCH_SIZE = 5
LEARNING_RATE = 0.01

class Network:
    def __init__(self, layer_sizes):
        self.layers = [Layer(*size) for size in layer_sizes]

    def __call__(self, x):
        return reduce(lambda out, layer: layer(out), self.layers, x)

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    # TODO: Fix overflow
    return 1.0 / (1.0 + np.exp(-x))

X, y = mnist.training_images(), mnist.training_labels()

num_inputs, num_outputs = X.shape[1], y.shape[1]
net = Network([(num_inputs, num_outputs)])

outputs = net(X.T)

print(outputs)

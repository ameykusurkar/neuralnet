import numpy as np
from functools import reduce

class Layer:
    def __init__(self, num_inputs, num_neurons):
        # Shape of weights is (inputs x neurons) to save doing a transpose
        self.weights = 0.1 * np.random.randn(num_inputs, num_neurons)
        self.biases = np.zeros((1, num_neurons))

    def __call__(self, inputs):
        return np.dot(inputs, self.weights) + self.biases

class Network:
    def __init__(self):
        self.layers = []

    def __call__(self, x):
        return reduce(lambda out, layer: layer(out), self.layers, x)

np.random.seed(0)

net = Network()
net.layers.append(Layer(4, 5))
net.layers.append(Layer(5, 2))

X = np.array([[1, 2, 3, 2.5],
              [2.0, 5.0, -1.0, 2.0],
              [-1.5, 2.7, 3.3, -0.8]])

outputs = net(X)

print(outputs)

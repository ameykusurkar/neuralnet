import numpy as np

class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        a = x
        for layer in self.layers:
            a = layer.forward(a)
        return a

    def backward(self, dc_da):
        for layer in reversed(self.layers):
            dc_da = layer.backward(dc_da)
        return dc_da

    def descend(self, lr):
        for layer in self.layers:
            layer.descend(lr)

class Linear:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.biases = np.zeros((output_size, 1))

    def forward(self, x):
        self.x = x # in, n
        return np.dot(self.weights, x) + self.biases # out, n

    def backward(self, dc_dz):
        # dc_dz: out, n
        dz_dw = self.x.T # n, in
        dz_dx = self.weights.T # in, out

        dc_dw = dc_dz.dot(dz_dw) / self.x.shape[1] # out, in
        dc_db = dc_dz.mean(axis=1, keepdims=True) # out, 1
        dc_dx = dz_dx.dot(dc_dz) # in, n

        self.grad = (dc_dw, dc_db)
        return dc_dx

    def descend(self, lr):
        dc_dw, dc_db = self.grad
        self.weights -= lr * dc_dw
        self.biases -= lr * dc_db

class Sigmoid:
    def forward(self, x):
        self.x = x
        return sigmoid(x)

    def backward(self, dc_dy):
        return d_sigmoid(self.x) * dc_dy

    def descend(self, lr):
        pass

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))
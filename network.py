import numpy as np

np.random.seed(0)

class Layer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.biases = np.zeros((output_size, 1))
        self.activation = Sigmoid()

    def forward(self, x):
        self.x = x # in, n
        self.z = np.dot(self.weights, x) + self.biases # out, n
        self.a = self.activation.forward(self.z) # out, n
        return self.a

    def backward(self, dc_da, lr):
        dc_dz = self.activation.backward(dc_da)
        dc_dw, dc_db, dc_dx = self.compute_linear_grad(dc_dz)

        self.weights -= lr * dc_dw
        self.biases -= lr * dc_db
        return dc_dx

    def compute_linear_grad(self, dc_dz):
        # dc_dz: out, n
        dz_dw = self.x.T # n, in 

        dc_dw = dc_dz.dot(dz_dw) / self.x.shape[1] # out, in
        dc_db = dc_dz.mean(axis=1, keepdims=True) # out, 1

        dz_dx = self.weights.T # in, out
        dc_dx = dz_dx.dot(dc_dz) # in, n

        return dc_dw, dc_db, dc_dx

class Cost:
    def forward(self, x):
        self.x = x
        return x

    def backward(self, y):
        return self.x - y

class Sigmoid:
    def forward(self, x):
        self.x = x
        return sigmoid(x)

    def backward(self, dc_dy):
        return d_sigmoid(self.x) * dc_dy

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

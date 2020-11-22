import numpy as np

np.random.seed(0)

class Layer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.biases = np.zeros((output_size, 1))

    def forward(self, x):
        self.x = x # in, n
        self.z = np.dot(self.weights, x) + self.biases # out, n
        self.a = sigmoid(self.z) # out, n
        return self.a

    def backward(self, dc_da, lr):
        dc_dw, dc_db, dc_dx = self.compute_grad(dc_da)

        self.weights -= lr * dc_dw
        self.biases -= lr * dc_db
        self.dc_dx = dc_dx

    def compute_grad(self, dc_da):
        # dc_da: out, n
        da_dz = d_sigmoid(self.z) # out, n
        dz_dw = self.x.T # n, in 
        dc_dz = dc_da * da_dz # out, n

        dc_dw = dc_dz.dot(dz_dw) / self.x.shape[1] # out, in
        dc_db = dc_dz.mean(axis=1, keepdims=True) # out, 1

        dz_dx = self.weights.T # in, out
        dc_dx = dz_dx.dot(dc_dz) # in, n

        return dc_dw, dc_db, dc_dx

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

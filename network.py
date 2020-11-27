import numpy as np

from layers import Sequential

class Network:
    def __init__(self, layers):
        self.loss = CrossEntropy
        self.sequential = Sequential(layers)

    def forward(self, x):
        self.a = self.sequential.forward(x)
        return self.a

    def backward(self, y):
        dc_da = self.loss.backward(self.a, y)
        self.sequential.backward(dc_da)

    def descend(self, lr):
        self.sequential.descend(lr)

class MeanSquaredError:
    @staticmethod
    def forward(y_hat, y):
        return 0.5 * ((y_hat - y) ** 2).sum(axis=0).mean()

    @staticmethod
    def backward(y_hat, y):
        return y_hat - y

class CrossEntropy:
    @staticmethod
    def forward(y_hat, y):
        loss = y_hat * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)
        return -loss.sum(axis=0).mean()

    @staticmethod
    def backward(y_hat, y):
        return (1 - y) / (1 - y_hat) - (y / y_hat)

import numpy as np

import mnist
import network

class Network:
    def __init__(self, layers):
        self.cost_function = network.Cost()
        self.sequential = network.Sequential(layers)

    def forward(self, x):
        """
        x: (m, n)
        """
        a = self.sequential.forward(x)
        return self.cost_function.forward(a)

    def backward(self, y):
        dc_da = self.cost_function.backward(y)
        self.sequential.backward(dc_da)

    def descend(self, lr):
        self.sequential.descend(lr)

    def accuracy(self, x, y):
        preds = self.forward(x) # j, n
        pred_labels = preds.argmax(axis=0)
        labels = y.argmax(axis=0) # y: j, n
        return (pred_labels == labels).astype(np.float).mean()

    def cost(self, x, y):
        preds = self.forward(x)
        return compute_cost(preds, y)

def compute_cost(preds, y):
    """
    preds, y: (j, n)
    """
    return 0.5 * ((preds - y) ** 2).sum(axis=0).mean()

def normalise(x):
    """
    Transform `x` to have mean 0 and std dev 1
    """
    m, s = x.mean(), x.std()
    return (x - m) / s

def pairwise(xs):
    """
    Returns a pairwise generator of a flat array.

    >>> list(pairwise([1, 2, 3, 4]))
    [(1, 2), (2, 3), (3, 4)]
    """
    return zip(xs, xs[1:])

x_train, y_train = mnist.training_images(), mnist.training_labels()
x_train = normalise(x_train)

x_test, y_test = mnist.test_images(), mnist.test_labels()
x_test = normalise(x_test)

LEARNING_RATE = 5
EPOCHS = 10
BATCH_SIZE = 10

net = Network([
    network.Linear(x_train.shape[1], 30),
    network.Sigmoid(),
    network.Linear(30, y_train.shape[1]),
    network.Sigmoid(),
])

x_train_batched = x_train.reshape(-1, BATCH_SIZE, x_train.shape[1])
y_train_batched = y_train.reshape(-1, BATCH_SIZE, y_train.shape[1])

print("Training against the MNIST dataset")
print(f"Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}, Learning rate: {LEARNING_RATE}")
for i in range(EPOCHS):
    for x_mini, y_mini in zip(x_train_batched, y_train_batched):
        net.forward(x_mini.T)
        net.backward(y_mini.T)
        net.descend(LEARNING_RATE)

    cst = net.cost(x_train.T, y_train.T)
    acc = net.accuracy(x_test.T, y_test.T)
    print(f"Epoch {i+1}/{EPOCHS}: cost = {cst:.5f}, accuracy = {acc:.3f}")

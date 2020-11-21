import numpy as np

import mnist

np.random.seed(0)

class Network:
    def __init__(self, layer_sizes):
        self.weights = []
        self.biases = []

        for l1, l2 in pairwise(layer_sizes):
            self.weights.append(np.random.randn(l2, l1))
            self.biases.append(np.zeros((l2, 1)))

    def forward(self, x):
        """
        x: (m, n)
        """
        self.z = []
        self.a = []

        for w, b in zip(self.weights, self.biases):
            a = self.a[-1] if self.a else x
            z = np.dot(w, a) + b
            self.z.append(z)
            self.a.append(sigmoid(z))

        return self.a[-1]

    def accuracy(self, x, y):
        preds = self.forward(x) # j, n
        pred_labels = preds.argmax(axis=0)
        labels = y.argmax(axis=0) # y: j, n
        return (pred_labels == labels).astype(np.float).mean()

    # TODO: Refactor to be iterative
    def nudges(self, x, y):
        """
        x: (m, n)
        y: (j, n)
        """
        dz1_dw1 = self.a[0].T # n, k
        da1_dz1 = d_sigmoid(self.z[1]) # j, n
        dc_da1 = self.a[1] - y # j, n

        # Derivative of the cost wrt to the neuron outputs `z0`
        dc_dz1 = da1_dz1 * dc_da1 # j, n

        # Derivative of the cost wrt to the weights and biases,
        # averaged over all training samples
        dc_dw1 = dc_dz1.dot(dz1_dw1) / x.shape[1]
        dc_db1 = dc_dz1.mean(axis=1, keepdims=True)

        ### Previous layer gradients

        dz1_da0 = self.weights[1].T # k, j
        dc_da0 = dz1_da0.dot(dc_dz1) # k, n

        dz0_dw0 = x.T # n, m
        da0_dz0 = d_sigmoid(self.z[0]) # k, n

        # Derivative of the cost wrt to the neuron outputs `z1`
        dc_dz0 = da0_dz0 * dc_da0 # k, n

        # Derivative of the cost wrt to the weights and biases,
        # averaged over all training samples
        dc_dw0 = dc_dz0.dot(dz0_dw0) / x.shape[1]
        dc_db0 = dc_dz0.mean(axis=1, keepdims=True)

        return (dc_dw1, dc_db1, dc_dw0, dc_db0)

    def grad_desc(self, x, y, lr):
        self.forward(x)
        d_weights1, d_biases1, d_weights0, d_biases0 = self.nudges(x, y)

        self.weights[1] -= lr * d_weights1
        self.biases[1] -= lr * d_biases1

        self.weights[0] -= lr * d_weights0
        self.biases[0] -= lr * d_biases0

    def cost(self, x, y):
        preds = self.forward(x)
        return compute_cost(preds, y)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

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

LEARNING_RATE = 3
EPOCHS = 10
BATCH_SIZE = 100

# j:10, k: 16 (hidden), m: 784
HIDDEN_SIZE = 16

net = Network([x_train.shape[1], HIDDEN_SIZE, y_train.shape[1]])

x_train_batched = x_train.reshape(-1, BATCH_SIZE, x_train.shape[1])
y_train_batched = y_train.reshape(-1, BATCH_SIZE, y_train.shape[1])

for i in range(EPOCHS):
    for x_mini, y_mini in zip(x_train_batched, y_train_batched):
        net.grad_desc(x_mini.T, y_mini.T, LEARNING_RATE)

    cst = net.cost(x_train.T, y_train.T)
    acc = net.accuracy(x_test.T, y_test.T)
    print(f"{i}: cost = {cst:.5f}, accuracy = {acc:.3f}")

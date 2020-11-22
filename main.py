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

    def nudges(self, x, y):
        """
        x: (m, n)
        y: (j, n)
        """
        dc_dw = [None] * len(self.weights)
        dc_db = [None] * len(self.biases)

        dc_dz = [None] * len(self.weights)

        for l in reversed(range(len(self.weights))):
            prev_a = self.a[l-1] if l else x # in, n
            dz_dw = prev_a.T # n, in
            da_dz = d_sigmoid(self.z[l]) # out, n
            if l == len(self.weights) - 1:
                dc_da = self.a[l] - y # out, n
            else:
                dzl1_da = self.weights[l+1].T # out, outl1
                dc_dzl1 = dc_dz[l+1] # outl1, n
                dc_da = dzl1_da.dot(dc_dzl1) # out, n

            # Derivative of the cost wrt to the neuron outputs `z` for layer `l`
            dc_dz[l] = da_dz * dc_da # out, n

            # Derivative of the cost wrt to the weights and biases,
            # averaged over all training samples
            dc_dw[l] = dc_dz[l].dot(dz_dw) / x.shape[1]
            dc_db[l] = dc_dz[l].mean(axis=1, keepdims=True)

        return (dc_dw, dc_db)

    def grad_desc(self, x, y, lr):
        self.forward(x)
        d_weights, d_biases = self.nudges(x, y)
        for l in range(len(self.weights)):
            self.weights[l] -= lr * d_weights[l]
            self.biases[l] -= lr * d_biases[l]

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

LEARNING_RATE = 5
EPOCHS = 10
BATCH_SIZE = 10

net = Network([x_train.shape[1], 30, y_train.shape[1]])

x_train_batched = x_train.reshape(-1, BATCH_SIZE, x_train.shape[1])
y_train_batched = y_train.reshape(-1, BATCH_SIZE, y_train.shape[1])

print("Training against the MNIST dataset")
print(f"Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}, Learning rate: {LEARNING_RATE}")
for i in range(EPOCHS):
    for x_mini, y_mini in zip(x_train_batched, y_train_batched):
        net.grad_desc(x_mini.T, y_mini.T, LEARNING_RATE)

    cst = net.cost(x_train.T, y_train.T)
    acc = net.accuracy(x_test.T, y_test.T)
    print(f"Epoch {i+1}/{EPOCHS}: cost = {cst:.5f}, accuracy = {acc:.3f}")

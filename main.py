import numpy as np

import mnist

np.random.seed(0)

# j:10, k: 16 (hidden), m: 784
HIDDEN_SIZE = 16

class Network:
    def __init__(self, input_size, output_size):
        self.weights0 = np.random.randn(HIDDEN_SIZE, input_size)
        self.biases0 = np.zeros((HIDDEN_SIZE, 1))

        self.weights1 = np.random.randn(output_size, HIDDEN_SIZE)
        self.biases1 = np.zeros((output_size, 1))

    def forward(self, x):
        """
        x: (m, n)
        """
        self.z0 = np.dot(self.weights0, x) + self.biases0 # k, n
        self.a0 = sigmoid(self.z0) # k, n

        self.z1 = np.dot(self.weights1, self.a0) + self.biases1 # j, n
        self.a1 = sigmoid(self.z1) # j, n
        return self.a1

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
        dz1_dw1 = self.a0.T # k, n
        da1_dz1 = d_sigmoid(self.z1) # j, n
        dc_da1 = self.a1 - y # j, n

        # Derivative of the cost wrt to the neuron outputs `z`
        dc_dz1 = da1_dz1 * dc_da1

        # Derivative of the cost wrt to the weights and biases,
        # averaged over all training samples
        dc_dw1 = dc_dz1.dot(dz1_dw1) / x.shape[1]
        dc_db1 = dc_dz1.mean(axis=1, keepdims=True)

        return (dc_dw1, dc_db1)

    def grad_desc(self, x, y, lr):
        self.forward(x)
        d_weights, d_biases = self.nudges(x, y)

        self.weights1 -= lr * d_weights
        self.biases1 -= lr * d_biases

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

x_train, y_train = mnist.training_images(), mnist.training_labels()
x_train = normalise(x_train)

x_test, y_test = mnist.test_images(), mnist.test_labels()
x_test = normalise(x_test)

net = Network(x_train.shape[1], y_train.shape[1])

LEARNING_RATE = 3
EPOCHS = 10
BATCH_SIZE = 100

x_train_batched = x_train.reshape(-1, BATCH_SIZE, x_train.shape[1])
y_train_batched = y_train.reshape(-1, BATCH_SIZE, y_train.shape[1])

for i in range(EPOCHS):
    for x_mini, y_mini in zip(x_train_batched, y_train_batched):
        net.grad_desc(x_mini.T, y_mini.T, LEARNING_RATE)

    cst = net.cost(x_train.T, y_train.T)
    acc = net.accuracy(x_test.T, y_test.T)
    print(f"{i}: cost = {cst:.5f}, accuracy = {acc:.3f}")

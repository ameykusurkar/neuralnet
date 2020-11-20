import numpy as np

import mnist

np.random.seed(0)

class Network:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.biases = np.zeros((output_size, 1))

    def forward(self, x):
        self.z = np.dot(self.weights, x.T) + self.biases # j, n
        self.a = sigmoid(self.z) # j, n
        return self.a

    def accuracy(self, x, y):
        preds = self.forward(x) # j, n
        pred_labels = preds.argmax(axis=0)
        labels = y.argmax(axis=1) # y: n, j
        return (pred_labels == labels).astype(np.float).mean()

    def nudges(self, x, y):
        d_c_wrt_a = self.a - y.T # j, n
        d_a_wrt_z = d_sigmoid(self.z) # j, n
        d_c_wrt_z = d_c_wrt_a * d_a_wrt_z # j, n

        d_weights = d_c_wrt_z.dot(x) / x.shape[0]
        d_biases = d_c_wrt_z.mean(axis=1, keepdims=True)
        return (d_weights, d_biases)

    def grad_desc(self, x, y, lr):
        self.forward(x)
        d_weights, d_biases = self.nudges(x, y)

        self.weights -= lr * d_weights
        self.biases -= lr * d_biases

    def cost(self, x, y):
        preds = self.forward(x)
        return compute_cost(preds.T, y)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

def compute_cost(preds, y):
    return 0.5 * ((preds - y) ** 2).sum(axis=1).mean()

def normalise(x):
    """
    Transform `x` to have mean 0 and std dev 1
    """
    m, s = x.mean(), x.std()
    return (x - m) / s

X, y = mnist.training_images(), mnist.training_labels()
X = normalise(X)

num_inputs, num_outputs = X.shape[1], y.shape[1]
net = Network(num_inputs, num_outputs)

x_test, y_test = mnist.test_images(), mnist.test_labels()
x_test = normalise(x_test)

LEARNING_RATE = 3
EPOCHS = 10
BATCH_SIZE = 100

X_mini_batched = X.reshape(-1, BATCH_SIZE, X.shape[1])
y_mini_batched = y.reshape(-1, BATCH_SIZE, y.shape[1])

for i in range(EPOCHS):
    for x_mini, y_mini in zip(X_mini_batched, y_mini_batched):
        net.grad_desc(x_mini, y_mini, LEARNING_RATE)

    cst = net.cost(X, y)
    acc = net.accuracy(x_test, y_test)
    print(f"{i}: cost = {cst:.5f}, accuracy = {acc:.3f}")

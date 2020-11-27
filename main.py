import numpy as np

import mnist
import network
import layers

np.random.seed(0)

def accuracy(preds, y):
    pred_labels = preds.argmax(axis=0)
    labels = y.argmax(axis=0) # y: j, n
    return (pred_labels == labels).astype(np.float).mean()

x_train, y_train = mnist.training_images(), mnist.training_labels()
x_test, y_test = mnist.test_images(), mnist.test_labels()

LEARNING_RATE = 0.8
EPOCHS = 10
BATCH_SIZE = 10

net = network.Network([
    layers.Linear(x_train.shape[1], 30),
    layers.Sigmoid(),
    layers.Linear(30, y_train.shape[1]),
    layers.Sigmoid(),
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

    y_train_hat = net.forward(x_train.T)
    y_test_hat = net.forward(x_test.T)
    cost = net.loss.forward(y_train_hat, y_train.T)
    acc = accuracy(y_test_hat, y_test.T)
    print(f"Epoch {i+1}/{EPOCHS}: cost = {cost:.5f}, accuracy = {acc:.3f}")

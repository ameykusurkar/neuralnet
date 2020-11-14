import torch
import math
import matplotlib.pyplot as plt

import mnist

def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)

def model(xb):
    return log_softmax(xb @ weights + bias)

weights = torch.randn(784, 10) / math.sqrt(784)
weights.requires_grad_()
bias = torch.zeros(10, requires_grad=True)

X, y = mnist.training_images(), mnist.training_labels()
x_train, y_train = torch.tensor(X), torch.tensor(y)

plt.imshow(x_train[0].reshape((28, 28)), cmap="gray")
plt.show()
print(y_train.min(), y_train.max())

import numpy as np

def training_images():
    with open("data/mnist/training_images", "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data[16:].astype(np.float64).reshape(60000, 28 * 28)

def training_labels():
    with open("data/mnist/training_labels", "rb") as f:
        labels = np.frombuffer(f.read(), dtype=np.uint8)[8:]
        one_hot = np.zeros((60000, 10))
        one_hot[np.arange(60000), labels] = 1.0
        return one_hot

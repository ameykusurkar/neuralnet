import numpy as np

def training_images(with_validation_set=False):
    with open("data/mnist/training_images", "rb") as f:
        raw_data = np.frombuffer(f.read(), dtype=np.uint8)
        training_data = raw_data[16:].astype(np.float64).reshape(60000, 28 * 28)
        if with_validation_set:
            return tuple(np.split(training_data, [50000]))
        else:
            return training_data

def training_labels(with_validation_set=False):
    with open("data/mnist/training_labels", "rb") as f:
        labels = np.frombuffer(f.read(), dtype=np.uint8)[8:]
        one_hot = np.zeros((60000, 10))
        one_hot[np.arange(60000), labels] = 1.0
        if with_validation_set:
            return tuple(np.split(one_hot, [50000]))
        else:
            return one_hot

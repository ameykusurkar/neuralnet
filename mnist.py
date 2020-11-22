import numpy as np

def training_images(with_validation_set=False):
    with open("data/mnist/training_images", "rb") as f:
        raw_data = np.frombuffer(f.read(), dtype=np.uint8)
        data = raw_data[16:].astype(np.float64).reshape(60000, 28 * 28)
        data = normalise(data)
        if with_validation_set:
            return tuple(np.split(data, [50000]))
        else:
            return data

def training_labels(with_validation_set=False):
    with open("data/mnist/training_labels", "rb") as f:
        labels = np.frombuffer(f.read(), dtype=np.uint8)[8:]
        one_hot = np.zeros((60000, 10))
        one_hot[np.arange(60000), labels] = 1.0
        if with_validation_set:
            return tuple(np.split(one_hot, [50000]))
        else:
            return one_hot

def test_images():
    with open("data/mnist/test_images", "rb") as f:
        raw_data = np.frombuffer(f.read(), dtype=np.uint8)
        data = raw_data[16:].astype(np.float64).reshape(10000, 28 * 28)
        return normalise(data)

def test_labels():
    with open("data/mnist/test_labels", "rb") as f:
        labels = np.frombuffer(f.read(), dtype=np.uint8)[8:]
        one_hot = np.zeros((10000, 10))
        one_hot[np.arange(10000), labels] = 1.0
        return one_hot

def normalise(x):
    """
    Transform `x` to have mean 0 and std dev 1
    """
    m, s = x.mean(), x.std()
    return (x - m) / s

import numpy as np
from urllib import request
import gzip
import pickle

# https://github.com/hsjeong5/MNIST-for-Numpy/blob/master/mnist.py

filename = [
["training_images","train-images-idx3-ubyte.gz"],
["test_images","t10k-images-idx3-ubyte.gz"],
["training_labels","train-labels-idx1-ubyte.gz"],
["test_labels","t10k-labels-idx1-ubyte.gz"]
]


def download_mnist():
    base_url = "http://yann.lecun.com/exdb/mnist/"
    for name in filename:
        print("Downloading " + name[1] + "...")
        request.urlretrieve(base_url + name[1], name[1])
    print("Download complete.")


def save_mnist():
    mnist = {}
    for name in filename[:2]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28*28)
    for name in filename[-2:]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
    with open("mnist.pkl", 'wb') as f:
        pickle.dump(mnist, f)
    print("Save complete.")


def init():
    download_mnist()
    save_mnist()


def load(path="data/"):
    with open(path + "mnist.pkl", 'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]


def load_tensor(num_per_class_train=100, num_per_class_test=10, num_classes=10, path="data/"):
    x_train, t_train, x_test, t_test = load(path)

    training_data = np.empty([0, 28, 28])
    test_data = np.empty([0, 28, 28])
    for i in range(num_classes):
        class_data = x_train[t_train == i, :].reshape([-1, 28, 28])
        training_data = np.concatenate((training_data, class_data[:num_per_class_train]), axis=0)

        class_data = x_test[t_test == i, :].reshape([-1, 28, 28])
        test_data = np.concatenate((test_data, class_data[:num_per_class_test]), axis=0)

    training_labels = np.kron(np.arange(num_classes), np.ones(num_per_class_train))
    test_labels = np.kron(np.arange(num_classes), np.ones(num_per_class_test))

    return training_data.transpose([1, 2, 0]), training_labels, test_data.transpose([1, 2, 0]), test_labels


if __name__ == '__main__':
    init()

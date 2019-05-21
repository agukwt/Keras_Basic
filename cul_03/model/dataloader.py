from keras.datasets import mnist
from keras.datasets import cifar10


def load_mnist():
    return mnist.load_data()


def load_cifar10():
    return cifar10.load_data()


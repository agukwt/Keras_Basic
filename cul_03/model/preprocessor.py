import numpy as np
import cv2
from keras.utils import np_utils
from keras.utils import to_categorical


class Preprocess(object):
    pass


class PreprocessAdaptingCifer10toInception3(object):
    def __init__(self, dataset, train_data_size=5000, test_data_size=5000):
        """
        Setting image size for inceptionv3
        Reference
            https://keras.io/applications/#inceptionv3
        """
        self.dataset = dataset
        self.image_shape = (190, 190, 3)
        self.num_classes = 10
        self.train_data_size = train_data_size
        self.test_data_size = test_data_size

    def upscale(self, x, data_size):
        data_upscaled = np.zeros((data_size,
                                  self.image_shape[0],
                                  self.image_shape[1],
                                  self.image_shape[2]))
        for i, img in enumerate(x):
            large_img = cv2.resize(img, dsize=(self.image_shape[0],
                                               self.image_shape[1]), )
            data_upscaled[i] = large_img
        return data_upscaled

    def get_batch(self):
        (x_train, y_train), (x_test, y_test) = self.dataset

        x_train = x_train[:self.train_data_size]
        y_train = y_train[:self.train_data_size]
        x_test = x_test[:self.test_data_size]
        y_test = y_test[:self.test_data_size]
        x_train = self.upscale(x_train, x_train.shape[0])
        x_test = self.upscale(x_test, x_test.shape[0])

        x_train, x_test = [self.preprocess(d) for d in [x_train, x_test]]
        y_train, y_test = [self.preprocess(d, True) for d in [y_train, y_test]]

        return x_train, y_train, x_test, y_test

    def preprocess(self, data, label_data=False):
        if label_data:
            # convert class vectors to binary class matrices
            data = to_categorical(data, self.num_classes)
        else:
            data = data.astype("float32")
            data /= 255  # convert the value to 0~1 scale
            shape = (data.shape[0],) + self.image_shape  # add dataset length
            data = data.reshape(shape)

        return data


class CleanPreprocess(object):
    def __init__(self, dataset):
        self.NB_CLASSES = 10
        self.shape = None

        (self.X_train, self.y_train), (self.X_test, self.y_test) = dataset

    def preprocess_mnist_common(self):
        NB_TRAIN = 60000
        NB_TEST = 10000

        # convert class vectors to binary class matrices
        self.y_train = np_utils.to_categorical(self.y_train, self.NB_CLASSES)
        self.y_test = np_utils.to_categorical(self.y_test, self.NB_CLASSES)

        self.X_train = self.X_train.astype('float32')
        self.X_test = self.X_test.astype('float32')
        # normalize
        self.X_train /= 255
        self.X_test /= 255

    def preprocess_mnist_dnn(self):
        self.shape = 784

        self.preprocess_mnist_common()

        self.X_train = self.X_train.reshape(len(self.X_train), self.shape)
        self.X_test = self.X_test.reshape(len(self.X_test), self.shape)

        return self.X_train, self.y_train, self.X_test, self.y_test

    def preprocess_mnist_cnn(self):
        self.shape = (28, 28, 1)

        self.preprocess_mnist_common()

        self.X_train = self.X_train.reshape(((len(self.X_train),) + self.shape))
        self.X_test = self.X_test.reshape(((len(self.X_test),) + self.shape))

        return self.X_train, self.y_train, self.X_test, self.y_test

    def preprocess_cifer10(self):
        self.shape = (32, 32, 3)

        def preprocess(data, label_data=False):
            if label_data:
                # convert class vectors to binary class matrices
                data = to_categorical(data, self.NB_CLASSES)
            else:
                data = data.astype("float32")
                data /= 255  # convert the value to 0~1 scale
                shape = (data.shape[0],) + self.shape  # add dataset length
                data = data.reshape(shape)

            return data

        self.X_train, self.X_test = [preprocess(d)
                                     for d in [self.X_train, self.X_test]]
        self.y_train, self.y_test = [preprocess(d, label_data=True)
                                     for d in [self.y_train, self.y_test]]

        return self.X_train, self.y_train, self.X_test, self.y_test

# -*- coding: utf-8 -*-

import abc
import os
import numpy as np
import logging
from time import gmtime, strftime

import logging
from keras.datasets import mnist
from keras.models import Sequential
from keras.initializers import TruncatedNormal
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.utils import np_utils
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping


""" log settings    """
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s: %(message)s')


def load_mnist_data():
    return mnist.load_data()


class MnistNeuralNetwork(metaclass=abc.ABCMeta):
    def __init__(self, dataset):
        (self.X_train, self.y_train), (self.X_test, self.y_test) = dataset

        logging.debug('NB of X train\'s Feature map is {}.'.format(len(self.X_train)))
        logging.debug('NB of y train\'s Feature map is {}.'.format(len(self.y_train)))
        logging.debug('NB of X test\'s Feature map is {}.'.format(len(self.X_test)))
        logging.debug('NB of y test\'s Feature map is {}.'.format(len(self.y_test)))

        logging.debug('Shape of X train\'s Feature map is {}.'.format(self.X_train.shape))
        logging.debug('Shape of y train\'s Feature map is {}.'.format(self.y_train.shape))
        logging.debug('Shape of X test\'s Feature map is {}.'.format(self.X_test.shape))
        logging.debug('Shape of y test\'s Feature map is {}.'.format(self.y_test.shape))

    @abc.abstractmethod
    def preprocess(self):
        pass


class MnistCNN(MnistNeuralNetwork):
    def preprocess(self):
        SHAPED = (28, 28, 1)
        NB_CLASSES = 10
        NB_TRAIN = 60000
        NB_TEST = 10000

        # convert class vectors to binary class matrices
        self.y_train = np_utils.to_categorical(self.y_train, NB_CLASSES)
        self.y_test = np_utils.to_categorical(self.y_test, NB_CLASSES)

        self.X_train = self.X_train.astype('float32')
        self.X_test = self.X_test.astype('float32')
        self.X_train /= 255
        self.X_test /= 255
        self.X_train = self.X_train.reshape(((NB_TRAIN,) + SHAPED))
        self.X_test = self.X_test.reshape(((NB_TEST,) + SHAPED))

        return self.X_train, self.y_train, self.X_test, self.y_test


class MnistDNN(MnistNeuralNetwork):
    def preprocess(self):
        RESHAPED = 784
        NB_CLASSES = 10
        NB_TRAIN = 60000
        NB_TEST = 10000

        # convert class vectors to binary class matrices
        self.y_train = np_utils.to_categorical(self.y_train, NB_CLASSES)
        self.y_test = np_utils.to_categorical(self.y_test, NB_CLASSES)

        self.X_train = self.X_train.astype('float32')
        self.X_test = self.X_test.astype('float32')
        # normalize
        self.X_train /= 255
        self.X_test /= 255

        self.X_train = self.X_train.reshape(NB_TRAIN, RESHAPED)
        self.X_test = self.X_test.reshape(NB_TEST, RESHAPED)

        return self.X_train, self.y_train, self.X_test, self.y_test


class NetworkModeler(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def create_fixed_model(self):
        pass


class LeNetModeler(NetworkModeler):
    def __init__(self):
        pass

    def create_xxx_model(self):
        pass

    def create_fixed_model(self):
        # 10 outputs
        # final stage is softmax
        input_shape = (28, 28, 1)
        num_classes = 10

        ACTIVATION = 'relu'
        NB_CLASSES = 10  # number of outputs = number of digits
        DROPOUT = 0.2

        model = Sequential()

        # extract image features by convolution and max pooling layers
        model.add(Conv2D(20,  # next 特徴量マップの深さ(Chanel)数
                         kernel_size=5,  # 特徴量マップの1辺のサイズ
                         padding="same",  # サイズ差の埋め方：same=>0埋め
                         input_shape=input_shape,  # 入力次元情報
                         activation=ACTIVATION))  # 活性化方法
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(50, kernel_size=5, padding="same", activation=ACTIVATION))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # classify the class by fully-connected layers
        model.add(Flatten())
        model.add(Dense(500, activation="relu"))
        model.add(Dense(num_classes))
        model.add(Activation("softmax"))

        # model.summary()

        return model


class DNNModeler(NetworkModeler):
    def __init__(self):
        self.N_HIDDEN = 128  # 隠れ層のニューロン数
        self.INPUT_SHAPE = 784  # input shape
        self.ACTIVATION = 'relu'
        self.NB_CLASSES = 10  # number of outputs = number of digits
        self.DROPOUT = 0.2

    def create_xxx_model(self):
        pass

    def create_multilayered_model(self, Hidden_Layers_Neuron_List):
        if Hidden_Layers_Neuron_List:
            model = Sequential()

            layers_neuron_list = Hidden_Layers_Neuron_List
            list_input_dim = ([self.INPUT_SHAPE] + layers_neuron_list)[:-1]
            # list_input_dim    : ([n_in] + n_hiddens)[:-1]
            #                     [784, 200, 200, 200] -> [784, 200, 200]

            for i, input_dim in enumerate(list_input_dim):
                model.add(Dense(layers_neuron_list[i],
                                input_dim=input_dim,
                                kernel_initializer=TruncatedNormal(stddev=0.01)))
                model.add(Activation(self.ACTIVATION))
                model.add(Dropout(self.DROPOUT))

            model.add(Dense(self.NB_CLASSES, kernel_initializer=TruncatedNormal(stddev=0.01)))
            model.add(Activation('softmax'))

            return model

        else:
            return ValueError

    def create_fixed_model(self):
        return self.create_multilayered_model([128, 128, 128])


class Trainer(object):
    def __init__(self, model, dir_nm="mnist_log", loss="categorical_crossentropy", optimizer=Adam()):
        self._train_model = model
        self.loss = loss
        self.optimizer = optimizer

        self._train_model.compile(loss=self.loss, optimizer=self.optimizer, metrics=["accuracy"])
        self.verbose = 1
        logdir = dir_nm
        self.log_dir = os.path.join(os.path.dirname(__file__), logdir)
        # print('log_dir: {}'.format(self.log_dir))
        self._train_model_file_name = "model_file.hdf5"

    def train(self, x_train, y_train, batch_size=128, epochs=20, validation_split=0.2):
        if os.path.exists(self.log_dir):
            import shutil
            shutil.rmtree(self.log_dir)  # remove previous execution
        os.mkdir(self.log_dir)

        model_path = os.path.join(self.log_dir, self._train_model_file_name)
        self._train_model.fit(x_train, y_train,
                              batch_size=batch_size,
                              epochs=epochs,
                              validation_split=validation_split,
                              callbacks=[TensorBoard(log_dir=self.log_dir),
                                         ModelCheckpoint(model_path, save_best_only=True),
                                         EarlyStopping(monitor='val_loss', patience=10, verbose=1)],
                              verbose=self.verbose)


def evaluate_test(model, x_test, y_test):
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

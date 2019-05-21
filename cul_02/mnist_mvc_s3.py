# -*- coding: utf-8 -*-

import os
import numpy as np
import logging
from time import gmtime, strftime

import logging
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.utils import np_utils
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping


""" log settings    """
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s :%(message)s')


def dnnet_model():
    # 10 outputs
    # final stage is softmax

    N_HIDDEN = 128      # 隠れ層のニューロン数
    RESHAPED = 784      # input shape
    ACTIVATION = 'relu'
    NB_CLASSES = 10     # number of outputs = number of digits
    DROPOUT = 0.2

    model = Sequential()
    model.add(Dense(N_HIDDEN, input_shape=(RESHAPED,)))
    model.add(BatchNormalization())
    model.add(Activation(ACTIVATION))
    model.add(Dropout(DROPOUT))

    model.add(Dense(N_HIDDEN))
    model.add(BatchNormalization())
    model.add(Activation(ACTIVATION))
    model.add(Dropout(DROPOUT))

    model.add(Dense(NB_CLASSES))
    model.add(Activation('softmax'))

    model.summary()

    return model


def lenet_model(input_shape=(28, 28, 1), num_classes=10):
    # 10 outputs
    # final stage is softmax

    ACTIVATION = 'relu'
    NB_CLASSES = 10     # number of outputs = number of digits
    DROPOUT = 0.2

    model = Sequential()

    # extract image features by convolution and max pooling layers
    model.add(Conv2D(20,                            # next 特徴量マップの深さ(Chanel)数
                     kernel_size=5,                 # 特徴量マップの1辺のサイズ
                     padding="same",                # サイズ差の埋め方：same=>0埋め
                     input_shape=input_shape,       # 入力次元情報
                     activation=ACTIVATION))        # 活性化方法
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(50, kernel_size=5, padding="same", activation=ACTIVATION))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # classify the class by fully-connected layers
    model.add(Flatten())
    model.add(Dense(500, activation="relu"))
    model.add(Dense(num_classes))
    model.add(Activation("softmax"))

    return model


class MNISTDatasetDNN(object):
    def __init__(self):
        (self.X_train, self.y_train), (self.X_test, self.y_test) = mnist.load_data()

    def preprocess(self, seed=1234):
        RESHAPED = 784
        NB_CLASSES = 10
        NB_TRAIN = 60000
        NB_TEST = 10000

        self.X_train = self.X_train.reshape(NB_TRAIN, RESHAPED)
        self.X_test = self.X_test.reshape(NB_TEST, RESHAPED)
        self.X_train = self.X_train.astype('float32')
        self.X_test = self.X_test.astype('float32')

        # normalize
        self.X_train /= 255
        self.X_test /= 255

        # convert class vectors to binary class matrices
        self.y_train = np_utils.to_categorical(self.y_train, NB_CLASSES)
        self.y_test = np_utils.to_categorical(self.y_test, NB_CLASSES)

        return self.X_train, self.y_train, self.X_test, self.y_test

class MNISTDatasetCNN(object):
    def __init__(self):
        (self.X_train, self.y_train), (self.X_test, self.y_test) = mnist.load_data()

    def preprocess(self, seed=1234):
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
        trin_dim = ((NB_TRAIN,), SHAPED)
        test_dim = ((NB_TRAIN,), SHAPED)
        self.X_train = self.X_train.reshape(((NB_TRAIN,) + SHAPED))
        self.X_test = self.X_test.reshape(((NB_TEST,) + SHAPED))

        return self.X_train, self.y_train, self.X_test, self.y_test


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

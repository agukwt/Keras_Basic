# -*- coding: utf-8 -*-

import os
import numpy as np
import logging
from time import gmtime, strftime

import logging
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
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


class Trainer(object):
    def __init__(self, model, loss="categorical_crossentropy", optimizer=Adam()):
        self._train_model = model
        self.loss = loss
        self.optimizer = optimizer

        self._train_model.compile(loss=self.loss, optimizer=self.optimizer, metrics=["accuracy"])
        self.verbose = 1
        logdir = "mnist_log"
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

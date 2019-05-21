# -*- coding: utf-8 -*-

import abc
import os
import sys
import logging

from pathlib import Path
import numpy as np
import logging
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras import regularizers

from make_tensorboard import make_tensorboard
from util import io


""" log settings    """
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s :%(message)s')


class DeepLearningModeler(object):
    logging.debug('DeepLearningModeler Start')

    # Reproducibility model
    RANDOM_SEED = 1024
    MODEL_LOG_DIR = 'model_log'

    # network and training
    RESHAPED = 784
    NB_TEST = 10000             # テスト用(秘匿)
    NB_TRAIN = 60000            # 学習用(検証用含む)
    NB_EPOCH = 5                # エポック回数
    BATCH_SIZE = 128            # バッチサイズ
    VERBOSE = 1                 # log revel
    NB_CLASSES = 10             # number of outputs = number of digits
    OPTIMIZER = Adam(lr=0.01, beta_1=0.9, beta_2=0.999)  # Adam optimizer
    N_HIDDEN = 128              # 隠れ層のニューロン数
    VALIDATION_SPLIT = 0.2      # how much TRAIN is reserved for VALIDATION
    DROPOUT = 0.3               #
    ACTIVATION = 'relu'         #

    def __init__(self):
        logging.debug('DeepLearningModeler.__init__ Start')

        print(os.path.dirname(__file__))

        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None
        self.y_train = None
        self.y_test = None

        self.model = None

        try:
            __model_log_dir = r'src/cultivation/' + DeepLearningModeler.MODEL_LOG_DIR
            os.path.isdir(__model_log_dir)
            logging.debug('model_log_dir = {}'.format(__model_log_dir))

        except IOError:
            print('MODEL_LOG_DIR is not found.')
            logging.critical('Fail to load MODEL_LOG_DIR.')
            sys.exit(1)

        logging.debug('DeepLearningModeler.__init__ End')

    def __del__(self):
        logging.debug('DeepLearningModeler End')

    def load_mnist_data(self):
        """ define data set  """

        logging.debug('DeepLearningModeler.load_mnist_data Start')
        # data: shuffled and split between train and test sets
        (self.X_train, self.y_train), (self.X_test, self.y_test) = mnist.load_data()

        # X_train is 60000 rows of 28x28 values --> reshaped in 60000 x 784
        self.X_train = self.X_train.reshape(DeepLearningModeler.NB_TRAIN,
                                            DeepLearningModeler.RESHAPED)
        self.X_test = self.X_test.reshape(DeepLearningModeler.NB_TEST,
                                          DeepLearningModeler.RESHAPED)
        self.X_train = self.X_train.astype('float32')
        self.X_test = self.X_test.astype('float32')

        # normalize
        self.X_train /= 255
        self.X_test /= 255
        logging.debug('train samples: {}'.format(self.X_train.shape[0]))
        logging.debug('test samples: {}'.format(self.X_test.shape[0]))

        # convert class vectors to binary class matrices
        self.Y_train = np_utils.to_categorical(
            self.y_train, DeepLearningModeler.NB_CLASSES)
        self.Y_test = np_utils.to_categorical(
            self.y_test, DeepLearningModeler.NB_CLASSES)

        logging.debug('DeepLearningModeler.load_mnist_data End')

    def define_model(self):
        pass

    def define_model_by_sequence(self):
        logging.debug('DeepLearningModeler.define_model_by_sequence Start')
        # 10 outputs
        # final stage is softmax

        self.model = Sequential()
        # l1: 728 -> 128
        self.model.add(Dense(DeepLearningModeler.N_HIDDEN,
                             input_shape=(DeepLearningModeler.RESHAPED,)))
        self.model.add(BatchNormalization())
        self.model.add(Activation(DeepLearningModeler.ACTIVATION))
        self.model.add(Dropout(DeepLearningModeler.DROPOUT))

        # l2: 128 -> 128
        self.model.add(Dense(DeepLearningModeler.N_HIDDEN))
        self.model.add(BatchNormalization())
        self.model.add(Activation(DeepLearningModeler.ACTIVATION))
        self.model.add(Dropout(DeepLearningModeler.DROPOUT))

        # l3: 128 -> 10
        self.model.add(Dense(DeepLearningModeler.NB_CLASSES))
        self.model.add(Activation('softmax'))

        self.model.summary()

        logging.debug('DeepLearningModeler.define_model_by_sequence End')

    def define_model_by_repetition(self):
        pass

    def compile_model(self):
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=DeepLearningModeler.OPTIMIZER,
                           metrics=['accuracy'],
                           )


class MNISTDatasetCNN(object):
    def __init__(self):
        self.image_shape = (28, 28, 1)  # image is 28x28x1 (grayscale)
        self.num_classes = 10

    def get_batch(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train, x_test = [self.preprocess(d) for d in [x_train, x_test]]
        y_train, y_test = [self.preprocess(d, label_data=True) for d in
                           [y_train, y_test]]

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
    
class MNISTDatasetDNN(object):
    def __init__(self):
        self.image_shape = 728  # image is 28x28x1 (grayscale)
        self.num_classes = 10

        self.RESHAPED = 784
        self.NB_TEST = 10000  # テスト用(秘匿)
        self.NB_TRAIN = 60000  # 学習用(検証用含む)

    def get_batch(self):
        # data: shuffled and split between train and test sets
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # X_train is 60000 rows of 28x28 values --> reshaped in 60000 x 784
        x_train = x_train.reshape(self.NB_TRAIN, self.RESHAPED)
        x_test = x_test.reshape(self.NB_TEST, self.RESHAPED)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        # normalize
        x_train /= 255
        x_test /= 255

        # convert class vectors to binary class matrices
        y_train = to_categorical(y_train, self.NB_CLASSES)
        y_test = to_categorical(y_test, self.NB_CLASSES)

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


def dnnet(input_shape, num_classes):
    # network and training
    RESHAPED = input_shape
    NB_CLASSES = num_classes    # number of outputs = number of digits
    N_HIDDEN = 128              # 隠れ層のニューロン数
    DROPOUT = 0.3               #
    ACTIVATION = 'relu'         #

    # Sequential design
    model = Sequential()

    # l1: 728 -> 128
    model.add(N_HIDDEN, input_shape=(RESHAPED,))
    model.add(BatchNormalization())
    model.add(Activation(ACTIVATION))
    model.add(Dropout(DROPOUT))

    # l2: 128 -> 128
    model.add(Dense(N_HIDDEN))
    model.add(BatchNormalization())
    model.add(Activation(ACTIVATION))
    model.add(Dropout(DROPOUT))

    # l3: 128 -> 10
    model.add(Dense(NB_CLASSES))
    model.add(Activation('softmax'))

    model.summary()

    return model


class Trainer(object):
    def __init__(self, model, loss, optimizer):
        self._target = model
        self._target.compile(
            loss=loss, optimizer=optimizer, metrics=["accuracy"]
            )
        self.verbose = 1
        logdir = "logdir_lenet"
        self.log_dir = os.path.join(os.path.dirname(__file__), logdir)

    def train(self, x_train, y_train, batch_size, epochs, validation_split):
        if os.path.exists(self.log_dir):
            import shutil
            shutil.rmtree(self.log_dir)  # remove previous execution
        os.mkdir(self.log_dir)

        self._target.fit(
            x_train, y_train,
            batch_size=batch_size, epochs=epochs,
            validation_split=validation_split,
            callbacks=[TensorBoard(log_dir=self.log_dir)],
            verbose=self.verbose
        )




def lenet(input_shape, num_classes):
    model = Sequential()

    # extract image features by convolution and max pooling layers
    model.add(Conv2D(
        20, kernel_size=5, padding="same",
        input_shape=input_shape, activation="relu"
    ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(50, kernel_size=5, padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # classify the class by fully-connected layers
    model.add(Flatten())
    model.add(Dense(500, activation="relu"))
    model.add(Dense(num_classes))
    model.add(Activation("softmax"))
    return model








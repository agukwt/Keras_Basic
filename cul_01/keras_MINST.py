# -*- coding: utf-8 -*-

import numpy as np
import logging
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras import regularizers
import matplotlib.pyplot as plt
from make_tensorboard import make_tensorboard


""" log settings    """
logging.basicConfig(level=logging.DEBUG, format='%(message)s')

""" initialize param    """
# for reproducibility
np.random.seed(1671)

# network and training
NB_EPOCH = 5          # エポック回数
BATCH_SIZE = 128        # バッチサイズ
VERBOSE = 1             #
NB_CLASSES = 10         # number of outputs = number of digits
OPTIMIZER = Adam(lr=0.01, beta_1=0.9, beta_2=0.999)      # Adam optimizer
N_HIDDEN = 128          # 隠れ層のニューロン数
VALIDATION_SPLIT = 0.2  # how much TRAIN is reserved for VALIDATION
DROPOUT = 0.3
ACTIVATION = 'relu'

""" define data set  """
# data: shuffled and split between train and test sets
# sklearn.datasets.fetch_mldataより、keras.datasets.mnist.load_dataの方が速い。
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# X_train is 60000 rows of 28x28 values --> reshaped in 60000 x 784
RESHAPED = 784
X_train = X_train.reshape(60000, RESHAPED)
X_test = X_test.reshape(10000, RESHAPED)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalize
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(y_test, NB_CLASSES)

""" design model  """
# 10 outputs
# final stage is softmax

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
model.compile(loss='categorical_crossentropy',
              optimizer=OPTIMIZER, metrics=['accuracy'])

callbacks = [make_tensorboard(set_dir_name='keras_MINST')]

""" run model  """
hist = model.fit(X_train, Y_train,
                 batch_size=BATCH_SIZE, epochs=NB_EPOCH,
                 callbacks=callbacks,
                 verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

''' visualization '''
val_acc = hist.history['val_acc']
val_loss = hist.history['val_loss']

train_acc = hist.history['acc']
train_loss = hist.history['loss']


fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10, 4))
fig.patch.set_facecolor('white')
plt.rc('font', family='serif')

axL.plot(range(len(val_loss)), train_loss, marker='.', label='loss')
axL.plot(range(len(val_loss)), val_loss, marker='.', label='val_loss')
axL.legend(loc='best', fontsize=10)
axL.set_title('model loss')
axL.set_xlabel('epoch')
axL.set_ylabel('loss')
axL.grid()

axR.plot(range(len(val_loss)), train_acc, marker='.', label='acc')
axR.plot(range(len(val_loss)), val_acc, marker='.', label='val_acc')
axR.legend(loc='best', fontsize=10)
axR.set_title('model acc')
axR.set_xlabel('epoch')
axR.set_ylabel('acc')
axR.grid()

plt.show()


""" evaluate  """
score = model.evaluate(X_test, Y_test, verbose=VERBOSE)
print("\nTest score:", score[0])
print('Test accuracy:', score[1])






